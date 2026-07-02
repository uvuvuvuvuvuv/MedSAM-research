from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METHOD_DEFAULT = "idea1_iterclean_bank_adaptshape"
MODEL_TYPE = "vit_b"


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_slice_name(item: dict[str, Any]) -> str:
    if item.get("slice_name"):
        return str(item["slice_name"])
    for key in ("teacher_img", "student_img", "teacher_gt", "student_gt"):
        if item.get(key):
            return Path(str(item[key])).name
    raise KeyError(f"Cannot infer slice_name from keys={list(item.keys())}")


def resolve_path(fold_root: Path, item: dict[str, Any], key: str) -> Path:
    value = item.get(key)
    if not value:
        raise KeyError(f"Missing manifest key '{key}' for {get_slice_name(item)}")
    path = Path(str(value))
    if not path.is_absolute():
        path = fold_root / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_image_tensor(path: Path, device: torch.device) -> torch.Tensor:
    image = np.load(path)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Unexpected image shape {image.shape} from {path}")

    image = image.astype(np.float32)
    if image.max() > 2.0:
        image /= 255.0
    image = np.clip(image, 0.0, 1.0)
    return (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )


def _load_gt_array(path: Path) -> np.ndarray:
    gt = np.load(path)
    if gt.ndim == 3:
        if gt.shape[-1] == 1:
            gt = gt[..., 0]
        elif gt.shape[0] == 1:
            gt = gt[0]
        else:
            raise ValueError(f"Unexpected GT shape {gt.shape} from {path}")
    return gt.astype(np.int64)


def _bbox_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x1 >= x2 or y1 >= y2:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 0)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 0)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def load_instance_target(
    path: Path,
    device: torch.device,
    label_id: int,
    bbox: list[float],
    component_id: int | None,
) -> torch.Tensor:
    gt = _load_gt_array(path)
    mask_all = (gt == int(label_id))

    if not mask_all.any():
        raise ValueError(
            f"label_id={label_id} not found in GT: {path}"
        )

    mask_all_uint8 = mask_all.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_all_uint8, connectivity=8
    )

    if num_labels <= 1:
        raise ValueError(
            f"No connected components found for label_id={label_id} "
            f"in {path}"
        )

    if component_id is not None:
        if not isinstance(component_id, int) or component_id < 1:
            raise ValueError(
                f"component_id must be positive int, got {component_id!r} "
                f"for label_id={label_id} in {path}"
            )
        if component_id >= num_labels:
            raise ValueError(
                f"component_id={component_id} out of range "
                f"(found {num_labels - 1} components) "
                f"for label_id={label_id} in {path}"
            )
        selected_label = int(component_id)
        selected_mask = (labels == selected_label)
        if not selected_mask.any():
            raise ValueError(
                f"component_id={component_id} produced empty mask "
                f"for label_id={label_id} in {path}"
            )
    else:
        best_idx: int | None = None
        best_iou = -1.0
        for comp_idx in range(1, num_labels):
            comp_mask = (labels == comp_idx)
            rows, cols = np.where(comp_mask)
            if len(rows) == 0:
                continue
            comp_bbox = [
                float(cols.min()), float(rows.min()),
                float(cols.max()) + 1, float(rows.max()) + 1,
            ]
            iou = _bbox_iou(bbox, comp_bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = comp_idx

        if best_idx is None or best_iou <= 0.0:
            raise ValueError(
                f"No connected component overlaps with bbox={bbox} "
                f"for label_id={label_id} in {path}"
            )
        selected_mask = (labels == best_idx)
        if not selected_mask.any():
            raise ValueError(
                f"bbox-matched component produced empty mask "
                f"for label_id={label_id} in {path}"
            )

    target = selected_mask.astype(np.float32)
    return torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float().to(device)


def build_medsam(checkpoint: Path, device: torch.device):
    from segment_anything import sam_model_registry

    model = sam_model_registry[MODEL_TYPE](checkpoint=str(checkpoint))
    model.to(device)
    return model


def freeze_for_mask_decoder_only(model) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.mask_decoder.parameters():
        parameter.requires_grad = True

    model.image_encoder.eval()
    model.prompt_encoder.eval()
    model.mask_decoder.train()
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def make_box_tensor(box: list[float] | tuple[float, ...], device: torch.device) -> torch.Tensor:
    array = np.asarray(box, dtype=np.float32).reshape(1, 4)
    return torch.from_numpy(array).float().to(device)


def dice_loss_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    target = target.float()

    reduce_dims = tuple(range(1, probability.ndim))

    intersection = (
        probability * target
    ).sum(dim=reduce_dims)

    denominator = (
        probability.square().sum(dim=reduce_dims)
        + target.square().sum(dim=reduce_dims)
    )

    dice = (
        2.0 * intersection + eps
    ) / (
        denominator + eps
    )

    return (1.0 - dice).mean()


def decode_low_res(
    model,
    image_embedding: torch.Tensor,
    sparse_embeddings: torch.Tensor,
    dense_embeddings: torch.Tensor,
) -> torch.Tensor:
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_logits


def update_ema(model, ema_model, decay: float) -> None:
    with torch.no_grad():
        model_state = model.state_dict()
        ema_state = ema_model.state_dict()
        for key, ema_value in ema_state.items():
            model_value = model_state[key].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(decay).add_(model_value, alpha=1.0 - decay)
            else:
                ema_value.copy_(model_value)


def parse_train_entries(
    fold_root: Path,
    manifest: list[dict[str, Any]],
    prompts: dict[str, Any],
    split_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest_by_slice = {get_slice_name(item): item for item in manifest}
    entries: list[dict[str, Any]] = []

    for record in split_records:
        slice_name = str(record["slice_name"])
        label_mode = str(record["label_mode"])
        if label_mode != "full":
            continue
        item = manifest_by_slice.get(slice_name)
        if item is None:
            raise KeyError(
                f"Full record slice_name={slice_name!r} not found "
                f"in manifest"
            )
        prompt_meta = prompts.get(slice_name)
        if prompt_meta is None:
            raise KeyError(
                f"Full record slice_name={slice_name!r} not found "
                f"in prompts"
            )
        if item.get("split") != "train":
            raise RuntimeError(f"Non-train item in supervision split: {slice_name}")

        image_path = resolve_path(fold_root, item, "teacher_img")
        instances = prompt_meta.get("instances")
        if not instances:
            raise ValueError(
                f"Full record slice_name={slice_name!r} has no instances "
                f"in prompts"
            )
        for instance in instances:
            box = instance.get("bbox_teacher", instance.get("bbox"))
            if box is None:
                raise KeyError(
                    f"Instance missing bbox_teacher and bbox: "
                    f"slice_name={slice_name!r} "
                    f"label_id={instance.get('label_id')!r} "
                    f"component_id={instance.get('component_id')!r}"
                )
            entry: dict[str, Any] = {
                "slice_name": slice_name,
                "label_mode": "full",
                "case_id": str(record.get("case_id", "")),
                "label_id": int(instance.get("label_id", 1)),
                "component_id": instance.get("component_id"),
                "bbox": [float(value) for value in box],
                "teacher_img": image_path,
                "teacher_gt": resolve_path(fold_root, item, "teacher_gt"),
            }
            entries.append(entry)

    return entries


def save_checkpoint(path: Path, model, extra: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), **extra}, path)


def train_one_dataset(args: argparse.Namespace, dataset: str) -> None:
    device = torch.device(args.device)
    fold_root = args.processed_root / dataset / args.fold
    meta_dir = fold_root / "meta"

    manifest = load_json(meta_dir / "manifest.json")
    prompts = load_json(fold_root / "prompts" / "prompts_train.json")
    split_records = load_json(
        meta_dir / f"full_box_split_{args.method}.json"
    )

    entries = parse_train_entries(fold_root, manifest, prompts, split_records)
    full_entries = [entry for entry in entries if entry["label_mode"] == "full"]
    if not full_entries:
        raise RuntimeError(f"No full entries for {dataset}")

    print(
        f"[{dataset}] full entries={len(full_entries)}"
    )

    model = build_medsam(args.checkpoint, device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p in ema_model.parameters():
        p.requires_grad = False

    trainable_parameters = freeze_for_mask_decoder_only(model)

    image_encoder_trainable = sum(
        p.numel() for p in model.image_encoder.parameters() if p.requires_grad
    )
    prompt_encoder_trainable = sum(
        p.numel() for p in model.prompt_encoder.parameters() if p.requires_grad
    )
    mask_decoder_trainable = sum(
        p.numel() for p in model.mask_decoder.parameters() if p.requires_grad
    )
    assert image_encoder_trainable == 0, (
        f"image_encoder has {image_encoder_trainable} trainable parameters"
    )
    assert prompt_encoder_trainable == 0, (
        f"prompt_encoder has {prompt_encoder_trainable} trainable parameters"
    )
    assert mask_decoder_trainable > 0, (
        "mask_decoder has no trainable parameters"
    )

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    out_dir = args.out_root / dataset / args.fold
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"medsam_ft_log_{args.method}.csv"
    last_checkpoint = out_dir / "medsam_sac_last.pth"
    ema_checkpoint = out_dir / "medsam_sac_ema.pth"

    log_fields = [
        "epoch",
        "step",
        "dataset",
        "slice_name",
        "label_id",
        "loss",
        "loss_dice",
        "loss_bce",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    global_step = 0
    for epoch in range(args.epochs):
        random.shuffle(full_entries)

        for entry in full_entries:
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            model.mask_decoder.train()
            model.image_encoder.eval()
            model.prompt_encoder.eval()
            ema_model.eval()

            label_id = int(entry["label_id"])
            component_id = entry.get("component_id")

            image = load_image_tensor(entry["teacher_img"], device)
            box_tensor = make_box_tensor(entry["bbox"], device)
            target = load_instance_target(
                entry["teacher_gt"], device, label_id,
                entry["bbox"], component_id,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                image_embedding = model.image_encoder(image)
                sparse_embeddings, dense_embeddings = model.prompt_encoder(
                    points=None,
                    boxes=box_tensor,
                    masks=None,
                )

            low_res_logits = decode_low_res(
                model,
                image_embedding,
                sparse_embeddings,
                dense_embeddings,
            )
            logits_full = F.interpolate(
                low_res_logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            loss_dice = dice_loss_with_logits(
                logits_full,
                target,
            )
            loss_bce = F.binary_cross_entropy_with_logits(
                logits_full, target.float(),
            )
            loss = loss_dice + loss_bce

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                trainable_parameters, max_norm=args.max_grad_norm
            )
            optimizer.step()
            update_ema(model, ema_model, args.ema_decay)
            global_step += 1

            row = {
                "epoch": epoch,
                "step": global_step,
                "dataset": dataset,
                "slice_name": entry["slice_name"],
                "label_id": label_id,
                "loss": float(loss.detach().cpu()),
                "loss_dice": float(loss_dice.detach().cpu()),
                "loss_bce": float(loss_bce.detach().cpu()),
            }
            with log_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=log_fields).writerow(row)

            if global_step % args.log_every == 0:
                print(
                    f"[{dataset}] epoch={epoch} step={global_step} "
                    f"loss={row['loss']:.4f} "
                    f"dice={row['loss_dice']:.4f} "
                    f"bce={row['loss_bce']:.4f}"
                )

        checkpoint_extra = {
            "epoch": epoch,
            "global_step": global_step,
            "method": args.method,
            "trainable": "mask_decoder_only",
        }
        save_checkpoint(last_checkpoint, model, checkpoint_extra)
        save_checkpoint(ema_checkpoint, ema_model, checkpoint_extra)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    num_full_records = sum(
        1 for r in split_records if str(r.get("label_mode")) == "full"
    )
    all_component_ids = [
        e.get("component_id") for e in full_entries
    ]
    has_none = any(c is None for c in all_component_ids)
    has_int = any(isinstance(c, int) for c in all_component_ids)
    if has_none and has_int:
        instance_target_mode = "mixed"
    elif has_none:
        instance_target_mode = "bbox_matching"
    else:
        instance_target_mode = "component_id"

    summary = {
        "dataset": dataset,
        "fold": args.fold,
        "method": args.method,
        "epochs": args.epochs,
        "global_step": global_step,
        "training_mode": "full_only",
        "trainable": "mask_decoder_only",
        "loss": "dice_plus_bce",
        "loss_bce_variant": "binary_cross_entropy_with_logits",
        "num_full_records": num_full_records,
        "num_full_instances": len(full_entries),
        "num_full_entries": len(full_entries),
        "num_missing_manifest": 0,
        "num_missing_prompts": 0,
        "num_empty_targets": 0,
        "ema_enabled": True,
        "ema_decay": args.ema_decay,
        "ema_num_updates": global_step,
        "ema_checkpoint_role": "stabilized_pseudo_inference",
        "instance_target_mode": instance_target_mode,
        "checkpoint_last": str(last_checkpoint),
        "checkpoint_ema": str(ema_checkpoint),
        "log_path": str(log_path),
    }
    save_json(summary, out_dir / f"medsam_ft_summary_{args.method}.json")

    print(f"[OK] training completed: {dataset}")
    print(f"     last = {last_checkpoint}")
    print(f"     ema  = {ema_checkpoint}")
    print(f"     log  = {log_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Full-only clean MedSAM mask-decoder fine-tuning"
    )
    parser.add_argument("--processed_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--out_root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--log_every", type=int, default=10)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be > 0, got {args.epochs}")
    if args.max_steps < 0:
        raise ValueError(f"--max_steps must be >= 0, got {args.max_steps}")
    if args.lr <= 0:
        raise ValueError(f"--lr must be > 0, got {args.lr}")
    if args.weight_decay < 0:
        raise ValueError(
            f"--weight_decay must be >= 0, got {args.weight_decay}"
        )
    if not (0 <= args.ema_decay < 1):
        raise ValueError(
            f"--ema_decay must be in [0, 1), got {args.ema_decay}"
        )
    if args.max_grad_norm <= 0:
        raise ValueError(
            f"--max_grad_norm must be > 0, got {args.max_grad_norm}"
        )
    if args.log_every <= 0:
        raise ValueError(
            f"--log_every must be > 0, got {args.log_every}"
        )
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )
    if not args.processed_root.is_dir():
        raise FileNotFoundError(
            f"processed_root not found: {args.processed_root}"
        )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; falling back to CPU")
        args.device = "cpu"

    set_seed(args.seed)
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not datasets:
        raise ValueError("--datasets is empty")

    for dataset in datasets:
        train_one_dataset(args, dataset)


if __name__ == "__main__":
    main()
