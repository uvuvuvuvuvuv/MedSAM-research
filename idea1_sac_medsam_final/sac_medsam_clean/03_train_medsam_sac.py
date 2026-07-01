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

import numpy as np
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METHOD_DEFAULT = "idea1_sac_medsam_final"
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


def load_full_target(
    path: Path,
    device: torch.device,
    label_id: int,
) -> torch.Tensor:
    gt = np.load(path)
    if gt.ndim == 3:
        if gt.shape[-1] == 1:
            gt = gt[..., 0]
        elif gt.shape[0] == 1:
            gt = gt[0]
        else:
            raise ValueError(f"Unexpected GT shape {gt.shape} from {path}")
    target = (gt.astype(np.int64) == int(label_id)).astype(np.float32)
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


def make_box_mask(
    box: list[float] | tuple[float, ...],
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0, min(width - 1, int(np.floor(x1))))
    y1 = max(0, min(height - 1, int(np.floor(y1))))
    x2 = max(0, min(width, int(np.ceil(x2))))
    y2 = max(0, min(height, int(np.ceil(y2))))

    mask = torch.zeros((1, 1, height, width), dtype=torch.float32, device=device)
    if x2 > x1 and y2 > y1:
        mask[:, :, y1:y2, x1:x2] = 1.0
    return mask


def dice_loss(probability: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = probability.float()
    target = target.float()
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denominator = (
        probability.sum(dim=(1, 2, 3))
        + target.sum(dim=(1, 2, 3))
        + eps
    )
    dice = (2.0 * intersection + eps) / denominator
    return 1.0 - dice.mean()


def weighted_bce(
    probability: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    probability = probability.clamp(eps, 1.0 - eps)
    loss_map = F.binary_cross_entropy(
        probability, target.float(), reduction="none"
    )
    if weight is None:
        return loss_map.mean()
    weighted = loss_map * weight.float()
    return weighted.sum() / weight.float().sum().clamp_min(1.0)


def tv_smooth_loss(probability: torch.Tensor) -> torch.Tensor:
    dy = torch.abs(probability[:, :, 1:, :] - probability[:, :, :-1, :]).mean()
    dx = torch.abs(probability[:, :, :, 1:] - probability[:, :, :, :-1]).mean()
    return dx + dy


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


def compute_qf_from_embedding(
    image_embedding: torch.Tensor,
    proto_fg: np.ndarray,
    proto_bg: np.ndarray,
    output_hw: tuple[int, int],
    temperature: float = 0.07,
) -> torch.Tensor:
    feature = F.normalize(image_embedding.detach(), dim=1)
    batch, channels, height, width = feature.shape
    if batch != 1:
        raise ValueError(f"Expected batch size 1, got {batch}")

    feature_flat = feature.permute(0, 2, 3, 1).reshape(-1, channels)
    foreground = F.normalize(
        torch.from_numpy(proto_fg).float().to(feature.device), dim=1
    )
    background = F.normalize(
        torch.from_numpy(proto_bg).float().to(feature.device), dim=1
    )

    max_fg = (feature_flat @ foreground.t()).max(dim=1).values
    max_bg = (feature_flat @ background.t()).max(dim=1).values
    logits = torch.stack([max_fg, max_bg], dim=1) / temperature
    qf = torch.softmax(logits, dim=1)[:, 0].reshape(1, 1, height, width)
    return F.interpolate(
        qf,
        size=output_hw,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0).detach()


def build_shape_map(
    shape_a: np.ndarray,
    box: list[float] | tuple[float, ...],
    output_height: int,
    output_width: int,
    device: torch.device,
) -> torch.Tensor:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0, min(output_width - 1, int(np.floor(x1))))
    y1 = max(0, min(output_height - 1, int(np.floor(y1))))
    x2 = max(0, min(output_width, int(np.ceil(x2))))
    y2 = max(0, min(output_height, int(np.ceil(y2))))

    output = torch.zeros(
        (1, 1, output_height, output_width),
        dtype=torch.float32,
        device=device,
    )
    if x2 <= x1 or y2 <= y1:
        return output

    template = (
        torch.from_numpy(shape_a)
        .float()
        .to(device)
        .reshape(1, 1, 64, 64)
    )
    template = F.interpolate(
        template,
        size=(y2 - y1, x2 - x1),
        mode="bilinear",
        align_corners=False,
    )
    output[:, :, y1:y2, x1:x2] = template
    return output.clamp(0.0, 1.0)


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
        item = manifest_by_slice.get(slice_name)
        prompt_meta = prompts.get(slice_name)
        if item is None or prompt_meta is None:
            continue
        if item.get("split") != "train":
            raise RuntimeError(f"Non-train item in supervision split: {slice_name}")

        image_path = resolve_path(fold_root, item, "teacher_img")
        instances = prompt_meta.get("instances", [])
        for instance in instances:
            box = instance.get("bbox_teacher", instance.get("bbox"))
            if box is None:
                continue
            label_mode = str(record["label_mode"])
            entry: dict[str, Any] = {
                "slice_name": slice_name,
                "label_mode": label_mode,
                "case_id": str(record.get("case_id", "")),
                "label_id": int(instance.get("label_id", 1)),
                "bbox": [float(value) for value in box],
                "teacher_img": image_path,
            }
            # Full samples alone receive a GT path. Box entries do not load GT.
            if label_mode == "full":
                entry["teacher_gt"] = resolve_path(fold_root, item, "teacher_gt")
            entries.append(entry)

    return entries


def build_mixed_schedule(
    full_entries: list[dict[str, Any]],
    box_entries: list[dict[str, Any]],
    boxes_per_full: int = 3,
) -> list[dict[str, Any]]:
    """Interleave entries without repeating or dropping any sample."""
    schedule: list[dict[str, Any]] = []
    full_index = 0
    box_index = 0

    while full_index < len(full_entries) or box_index < len(box_entries):
        if full_index < len(full_entries):
            schedule.append(full_entries[full_index])
            full_index += 1
        for _ in range(boxes_per_full):
            if box_index < len(box_entries):
                schedule.append(box_entries[box_index])
                box_index += 1
    return schedule


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
    template_path = meta_dir / f"support_template_{args.method}.npz"
    template = np.load(template_path)

    entries = parse_train_entries(fold_root, manifest, prompts, split_records)
    full_entries = [entry for entry in entries if entry["label_mode"] == "full"]
    box_entries = [entry for entry in entries if entry["label_mode"] == "box"]
    if not full_entries:
        raise RuntimeError(f"No full entries for {dataset}")
    if not box_entries:
        raise RuntimeError(f"No box entries for {dataset}")

    print(
        f"[{dataset}] instance entries: full={len(full_entries)} "
        f"box={len(box_entries)} total={len(entries)}"
    )

    model = build_medsam(args.checkpoint, device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()

    trainable_parameters = freeze_for_mask_decoder_only(model)
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
        "label_mode",
        "label_id",
        "loss",
        "loss_full",
        "loss_out",
        "loss_seed",
        "loss_wac",
        "loss_proto",
        "loss_smooth",
        "mean_p",
        "mean_qf",
        "weak1_sum",
        "weak2_sum",
        "strong_sum",
        "gate1_sum",
        "gate2_sum",
        "box_area_low",
        "seed_sum",
        "unc_sum",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    global_step = 0
    for epoch in range(args.epochs):
        random.shuffle(full_entries)
        random.shuffle(box_entries)
        schedule = build_mixed_schedule(full_entries, box_entries, boxes_per_full=3)

        for entry in schedule:
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            model.mask_decoder.train()
            model.image_encoder.eval()
            model.prompt_encoder.eval()
            ema_model.eval()

            label_id = int(entry["label_id"])
            foreground_key = f"proto_fg_c{label_id}"
            background_key = f"proto_bg_c{label_id}"
            shape_key = f"shape_A_c{label_id}"
            for key in (foreground_key, background_key, shape_key):
                if key not in template:
                    raise KeyError(
                        f"Support template missing '{key}' for {dataset}, "
                        f"slice={entry['slice_name']}"
                    )

            image = load_image_tensor(entry["teacher_img"], device)
            box_tensor = make_box_tensor(entry["bbox"], device)
            target: torch.Tensor | None = None
            if entry["label_mode"] == "full":
                target = load_full_target(entry["teacher_gt"], device, label_id)

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
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            probability_full = torch.sigmoid(logits_full)
            probability_low = torch.sigmoid(low_res_logits)

            with torch.no_grad():
                ema_low_logits = decode_low_res(
                    ema_model,
                    image_embedding,
                    sparse_embeddings,
                    dense_embeddings,
                )
                probability_ema_low = torch.sigmoid(ema_low_logits)

            low_hw = probability_low.shape[-2:]
            full_hw = probability_full.shape[-2:]
            box_mask_full = make_box_mask(
                entry["bbox"], full_hw[0], full_hw[1], device
            )
            box_mask_low = F.interpolate(
                box_mask_full, size=low_hw, mode="nearest"
            )

            qf_low = compute_qf_from_embedding(
                image_embedding,
                template[foreground_key],
                template[background_key],
                low_hw,
            )
            shape_low = build_shape_map(
                template[shape_key],
                entry["bbox"],
                low_hw[0],
                low_hw[1],
                device,
            )

            zero = torch.tensor(0.0, device=device)
            loss_full = zero
            loss_out = zero
            loss_seed = zero
            loss_wac = zero
            loss_proto = zero
            loss_smooth = zero

            weak1_sum = 0.0
            weak2_sum = 0.0
            strong_sum = 0.0
            gate1_sum = 0.0
            gate2_sum = 0.0
            box_area_low = float(box_mask_low.sum().detach().cpu())
            seed_sum = 0.0
            unc_sum = 0.0

            if entry["label_mode"] == "full":
                if target is None:
                    raise RuntimeError("Full entry reached full branch without target")
                loss_full = dice_loss(probability_full, target) + weighted_bce(
                    probability_full, target
                )
                loss = loss_full
            else:
                outside = 1.0 - box_mask_full
                loss_out = weighted_bce(
                    probability_full,
                    torch.zeros_like(probability_full),
                    outside,
                )

                seed = (
                    (probability_ema_low > args.seed_p_threshold)
                    & (qf_low > args.seed_qf_threshold)
                    & (box_mask_low > 0.5)
                ).float()
                seed_sum = float(seed.sum().detach().cpu())
                if seed_sum > 0:
                    loss_seed = weighted_bce(
                        probability_low,
                        torch.ones_like(probability_low),
                        seed,
                    )

                uncertain = (
                    (probability_low > args.proto_p_low)
                    & (probability_low < args.proto_p_high)
                    & (box_mask_low > 0.5)
                ).float()
                unc_sum = float(uncertain.sum().detach().cpu())
                if unc_sum > 0:
                    loss_proto = weighted_bce(
                        probability_low,
                        qf_low,
                        uncertain,
                    )

                weak1 = (
                    (probability_low > args.weak1_low)
                    & (probability_low < args.weak1_high)
                    & (box_mask_low > 0.5)
                ).float()
                weak2 = (
                    (probability_low >= args.weak2_low)
                    & (probability_low < args.weak2_high)
                    & (box_mask_low > 0.5)
                ).float()
                strong = (
                    (probability_ema_low >= args.strong_p_threshold)
                    & (qf_low > args.strong_qf_threshold)
                    & (box_mask_low > 0.5)
                ).float()

                local_target = F.max_pool2d(
                    probability_ema_low * strong,
                    kernel_size=args.wac_kernel,
                    stride=1,
                    padding=args.wac_kernel // 2,
                )
                local_presence = F.max_pool2d(
                    strong,
                    kernel_size=args.wac_kernel,
                    stride=1,
                    padding=args.wac_kernel // 2,
                )

                shape_reliability = (
                    1.0 - 4.0 * shape_low * (1.0 - shape_low)
                ).clamp(0.0, 1.0)
                shape_gate = (0.50 + 0.50 * shape_reliability).clamp(0.50, 1.0)
                qf_gate = (0.25 + 0.75 * qf_low).clamp(0.25, 1.0)
                gate_base = (
                    (local_presence > 0.50).float() * shape_gate * qf_gate
                )
                gate1 = weak1 * gate_base
                gate2 = weak2 * gate_base

                weak1_sum = float(weak1.sum().detach().cpu())
                weak2_sum = float(weak2.sum().detach().cpu())
                strong_sum = float(strong.sum().detach().cpu())
                gate1_sum = float(gate1.sum().detach().cpu())
                gate2_sum = float(gate2.sum().detach().cpu())

                wac_map = F.smooth_l1_loss(
                    probability_low,
                    local_target.detach(),
                    reduction="none",
                )
                if gate1_sum > 0:
                    loss_wac = loss_wac + (
                        (wac_map * gate1).sum() / gate1.sum().clamp_min(1.0)
                    )
                if gate2_sum > 0:
                    loss_wac = loss_wac + args.weak2_weight * (
                        (wac_map * gate2).sum() / gate2.sum().clamp_min(1.0)
                    )

                loss_smooth = tv_smooth_loss(probability_low)
                loss = (
                    args.lambda_out * loss_out
                    + args.lambda_seed * loss_seed
                    + args.lambda_wac * loss_wac
                    + args.lambda_proto * loss_proto
                    + args.lambda_smooth * loss_smooth
                )

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
                "label_mode": entry["label_mode"],
                "label_id": label_id,
                "loss": float(loss.detach().cpu()),
                "loss_full": float(loss_full.detach().cpu()),
                "loss_out": float(loss_out.detach().cpu()),
                "loss_seed": float(loss_seed.detach().cpu()),
                "loss_wac": float(loss_wac.detach().cpu()),
                "loss_proto": float(loss_proto.detach().cpu()),
                "loss_smooth": float(loss_smooth.detach().cpu()),
                "mean_p": float(probability_low.detach().mean().cpu()),
                "mean_qf": float(qf_low.detach().mean().cpu()),
                "weak1_sum": weak1_sum,
                "weak2_sum": weak2_sum,
                "strong_sum": strong_sum,
                "gate1_sum": gate1_sum,
                "gate2_sum": gate2_sum,
                "box_area_low": box_area_low,
                "seed_sum": seed_sum,
                "unc_sum": unc_sum,
            }
            with log_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=log_fields).writerow(row)

            if global_step % args.log_every == 0:
                print(
                    f"[{dataset}] epoch={epoch} step={global_step} "
                    f"mode={entry['label_mode']} loss={row['loss']:.4f} "
                    f"full={row['loss_full']:.4f} out={row['loss_out']:.4f} "
                    f"seed={row['loss_seed']:.4f} wac={row['loss_wac']:.4f} "
                    f"proto={row['loss_proto']:.4f} smooth={row['loss_smooth']:.4f} "
                    f"p={row['mean_p']:.4f} qf={row['mean_qf']:.4f}"
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

    summary = {
        "dataset": dataset,
        "fold": args.fold,
        "method": args.method,
        "epochs": args.epochs,
        "global_step": global_step,
        "checkpoint_last": str(last_checkpoint),
        "checkpoint_ema": str(ema_checkpoint),
        "log_path": str(log_path),
        "trainable": "mask_decoder_only",
        "full_loss": "dice_plus_bce",
        "lambda_out": args.lambda_out,
        "lambda_seed": args.lambda_seed,
        "lambda_wac": args.lambda_wac,
        "lambda_proto": args.lambda_proto,
        "lambda_smooth": args.lambda_smooth,
        "ema_decay": args.ema_decay,
        "boxes_per_full_in_schedule": 3,
        "schedule_repeats_samples": False,
    }
    save_json(summary, out_dir / f"medsam_ft_summary_{args.method}.json")

    print(f"[OK] training completed: {dataset}")
    print(f"     last = {last_checkpoint}")
    print(f"     ema  = {ema_checkpoint}")
    print(f"     log  = {log_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fine-tune only the MedSAM mask decoder with full and box supervision."
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

    parser.add_argument("--lambda_out", type=float, default=1.0)
    parser.add_argument("--lambda_seed", type=float, default=0.5)
    parser.add_argument("--lambda_wac", type=float, default=0.5)
    parser.add_argument("--lambda_proto", type=float, default=0.3)
    parser.add_argument("--lambda_smooth", type=float, default=0.03)

    parser.add_argument("--seed_p_threshold", type=float, default=0.65)
    parser.add_argument("--seed_qf_threshold", type=float, default=0.55)
    parser.add_argument("--proto_p_low", type=float, default=0.15)
    parser.add_argument("--proto_p_high", type=float, default=0.85)
    parser.add_argument("--weak1_low", type=float, default=0.30)
    parser.add_argument("--weak1_high", type=float, default=0.40)
    parser.add_argument("--weak2_low", type=float, default=0.40)
    parser.add_argument("--weak2_high", type=float, default=0.50)
    parser.add_argument("--weak2_weight", type=float, default=0.5)
    parser.add_argument("--strong_p_threshold", type=float, default=0.50)
    parser.add_argument("--strong_qf_threshold", type=float, default=0.55)
    parser.add_argument("--wac_kernel", type=int, default=31)

    parser.add_argument("--log_every", type=int, default=10)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.wac_kernel <= 0 or args.wac_kernel % 2 == 0:
        raise ValueError("--wac_kernel must be a positive odd integer")
    for name in (
        "lambda_out",
        "lambda_seed",
        "lambda_wac",
        "lambda_proto",
        "lambda_smooth",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be non-negative")


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
