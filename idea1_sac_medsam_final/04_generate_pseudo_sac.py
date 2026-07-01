from __future__ import annotations

import argparse
import csv
import json
import os
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

METHOD_DEFAULT = "idea1_sac_medsam_final"
MODEL_TYPE = "vit_b"


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


def load_image_tensor(
    path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, tuple[int, int]]:
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
    tensor = (
        torch.from_numpy(image)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )
    return tensor, (int(image.shape[0]), int(image.shape[1]))


def load_mask(path: Path) -> np.ndarray:
    mask = np.load(path)
    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[0] == 1:
            mask = mask[0]
        else:
            raise ValueError(f"Unexpected mask shape {mask.shape} from {path}")
    return mask.astype(np.uint8)


def build_medsam(
    base_checkpoint: Path,
    sac_checkpoint: Path,
    device: torch.device,
):
    from segment_anything import sam_model_registry

    model = sam_model_registry[MODEL_TYPE](checkpoint=str(base_checkpoint))
    payload = torch.load(sac_checkpoint, map_location="cpu")
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[load SAC] missing={len(missing)} unexpected={len(unexpected)}")
    model.to(device)
    model.eval()
    return model


def make_box_tensor(box: list[float] | tuple[float, ...], device: torch.device) -> torch.Tensor:
    array = np.asarray(box, dtype=np.float32).reshape(1, 4)
    return torch.from_numpy(array).float().to(device)


def make_box_mask_np(
    box: list[float] | tuple[float, ...],
    height: int,
    width: int,
) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0, min(width - 1, int(np.floor(x1))))
    y1 = max(0, min(height - 1, int(np.floor(y1))))
    x2 = max(0, min(width, int(np.ceil(x2))))
    y2 = max(0, min(height, int(np.ceil(y2))))

    mask = np.zeros((height, width), dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = True
    return mask


@torch.no_grad()
def forward_medsam(
    model,
    image_tensor: torch.Tensor,
    box_tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_embedding = model.image_encoder(image_tensor)
    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=box_tensor,
        masks=None,
    )
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    logits_full = F.interpolate(
        low_res_logits,
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    return logits_full, image_embedding


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

    flat = feature.permute(0, 2, 3, 1).reshape(-1, channels)
    foreground = F.normalize(
        torch.from_numpy(proto_fg).float().to(feature.device), dim=1
    )
    background = F.normalize(
        torch.from_numpy(proto_bg).float().to(feature.device), dim=1
    )
    max_fg = (flat @ foreground.t()).max(dim=1).values
    max_bg = (flat @ background.t()).max(dim=1).values
    logits = torch.stack([max_fg, max_bg], dim=1) / temperature
    qf = torch.softmax(logits, dim=1)[:, 0].reshape(1, 1, height, width)
    return F.interpolate(
        qf,
        size=output_hw,
        mode="bilinear",
        align_corners=False,
    ).clamp(0.0, 1.0)


def build_shape_map_np(
    shape_a: np.ndarray,
    box: list[float] | tuple[float, ...],
    output_height: int,
    output_width: int,
) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = max(0, min(output_width - 1, int(np.floor(x1))))
    y1 = max(0, min(output_height - 1, int(np.floor(y1))))
    x2 = max(0, min(output_width, int(np.ceil(x2))))
    y2 = max(0, min(output_height, int(np.ceil(y2))))

    output = np.zeros((output_height, output_width), dtype=np.float32)
    if x2 <= x1 or y2 <= y1:
        return output

    resized = cv2.resize(
        shape_a.astype(np.float32),
        (x2 - x1, y2 - y1),
        interpolation=cv2.INTER_LINEAR,
    )
    output[y1:y2, x1:x2] = np.clip(resized, 0.0, 1.0)
    return output


def geometry_transforms(
    geometry: dict[str, Any],
    teacher_hw: tuple[int, int],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Normalize old and new geometry_meta formats into two forward transforms."""
    if not geometry:
        raise KeyError("geometry_meta entry is empty")

    teacher_height, teacher_width = teacher_hw
    native_hw = geometry.get("native_hw", [None, None])

    if geometry.get("native_to_teacher"):
        teacher = dict(geometry["native_to_teacher"])
    elif geometry.get("teacher"):
        teacher = dict(geometry["teacher"])
    elif geometry.get("teacher_to_native"):
        inverse = dict(geometry["teacher_to_native"])
        teacher = {
            "orig_h": inverse.get("to_h", inverse.get("orig_h", native_hw[0])),
            "orig_w": inverse.get("to_w", inverse.get("orig_w", native_hw[1])),
            "target_h": inverse.get("target_h", teacher_height),
            "target_w": inverse.get("target_w", teacher_width),
            "new_h": inverse.get("crop_h", inverse.get("new_h", teacher_height)),
            "new_w": inverse.get("crop_w", inverse.get("new_w", teacher_width)),
            "offset_x": inverse.get("offset_x", 0),
            "offset_y": inverse.get("offset_y", 0),
        }
    else:
        teacher = {
            "orig_h": geometry.get("orig_h", native_hw[0]),
            "orig_w": geometry.get("orig_w", native_hw[1]),
            "target_h": (geometry.get("teacher_hw") or teacher_hw)[0],
            "target_w": (geometry.get("teacher_hw") or teacher_hw)[1],
            "new_h": geometry.get("new_h", teacher_height),
            "new_w": geometry.get("new_w", teacher_width),
            "offset_x": geometry.get("offset_x", 0),
            "offset_y": geometry.get("offset_y", 0),
        }

    if geometry.get("native_to_student"):
        student = dict(geometry["native_to_student"])
    elif geometry.get("student"):
        student = dict(geometry["student"])
    else:
        raise KeyError("geometry_meta entry is missing native_to_student")

    for transform_name, transform in (("teacher", teacher), ("student", student)):
        for key in ("orig_h", "orig_w", "new_h", "new_w", "target_h", "target_w"):
            if transform.get(key) is None:
                raise KeyError(f"{transform_name} geometry is missing '{key}'")
            transform[key] = int(round(float(transform[key])))
        transform["offset_x"] = int(round(float(transform.get("offset_x", 0))))
        transform["offset_y"] = int(round(float(transform.get("offset_y", 0))))

    return teacher, student


def teacher_mask_to_native(
    teacher_mask: np.ndarray,
    geometry: dict[str, Any],
) -> np.ndarray:
    teacher, _ = geometry_transforms(geometry, teacher_mask.shape[:2])
    x0 = teacher["offset_x"]
    y0 = teacher["offset_y"]
    new_width = teacher["new_w"]
    new_height = teacher["new_h"]

    x1 = x0 + new_width
    y1 = y0 + new_height
    if x0 < 0 or y0 < 0 or x1 > teacher_mask.shape[1] or y1 > teacher_mask.shape[0]:
        raise ValueError(
            "Teacher valid region is outside the teacher canvas: "
            f"canvas={teacher_mask.shape}, offset=({x0},{y0}), "
            f"new_hw=({new_height},{new_width})"
        )

    valid = teacher_mask[y0:y1, x0:x1]
    return cv2.resize(
        valid,
        (teacher["orig_w"], teacher["orig_h"]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.uint8)


def native_mask_to_student(
    native_mask: np.ndarray,
    geometry: dict[str, Any],
    teacher_hw: tuple[int, int],
) -> np.ndarray:
    _, student = geometry_transforms(geometry, teacher_hw)
    resized = cv2.resize(
        native_mask.astype(np.uint8),
        (student["new_w"], student["new_h"]),
        interpolation=cv2.INTER_NEAREST,
    )

    canvas = np.zeros(
        (student["target_h"], student["target_w"]), dtype=np.uint8
    )
    x0 = student["offset_x"]
    y0 = student["offset_y"]
    x1 = x0 + student["new_w"]
    y1 = y0 + student["new_h"]
    if x0 < 0 or y0 < 0 or x1 > canvas.shape[1] or y1 > canvas.shape[0]:
        raise ValueError(
            "Student valid region is outside the student canvas: "
            f"canvas={canvas.shape}, offset=({x0},{y0}), "
            f"new_hw=({student['new_h']},{student['new_w']})"
        )
    canvas[y0:y1, x0:x1] = resized
    return canvas


def merge_instance_into_tri(
    tri: np.ndarray,
    foreground_mask: np.ndarray,
    unknown_mask: np.ndarray,
    label_id: int,
) -> np.ndarray:
    existing_foreground = (tri > 0) & (tri != 255)
    conflict = foreground_mask & existing_foreground & (tri != label_id)
    tri[conflict] = 255
    tri[foreground_mask & (~conflict)] = np.uint8(label_id)
    tri[unknown_mask & (tri == 0)] = 255
    return tri


def connected_component_stats(
    tri: np.ndarray,
    class_ids: list[int],
) -> tuple[int, float, int]:
    total_components = 0
    largest_ratios: list[float] = []
    present_classes = 0

    for class_id in class_ids:
        foreground = (tri == class_id).astype(np.uint8)
        if foreground.max() == 0:
            continue
        present_classes += 1
        n_labels, labels = cv2.connectedComponents(foreground)
        areas = [(labels == index).sum() for index in range(1, n_labels)]
        total_components += len(areas)
        if areas:
            largest_ratios.append(float(max(areas)) / max(float(sum(areas)), 1.0))

    mean_largest_ratio = float(np.mean(largest_ratios)) if largest_ratios else 0.0
    return total_components, mean_largest_ratio, present_classes


def generate_one_slice(
    model,
    image_tensor: torch.Tensor,
    prompt_meta: dict[str, Any],
    label_mode: str,
    output_hw: tuple[int, int],
    template,
    args: argparse.Namespace,
    teacher_gt: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    height, width = output_hw
    if label_mode == "full":
        if teacher_gt is None:
            raise ValueError("Full sample requires teacher_gt")
        if teacher_gt.shape != (height, width):
            raise ValueError(
                f"Full GT shape {teacher_gt.shape} does not match image shape {(height, width)}"
            )
        return teacher_gt.astype(np.uint8).copy(), {
            "mean_P_inside_box": 0.0,
            "mean_qF_inside_box": 0.0,
            "mean_A_inside_box": 0.0,
            "weak_activation_ratio_0.30_0.40": 0.0,
            "weak_activation_ratio_0.40_0.50": 0.0,
            "box_fg_ratio": 0.0,
            "box_unknown_ratio": 0.0,
        }

    tri = np.zeros((height, width), dtype=np.uint8)
    instances = prompt_meta.get("instances", [])
    if not instances:
        return tri, {}

    all_boxes = np.zeros((height, width), dtype=bool)
    p_values: list[np.ndarray] = []
    q_values: list[np.ndarray] = []
    a_values: list[np.ndarray] = []
    weak_030_040: list[float] = []
    weak_040_050: list[float] = []

    for instance in instances:
        box = instance.get("bbox_teacher", instance.get("bbox"))
        if box is None:
            continue
        label_id = int(instance.get("label_id", 1))
        foreground_key = f"proto_fg_c{label_id}"
        background_key = f"proto_bg_c{label_id}"
        shape_key = f"shape_A_c{label_id}"
        for key in (foreground_key, background_key, shape_key):
            if key not in template:
                raise KeyError(f"Support template is missing '{key}'")

        logits_full, image_embedding = forward_medsam(
            model,
            image_tensor,
            make_box_tensor(box, image_tensor.device),
        )
        probability = torch.sigmoid(logits_full)
        qf = compute_qf_from_embedding(
            image_embedding,
            template[foreground_key],
            template[background_key],
            output_hw=(height, width),
        )

        p_np = probability[0, 0].cpu().numpy().astype(np.float32)
        q_np = qf[0, 0].cpu().numpy().astype(np.float32)
        a_np = build_shape_map_np(
            template[shape_key], box, height, width
        )
        box_mask = make_box_mask_np(box, height, width)
        all_boxes |= box_mask

        quality = args.p_weight * p_np + args.qf_weight * q_np
        foreground = (
            box_mask
            & (p_np >= args.p_threshold)
            & (quality >= args.q_threshold)
            & (q_np >= args.fg_q_threshold)
            & (a_np >= args.shape_threshold)
        )
        background = (
            box_mask
            & (p_np <= args.bg_p_threshold)
            & (q_np <= args.bg_qf_threshold)
        )
        unknown = box_mask & (~foreground) & (~background)
        tri = merge_instance_into_tri(tri, foreground, unknown, label_id)

        p_values.append(p_np[box_mask])
        q_values.append(q_np[box_mask])
        a_values.append(a_np[box_mask])
        box_area = max(int(box_mask.sum()), 1)
        weak_030_040.append(
            float(((p_np > 0.30) & (p_np < 0.40) & box_mask).sum()) / box_area
        )
        weak_040_050.append(
            float(((p_np >= 0.40) & (p_np < 0.50) & box_mask).sum()) / box_area
        )

    if p_values:
        p_cat = np.concatenate(p_values)
        q_cat = np.concatenate(q_values)
        a_cat = np.concatenate(a_values)
    else:
        p_cat = q_cat = a_cat = np.asarray([0.0], dtype=np.float32)

    stats = {
        "mean_P_inside_box": float(p_cat.mean()),
        "mean_qF_inside_box": float(q_cat.mean()),
        "mean_A_inside_box": float(a_cat.mean()),
        "weak_activation_ratio_0.30_0.40": float(np.mean(weak_030_040))
        if weak_030_040
        else 0.0,
        "weak_activation_ratio_0.40_0.50": float(np.mean(weak_040_050))
        if weak_040_050
        else 0.0,
        "box_fg_ratio": float(
            ((tri > 0) & (tri != 255) & all_boxes).sum()
            / max(int(all_boxes.sum()), 1)
        ),
        "box_unknown_ratio": float(
            ((tri == 255) & all_boxes).sum()
            / max(int(all_boxes.sum()), 1)
        ),
    }
    return tri, stats


def process_dataset(args: argparse.Namespace, dataset: str) -> None:
    device = torch.device(args.device)
    fold_root = args.processed_root / dataset / args.fold
    meta_dir = fold_root / "meta"

    manifest = load_json(meta_dir / "manifest.json")
    prompts = load_json(fold_root / "prompts" / "prompts_train.json")
    split_records = load_json(
        meta_dir / f"full_box_split_{args.method}.json"
    )
    geometry_meta = load_json(meta_dir / "geometry_meta.json")
    manifest_by_slice = {get_slice_name(item): item for item in manifest}

    template = np.load(meta_dir / f"support_template_{args.method}.npz")
    class_ids = [int(value) for value in template["class_ids"].tolist()]
    model = build_medsam(args.base_checkpoint, args.sac_checkpoint, device)

    teacher_dir = fold_root / "pseudo_teacher" / f"tri_train_{args.method}"
    student_dir = fold_root / "pseudo_student" / f"tri_train_{args.method}"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    student_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for directory in (teacher_dir, student_dir):
            for path in directory.glob("*.npy"):
                path.unlink()

    stats_path = meta_dir / f"pseudo_quality_stats_{args.method}.csv"
    fieldnames = [
        "slice_name",
        "label_mode",
        "fg_ratio",
        "bg_ratio",
        "unknown_ratio",
        "box_fg_ratio",
        "box_unknown_ratio",
        "weak_activation_ratio_0.30_0.40",
        "weak_activation_ratio_0.40_0.50",
        "mean_P_inside_box",
        "mean_qF_inside_box",
        "mean_A_inside_box",
        "num_connected_components",
        "largest_component_ratio",
        "num_present_classes",
    ]

    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        count = 0
        for split_record in split_records:
            if args.max_samples > 0 and count >= args.max_samples:
                break

            slice_name = str(split_record["slice_name"])
            item = manifest_by_slice.get(slice_name)
            prompt_meta = prompts.get(slice_name)
            if item is None or prompt_meta is None:
                continue
            if item.get("split") != "train":
                raise RuntimeError(f"Non-train item in split file: {slice_name}")

            image_tensor, teacher_hw = load_image_tensor(
                resolve_path(fold_root, item, "teacher_img"), device
            )
            label_mode = str(split_record["label_mode"])
            teacher_gt: np.ndarray | None = None
            if label_mode == "full":
                teacher_gt = load_mask(resolve_path(fold_root, item, "teacher_gt"))

            tri_teacher, aux = generate_one_slice(
                model=model,
                image_tensor=image_tensor,
                prompt_meta=prompt_meta,
                label_mode=label_mode,
                output_hw=teacher_hw,
                template=template,
                args=args,
                teacher_gt=teacher_gt,
            )

            geometry_key = item.get("geometry_key", slice_name)
            geometry = geometry_meta.get(slice_name, geometry_meta.get(geometry_key))
            if geometry is None:
                raise KeyError(f"Missing geometry metadata for {slice_name}")

            tri_native = teacher_mask_to_native(tri_teacher, geometry)
            tri_student = native_mask_to_student(tri_native, geometry, teacher_hw)

            np.save(teacher_dir / slice_name, tri_teacher.astype(np.uint8))
            np.save(student_dir / slice_name, tri_student.astype(np.uint8))

            foreground = (tri_teacher > 0) & (tri_teacher != 255)
            background = tri_teacher == 0
            unknown = tri_teacher == 255
            components, largest_ratio, present_classes = connected_component_stats(
                tri_teacher, class_ids
            )

            writer.writerow(
                {
                    "slice_name": slice_name,
                    "label_mode": label_mode,
                    "fg_ratio": float(foreground.mean()),
                    "bg_ratio": float(background.mean()),
                    "unknown_ratio": float(unknown.mean()),
                    "box_fg_ratio": float(aux.get("box_fg_ratio", 0.0)),
                    "box_unknown_ratio": float(aux.get("box_unknown_ratio", 0.0)),
                    "weak_activation_ratio_0.30_0.40": float(
                        aux.get("weak_activation_ratio_0.30_0.40", 0.0)
                    ),
                    "weak_activation_ratio_0.40_0.50": float(
                        aux.get("weak_activation_ratio_0.40_0.50", 0.0)
                    ),
                    "mean_P_inside_box": float(aux.get("mean_P_inside_box", 0.0)),
                    "mean_qF_inside_box": float(aux.get("mean_qF_inside_box", 0.0)),
                    "mean_A_inside_box": float(aux.get("mean_A_inside_box", 0.0)),
                    "num_connected_components": int(components),
                    "largest_component_ratio": float(largest_ratio),
                    "num_present_classes": int(present_classes),
                }
            )

            count += 1
            if count % args.log_every == 0:
                print(f"[{dataset}] generated {count}/{len(split_records)}")

    config = {
        "dataset": dataset,
        "fold": args.fold,
        "method": args.method,
        "base_checkpoint": str(args.base_checkpoint),
        "sac_checkpoint": str(args.sac_checkpoint),
        "class_ids": class_ids,
        "p_weight": args.p_weight,
        "qf_weight": args.qf_weight,
        "p_threshold": args.p_threshold,
        "q_threshold": args.q_threshold,
        "fg_q_threshold": args.fg_q_threshold,
        "shape_threshold": args.shape_threshold,
        "bg_p_threshold": args.bg_p_threshold,
        "bg_qf_threshold": args.bg_qf_threshold,
        "teacher_dir": str(teacher_dir),
        "student_dir": str(student_dir),
        "stats_path": str(stats_path),
        "box_samples_load_gt": False,
    }
    save_json(config, meta_dir / f"pseudo_generation_config_{args.method}.json")

    print(f"[OK] pseudo generated: {dataset}")
    print(f"     teacher = {teacher_dir}")
    print(f"     student = {student_dir}")
    print(f"     stats   = {stats_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate teacher- and student-space tri-state pseudo-labels without loading box GT."
    )
    parser.add_argument("--processed_root", type=Path, required=True)
    parser.add_argument("--base_checkpoint", type=Path, required=True)
    parser.add_argument("--sac_checkpoint", type=Path, required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=0)

    parser.add_argument("--p_weight", type=float, default=0.65)
    parser.add_argument("--qf_weight", type=float, default=0.35)
    parser.add_argument("--p_threshold", type=float, default=0.50)
    parser.add_argument("--q_threshold", type=float, default=0.58)
    parser.add_argument("--fg_q_threshold", type=float, default=0.45)
    parser.add_argument("--shape_threshold", type=float, default=0.10)
    parser.add_argument("--bg_p_threshold", type=float, default=0.15)
    parser.add_argument("--bg_qf_threshold", type=float, default=0.25)

    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.p_weight < 0 or args.qf_weight < 0:
        raise ValueError("p_weight and qf_weight must be non-negative")
    weight_sum = args.p_weight + args.qf_weight
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"p_weight + qf_weight must equal 1, got {weight_sum}"
        )
    for name in (
        "p_threshold",
        "q_threshold",
        "fg_q_threshold",
        "shape_threshold",
        "bg_p_threshold",
        "bg_qf_threshold",
    ):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1], got {value}")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; falling back to CPU")
        args.device = "cpu"

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not datasets:
        raise ValueError("--datasets is empty")

    for dataset in datasets:
        process_dataset(args, dataset)


if __name__ == "__main__":
    main()
