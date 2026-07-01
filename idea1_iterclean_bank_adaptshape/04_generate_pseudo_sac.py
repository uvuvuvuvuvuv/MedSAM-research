from __future__ import annotations

import argparse
import csv
import json
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


def get_npy_output_name(slice_name: str) -> str:
    """Return the actual filename produced by np.save."""
    return (
        slice_name
        if slice_name.lower().endswith(".npy")
        else f"{slice_name}.npy"
    )

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



def build_manifest_index(
    manifest: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    manifest_by_slice: dict[str, dict[str, Any]] = {}

    for item in manifest:
        if not isinstance(item, dict):
            raise TypeError(
                "Each manifest item must be a dict, "
                f"got {type(item).__name__}"
            )

        slice_name = get_slice_name(item)
        if slice_name in manifest_by_slice:
            raise RuntimeError(
                f"Duplicate slice_name found in manifest: {slice_name}"
            )
        manifest_by_slice[slice_name] = item

    return manifest_by_slice


def prepare_target_records(
    split_records: list[dict[str, Any]],
    max_samples: int,
    split_path: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    target_records = (
        split_records[:max_samples]
        if max_samples > 0
        else split_records
    )

    if not target_records:
        raise RuntimeError(
            f"No target records found in split file: {split_path}"
        )

    expected_slice_names: list[str] = []

    for record in target_records:
        if not isinstance(record, dict):
            raise TypeError(
                "Each Full/Box split record must be a dict, "
                f"got {type(record).__name__}"
            )

        slice_name = get_slice_name(record)
        label_mode = str(record.get("label_mode", ""))

        if label_mode not in {"full", "box"}:
            raise ValueError(
                f"Invalid label_mode={label_mode!r} for sample {slice_name}"
            )

        expected_slice_names.append(slice_name)

    if len(expected_slice_names) != len(set(expected_slice_names)):
        seen: set[str] = set()
        duplicates: list[str] = []

        for name in expected_slice_names:
            if name in seen:
                duplicates.append(name)
            seen.add(name)

        raise RuntimeError(
            "Duplicate slice names found in Full/Box split. "
            f"Examples: {sorted(set(duplicates))[:10]}"
        )

    return target_records, expected_slice_names


def validate_inputs(
    *,
    dataset: str,
    fold: str,
    method: str,
    target_records: list[dict[str, Any]],
    expected_slice_names: list[str],
    manifest_by_slice: dict[str, dict[str, Any]],
    prompts: dict[str, Any],
    meta_dir: Path,
) -> dict[str, Any]:
    missing_in_manifest = [
        name for name in expected_slice_names
        if name not in manifest_by_slice
    ]
    missing_in_prompts = [
        name for name in expected_slice_names
        if name not in prompts
    ]
    non_train_samples = [
        name for name in expected_slice_names
        if (
            name in manifest_by_slice
            and manifest_by_slice[name].get("split") != "train"
        )
    ]
    invalid_prompt_records = [
        name for name in expected_slice_names
        if name in prompts and not isinstance(prompts[name], dict)
    ]

    report = {
        "dataset": dataset,
        "fold": fold,
        "method": method,
        "num_target_records": len(target_records),
        "num_missing_in_manifest": len(missing_in_manifest),
        "missing_in_manifest_examples": missing_in_manifest[:10],
        "num_missing_in_prompts": len(missing_in_prompts),
        "missing_in_prompts_examples": missing_in_prompts[:10],
        "num_non_train_samples": len(non_train_samples),
        "non_train_examples": non_train_samples[:10],
        "num_invalid_prompt_records": len(invalid_prompt_records),
        "invalid_prompt_examples": invalid_prompt_records[:10],
        "input_complete": not (
            missing_in_manifest
            or missing_in_prompts
            or non_train_samples
            or invalid_prompt_records
        ),
    }

    audit_path = meta_dir / f"pseudo_generation_input_audit_{method}.json"
    save_json(report, audit_path)

    if not report["input_complete"]:
        raise RuntimeError(
            "Pseudo-generation input completeness check failed:\n"
            + json.dumps(report, indent=2, ensure_ascii=False)
        )

    return report


def validate_outputs(
    *,
    dataset: str,
    fold: str,
    method: str,
    expected_slice_names: list[str],
    processed_slice_names: list[str],
    teacher_dir: Path,
    student_dir: Path,
    meta_dir: Path,
) -> dict[str, Any]:
    expected_count = len(expected_slice_names)

    if len(processed_slice_names) != expected_count:
        raise RuntimeError(
            "Generated sample count mismatch: "
            f"expected={expected_count}, "
            f"generated={len(processed_slice_names)}"
        )

    if len(processed_slice_names) != len(set(processed_slice_names)):
        raise RuntimeError(
            "Duplicate samples were processed during pseudo generation"
        )

    expected_output_names = {
        get_npy_output_name(name)
        for name in expected_slice_names
    }
    actual_teacher_names = {
        path.name for path in teacher_dir.glob("*.npy")
    }
    actual_student_names = {
        path.name for path in student_dir.glob("*.npy")
    }

    missing_teacher = expected_output_names - actual_teacher_names
    extra_teacher = actual_teacher_names - expected_output_names
    missing_student = expected_output_names - actual_student_names
    extra_student = actual_student_names - expected_output_names

    report = {
        "dataset": dataset,
        "fold": fold,
        "method": method,
        "expected_count": expected_count,
        "processed_count": len(processed_slice_names),
        "teacher_file_count": len(actual_teacher_names),
        "student_file_count": len(actual_student_names),
        "missing_teacher_count": len(missing_teacher),
        "missing_teacher_examples": sorted(missing_teacher)[:10],
        "extra_teacher_count": len(extra_teacher),
        "extra_teacher_examples": sorted(extra_teacher)[:10],
        "missing_student_count": len(missing_student),
        "missing_student_examples": sorted(missing_student)[:10],
        "extra_student_count": len(extra_student),
        "extra_student_examples": sorted(extra_student)[:10],
        "output_complete": not (
            missing_teacher
            or extra_teacher
            or missing_student
            or extra_student
        ),
    }

    audit_path = meta_dir / f"pseudo_generation_output_audit_{method}.json"
    save_json(report, audit_path)

    if not report["output_complete"]:
        raise RuntimeError(
            "Pseudo-generation output completeness check failed:\n"
            + json.dumps(report, indent=2, ensure_ascii=False)
        )

    return report


def process_dataset(args: argparse.Namespace, dataset: str) -> None:
    device = torch.device(args.device)
    fold_root = args.processed_root / dataset / args.fold
    meta_dir = fold_root / "meta"

    manifest_path = meta_dir / "manifest.json"
    prompt_path = fold_root / "prompts" / "prompts_train.json"
    split_path = meta_dir / f"full_box_split_{args.method}.json"
    geometry_path = meta_dir / "geometry_meta.json"
    template_path = meta_dir / f"support_template_{args.method}.npz"

    for required_path in (
        manifest_path,
        prompt_path,
        split_path,
        geometry_path,
        template_path,
        args.base_checkpoint,
        args.sac_checkpoint,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required input does not exist: {required_path}"
            )

    manifest = load_json(manifest_path)
    prompts = load_json(prompt_path)
    split_records = load_json(split_path)
    geometry_meta = load_json(geometry_path)

    if not isinstance(manifest, list):
        raise TypeError(
            f"manifest.json must contain a list: {manifest_path}"
        )
    if not isinstance(prompts, dict):
        raise TypeError(
            f"prompts_train.json must contain a dict: {prompt_path}"
        )
    if not isinstance(split_records, list):
        raise TypeError(
            f"Full/Box split must contain a list: {split_path}"
        )
    if not isinstance(geometry_meta, dict):
        raise TypeError(
            f"geometry_meta.json must contain a dict: {geometry_path}"
        )

    manifest_by_slice = build_manifest_index(manifest)
    target_records, expected_slice_names = prepare_target_records(
        split_records=split_records,
        max_samples=args.max_samples,
        split_path=split_path,
    )

    input_audit = validate_inputs(
        dataset=dataset,
        fold=args.fold,
        method=args.method,
        target_records=target_records,
        expected_slice_names=expected_slice_names,
        manifest_by_slice=manifest_by_slice,
        prompts=prompts,
        meta_dir=meta_dir,
    )

    print(
        f"[CHECK] input completeness passed: "
        f"dataset={dataset} expected={len(target_records)}"
    )

    template = np.load(template_path, allow_pickle=False)
    if "class_ids" not in template:
        raise KeyError(
            f"Support template is missing 'class_ids': {template_path}"
        )

    class_ids = [
        int(value)
        for value in template["class_ids"].tolist()
    ]
    model = build_medsam(
        args.base_checkpoint,
        args.sac_checkpoint,
        device,
    )

    teacher_dir = (
        fold_root
        / "pseudo_teacher"
        / f"tri_train_{args.method}"
    )
    student_dir = (
        fold_root
        / "pseudo_student"
        / f"tri_train_{args.method}"
    )
    teacher_dir.mkdir(parents=True, exist_ok=True)
    student_dir.mkdir(parents=True, exist_ok=True)

    if args.overwrite:
        for directory in (teacher_dir, student_dir):
            for path in directory.glob("*.npy"):
                path.unlink()

    stats_path = meta_dir / f"pseudo_quality_stats_{args.method}.csv"
    fieldnames = [
        "slice_name",
        "case_id",
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

    processed_slice_names: list[str] = []

    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for split_record in target_records:
            slice_name = get_slice_name(split_record)

            # These keys were checked before loading the model.
            item = manifest_by_slice[slice_name]
            prompt_meta = prompts[slice_name]

            if item.get("split") != "train":
                raise RuntimeError(
                    f"Non-train item in split file: {slice_name}"
                )
            if not isinstance(prompt_meta, dict):
                raise TypeError(
                    f"Prompt metadata must be a dict: {slice_name}"
                )

            image_tensor, teacher_hw = load_image_tensor(
                resolve_path(fold_root, item, "teacher_img"),
                device,
            )
            label_mode = str(split_record["label_mode"])

            teacher_gt: np.ndarray | None = None
            if label_mode == "full":
                teacher_gt = load_mask(
                    resolve_path(fold_root, item, "teacher_gt")
                )

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
            geometry = geometry_meta.get(
                slice_name,
                geometry_meta.get(geometry_key),
            )
            if geometry is None:
                raise KeyError(
                    f"Missing geometry metadata for {slice_name}"
                )

            tri_native = teacher_mask_to_native(
                tri_teacher,
                geometry,
            )
            tri_student = native_mask_to_student(
                tri_native,
                geometry,
                teacher_hw,
            )

            teacher_output_path = (
                teacher_dir / get_npy_output_name(slice_name)
            )
            student_output_path = (
                student_dir / get_npy_output_name(slice_name)
            )

            np.save(
                teacher_output_path,
                tri_teacher.astype(np.uint8),
            )
            np.save(
                student_output_path,
                tri_student.astype(np.uint8),
            )
            processed_slice_names.append(slice_name)

            foreground = (tri_teacher > 0) & (tri_teacher != 255)
            background = tri_teacher == 0
            unknown = tri_teacher == 255
            components, largest_ratio, present_classes = (
                connected_component_stats(tri_teacher, class_ids)
            )

            writer.writerow(
                {
                    "slice_name": slice_name,
                    "case_id": str(item.get("case_id", "unknown_case")),
                    "label_mode": label_mode,
                    "fg_ratio": float(foreground.mean()),
                    "bg_ratio": float(background.mean()),
                    "unknown_ratio": float(unknown.mean()),
                    "box_fg_ratio": float(
                        aux.get("box_fg_ratio", 0.0)
                    ),
                    "box_unknown_ratio": float(
                        aux.get("box_unknown_ratio", 0.0)
                    ),
                    "weak_activation_ratio_0.30_0.40": float(
                        aux.get(
                            "weak_activation_ratio_0.30_0.40",
                            0.0,
                        )
                    ),
                    "weak_activation_ratio_0.40_0.50": float(
                        aux.get(
                            "weak_activation_ratio_0.40_0.50",
                            0.0,
                        )
                    ),
                    "mean_P_inside_box": float(
                        aux.get("mean_P_inside_box", 0.0)
                    ),
                    "mean_qF_inside_box": float(
                        aux.get("mean_qF_inside_box", 0.0)
                    ),
                    "mean_A_inside_box": float(
                        aux.get("mean_A_inside_box", 0.0)
                    ),
                    "num_connected_components": int(components),
                    "largest_component_ratio": float(largest_ratio),
                    "num_present_classes": int(present_classes),
                }
            )

            count = len(processed_slice_names)
            if count % args.log_every == 0:
                print(
                    f"[{dataset}] generated "
                    f"{count}/{len(target_records)}"
                )

    output_audit = validate_outputs(
        dataset=dataset,
        fold=args.fold,
        method=args.method,
        expected_slice_names=expected_slice_names,
        processed_slice_names=processed_slice_names,
        teacher_dir=teacher_dir,
        student_dir=student_dir,
        meta_dir=meta_dir,
    )

    expected_count = len(expected_slice_names)
    generated_count = len(processed_slice_names)

    config = {
        "schema_version": 2,
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
        "max_samples": int(args.max_samples),
        "num_split_records": len(split_records),
        "num_expected_samples": expected_count,
        "num_generated_samples": generated_count,
        "num_teacher_files": output_audit["teacher_file_count"],
        "num_student_files": output_audit["student_file_count"],
        "generation_complete": True,
        "input_complete": bool(input_audit["input_complete"]),
        "output_complete": bool(output_audit["output_complete"]),
        "missing_in_manifest": 0,
        "missing_in_prompts": 0,
        "missing_teacher_outputs": 0,
        "missing_student_outputs": 0,
        "test_samples_processed": 0,
    }

    config_path = meta_dir / f"pseudo_generation_config_{args.method}.json"
    save_json(config, config_path)

    print(
        f"[CHECK] generation complete: "
        f"expected={expected_count} "
        f"teacher={output_audit['teacher_file_count']} "
        f"student={output_audit['student_file_count']}"
    )
    print(f"[OK] pseudo generated: {dataset}")
    print(f"     method       = {args.method}")
    print(f"     expected     = {expected_count}")
    print(f"     generated    = {generated_count}")
    print(f"     teacher      = {teacher_dir}")
    print(f"     student      = {student_dir}")
    print(f"     stats        = {stats_path}")
    print(f"     config       = {config_path}")
    print("     completeness = PASS")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate teacher- and student-space tri-state SAC "
            "pseudo-labels without loading Box-sample GT."
        )
    )
    parser.add_argument("--processed_root", type=Path, required=True)
    parser.add_argument("--base_checkpoint", type=Path, required=True)
    parser.add_argument("--sac_checkpoint", type=Path, required=True)
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated dataset names",
    )
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument(
        "--method",
        required=True,
        help=(
            "Round-specific method name, for example "
            "idea1_iter_gt_iou_s2026_r0_full5"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help=(
            "0 means all split records. Positive values are only "
            "recommended for smoke tests."
        ),
    )

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
    if args.max_samples < 0:
        raise ValueError("--max_samples must be >= 0")
    if args.log_every <= 0:
        raise ValueError("--log_every must be positive")
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
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; falling back to CPU")
        args.device = "cpu"

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not datasets:
        raise ValueError("--datasets is empty")

    for dataset in datasets:
        process_dataset(args, dataset)


if __name__ == "__main__":
    main()
