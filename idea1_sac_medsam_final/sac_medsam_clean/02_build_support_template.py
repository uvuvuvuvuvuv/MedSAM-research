from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


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


def resolve_path(
    fold_root: Path,
    item: dict[str, Any],
    keys: tuple[str, ...],
    fallback_dirs: tuple[str, ...],
) -> Path:
    for key in keys:
        value = item.get(key)
        if not value:
            continue
        path = Path(str(value))
        if path.is_absolute() and path.exists():
            return path
        candidate = fold_root / path
        if candidate.exists():
            return candidate

    slice_name = get_slice_name(item)
    for directory in fallback_dirs:
        base = fold_root / directory
        for candidate in (
            base / slice_name,
            base / "train" / slice_name,
            base / "test" / slice_name,
        ):
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Cannot resolve path for slice={slice_name}, keys={keys}, "
        f"fallback_dirs={fallback_dirs}"
    )


def load_image_npy(path: Path) -> np.ndarray:
    image = np.load(path)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Unexpected image shape {image.shape} from {path}")

    image = image.astype(np.float32)
    if image.max() > 2.0:
        image /= 255.0
    return np.clip(image, 0.0, 1.0)


def load_mask_npy(path: Path) -> np.ndarray:
    mask = np.load(path)
    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[0] == 1:
            mask = mask[0]
        else:
            raise ValueError(f"Unexpected mask shape {mask.shape} from {path}")
    return mask.astype(np.int64)


def build_medsam(checkpoint: Path, device: torch.device):
    try:
        from segment_anything import sam_model_registry
    except Exception as exc:
        raise ImportError(
            "Cannot import segment_anything. Run this script in the MedSAM environment."
        ) from exc

    model = sam_model_registry[MODEL_TYPE](checkpoint=str(checkpoint))
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_encoder_feature(
    model,
    image_hwc: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    tensor = (
        torch.from_numpy(image_hwc)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
    )
    feature = model.image_encoder(tensor)
    if isinstance(feature, (list, tuple)):
        feature = feature[0]
    feature_np = feature.detach().float().cpu().numpy()[0]
    feature_np = np.transpose(feature_np, (1, 2, 0))
    feature_np /= np.linalg.norm(feature_np, axis=-1, keepdims=True) + 1e-6
    return feature_np.astype(np.float32)


def resize_mask_nearest(mask: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    return cv2.resize(
        mask.astype(np.int32),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int64)


def resize_float(mask: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = size_hw
    return cv2.resize(
        mask.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)


def bbox_from_binary(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def sample_rows(
    array: np.ndarray,
    max_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if array.shape[0] <= max_count:
        return array
    indices = rng.choice(array.shape[0], size=max_count, replace=False)
    return array[indices]


def spherical_kmeans(
    features: np.ndarray,
    n_clusters: int,
    seed: int,
    num_iter: int = 25,
) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError("spherical_kmeans requires a non-empty [N, C] array")

    rng = np.random.default_rng(seed)
    n_samples = features.shape[0]
    n_clusters = min(n_clusters, n_samples)
    centers = features[rng.choice(n_samples, size=n_clusters, replace=False)].copy()

    for _ in range(num_iter):
        labels = (features @ centers.T).argmax(axis=1)
        updated: list[np.ndarray] = []
        for cluster_id in range(n_clusters):
            part = features[labels == cluster_id]
            if part.shape[0] == 0:
                center = features[rng.integers(0, n_samples)]
            else:
                center = part.mean(axis=0)
            center = center / (np.linalg.norm(center) + 1e-6)
            updated.append(center.astype(np.float32))
        centers = np.stack(updated, axis=0)

    return centers.astype(np.float32)


def read_class_ids(meta_dir: Path, method: str, full_records: list[dict[str, Any]]) -> list[int]:
    label_meta_path = meta_dir / "label_meta.json"
    summary_path = meta_dir / f"full_box_split_summary_{method}.json"

    class_ids: list[int] = []
    if label_meta_path.exists():
        label_meta = load_json(label_meta_path)
        for key in ("unique_labels", "class_ids"):
            values = label_meta.get(key, []) if isinstance(label_meta, dict) else []
            if isinstance(values, list):
                class_ids.extend(int(value) for value in values if int(value) > 0)

    if not class_ids and summary_path.exists():
        summary = load_json(summary_path)
        class_ids.extend(
            int(value) for value in summary.get("class_ids", []) if int(value) > 0
        )

    if not class_ids:
        class_ids.extend(
            int(value)
            for record in full_records
            for value in record.get("class_ids", [1])
            if int(value) > 0
        )

    result = sorted(set(class_ids))
    return result or [1]


def process_dataset(args: argparse.Namespace, dataset: str) -> None:
    rng = np.random.default_rng(args.seed)
    fold_root = args.processed_root / dataset / args.fold
    meta_dir = fold_root / "meta"

    manifest_path = meta_dir / "manifest.json"
    split_path = meta_dir / f"full_box_split_{args.method}.json"
    out_npz = meta_dir / f"support_template_{args.method}.npz"
    out_stats = meta_dir / f"support_template_stats_{args.method}.json"

    if out_npz.exists() and out_stats.exists() and not args.overwrite:
        print(f"[SKIP] exists: {out_npz}")
        return

    manifest = load_json(manifest_path)
    split_records = load_json(split_path)
    manifest_by_slice = {get_slice_name(item): item for item in manifest}
    full_records = [
        record for record in split_records if record.get("label_mode") == "full"
    ]
    if not full_records:
        raise RuntimeError(f"No full records found in {split_path}")

    class_ids = read_class_ids(meta_dir, args.method, full_records)
    print(f"[CLASS_IDS] {dataset}/{args.fold}: {class_ids}")

    device = torch.device(args.device)
    model = build_medsam(args.checkpoint, device)

    foreground_features: dict[int, list[np.ndarray]] = {c: [] for c in class_ids}
    background_features: dict[int, list[np.ndarray]] = {c: [] for c in class_ids}
    shape_masks: dict[int, list[np.ndarray]] = {c: [] for c in class_ids}

    per_image_stats: list[dict[str, Any]] = []
    missing_manifest = 0
    feature_dim: int | None = None

    for index, record in enumerate(full_records):
        slice_name = str(record["slice_name"])
        item = manifest_by_slice.get(slice_name)
        if item is None:
            missing_manifest += 1
            print(f"[WARN] missing in manifest: {slice_name}")
            continue

        image_path = resolve_path(
            fold_root,
            item,
            keys=("teacher_img", "teacher_image", "img_teacher"),
            fallback_dirs=("teacher_npy/imgs",),
        )
        gt_path = resolve_path(
            fold_root,
            item,
            keys=("teacher_gt", "teacher_mask", "gt_teacher"),
            fallback_dirs=("teacher_npy/gts",),
        )

        image = load_image_npy(image_path)
        gt = load_mask_npy(gt_path)
        if image.shape[:2] != gt.shape[:2]:
            raise ValueError(
                f"Image/GT shape mismatch for {slice_name}: "
                f"image={image.shape}, gt={gt.shape}"
            )

        feature = extract_encoder_feature(model, image, device)
        feature_dim = int(feature.shape[-1])
        feature_h, feature_w = feature.shape[:2]
        gt_feature = resize_mask_nearest(gt, (feature_h, feature_w))

        sample_stat: dict[str, Any] = {
            "slice_name": slice_name,
            "image_path": str(image_path),
            "gt_path": str(gt_path),
            "classes": {},
        }

        for class_id in class_ids:
            binary_mask = (gt == class_id).astype(np.uint8)
            bbox = bbox_from_binary(binary_mask)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                crop = binary_mask[y1:y2, x1:x2]
                if crop.size:
                    crop64 = np.clip(resize_float(crop, (64, 64)), 0.0, 1.0)
                    shape_masks[class_id].append(crop64.astype(np.float32))

            fg_locations = np.where(gt_feature == class_id)
            bg_locations = np.where(gt_feature == 0)
            fg_array = feature[fg_locations]
            bg_array = feature[bg_locations]

            if fg_array.shape[0] > 0:
                foreground_features[class_id].append(
                    sample_rows(fg_array, args.max_fg_per_image, rng)
                )
            if bg_array.shape[0] > 0:
                background_features[class_id].append(
                    sample_rows(bg_array, args.max_bg_per_image, rng)
                )

            sample_stat["classes"][str(class_id)] = {
                "gt_pixels_teacher": int(binary_mask.sum()),
                "fg_feature_pixels": int(fg_array.shape[0]),
                "bg_feature_pixels": int(bg_array.shape[0]),
                "has_shape": bbox is not None,
            }

        per_image_stats.append(sample_stat)
        if (index + 1) % 20 == 0:
            print(
                f"[{dataset}] processed full templates "
                f"{index + 1}/{len(full_records)}"
            )

    if feature_dim is None:
        raise RuntimeError(f"No valid full sample was processed for {dataset}")

    npz_data: dict[str, np.ndarray] = {
        "class_ids": np.asarray(class_ids, dtype=np.int64)
    }
    stats: dict[str, Any] = {
        "template_version": 2,
        "dataset": dataset,
        "fold": args.fold,
        "method": args.method,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "model_type": MODEL_TYPE,
        "feature_dim": feature_dim,
        "num_full_records": len(full_records),
        "num_missing_manifest": missing_manifest,
        "class_ids": class_ids,
        "k_fg": args.k_fg,
        "k_bg": args.k_bg,
        "max_fg_per_image": args.max_fg_per_image,
        "max_bg_per_image": args.max_bg_per_image,
        "max_total_features": args.max_total_features,
        "classes": {},
        "per_image_stats": per_image_stats[:50],
    }

    for class_id in class_ids:
        fg_parts = foreground_features[class_id]
        bg_parts = background_features[class_id]
        if not fg_parts:
            raise RuntimeError(f"No foreground features collected for class {class_id}")
        if not bg_parts:
            raise RuntimeError(f"No background features collected for class {class_id}")

        fg = np.concatenate(fg_parts, axis=0).astype(np.float32)
        bg = np.concatenate(bg_parts, axis=0).astype(np.float32)
        fg = sample_rows(fg, args.max_total_features, rng)
        bg = sample_rows(bg, args.max_total_features, rng)

        proto_fg = spherical_kmeans(
            fg, args.k_fg, seed=args.seed + class_id * 11
        )
        proto_bg = spherical_kmeans(
            bg, args.k_bg, seed=args.seed + class_id * 17
        )
        shape_a = (
            np.stack(shape_masks[class_id], axis=0).mean(axis=0).astype(np.float32)
            if shape_masks[class_id]
            else np.zeros((64, 64), dtype=np.float32)
        )

        npz_data[f"proto_fg_c{class_id}"] = proto_fg
        npz_data[f"proto_bg_c{class_id}"] = proto_bg
        npz_data[f"shape_A_c{class_id}"] = shape_a

        stats["classes"][str(class_id)] = {
            "num_fg_features": int(fg.shape[0]),
            "num_bg_features": int(bg.shape[0]),
            "num_shape_masks": int(len(shape_masks[class_id])),
            "proto_fg_shape": list(proto_fg.shape),
            "proto_bg_shape": list(proto_bg.shape),
            "shape_A_shape": list(shape_a.shape),
            "shape_A_mean": float(shape_a.mean()),
            "shape_A_min": float(shape_a.min()),
            "shape_A_max": float(shape_a.max()),
        }

    np.savez_compressed(out_npz, **npz_data)
    save_json(stats, out_stats)

    print(f"[OK] support template built: {dataset}/{args.fold}")
    print(f"     out_npz   = {out_npz}")
    print(f"     out_stats = {out_stats}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build foreground/background prototypes and a shape template from full samples."
    )
    parser.add_argument("--processed_root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--k_fg", type=int, default=3)
    parser.add_argument("--k_bg", type=int, default=5)
    parser.add_argument("--max_fg_per_image", type=int, default=512)
    parser.add_argument("--max_bg_per_image", type=int, default=512)
    parser.add_argument("--max_total_features", type=int, default=200000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
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
