from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_TYPE = "vit_b"
SHAPE_TEMPLATE_SIZE = (64, 64)
KMEANS_ITERATIONS = 25
DOWNSTREAM_TEMPERATURE = 0.07


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

    raise KeyError(
        f"Cannot infer slice_name from keys={list(item.keys())}"
    )


def build_unique_manifest_index(
    manifest: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    manifest_by_slice: dict[str, dict[str, Any]] = {}

    for item in manifest:
        if not isinstance(item, dict):
            raise TypeError(
                "Each manifest entry must be a dict, "
                f"got {type(item).__name__}"
            )

        slice_name = get_slice_name(item)

        if slice_name in manifest_by_slice:
            raise RuntimeError(
                f"Duplicate slice_name in manifest: {slice_name}"
            )

        manifest_by_slice[slice_name] = item

    return manifest_by_slice


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
        f"Cannot resolve path for slice={slice_name}, "
        f"keys={keys}, fallback_dirs={fallback_dirs}"
    )


def load_image_npy(path: Path) -> np.ndarray:
    image = np.load(path)

    if (
        image.ndim == 3
        and image.shape[0] in (1, 3)
        and image.shape[-1] not in (1, 3)
    ):
        image = np.transpose(image, (1, 2, 0))

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)

    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(
            f"Unexpected image shape {image.shape} from {path}"
        )

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
            raise ValueError(
                f"Unexpected mask shape {mask.shape} from {path}"
            )

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask, got shape={mask.shape} from {path}"
        )

    return mask.astype(np.int64)


def build_medsam(
    checkpoint: Path,
    device: torch.device,
):
    try:
        from segment_anything import sam_model_registry
    except Exception as exc:
        raise ImportError(
            "Cannot import segment_anything. "
            "Run this script in the MedSAM environment."
        ) from exc

    model = sam_model_registry[MODEL_TYPE](
        checkpoint=str(checkpoint)
    )

    # This script only extracts support features. No model parameter is trained.
    for parameter in model.parameters():
        parameter.requires_grad = False

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

    feature_np = (
        feature.detach()
        .float()
        .cpu()
        .numpy()[0]
    )

    # [C, Hf, Wf] -> [Hf, Wf, C]
    feature_np = np.transpose(
        feature_np,
        (1, 2, 0),
    )

    # Per-token L2 normalization.
    feature_np /= (
        np.linalg.norm(
            feature_np,
            axis=-1,
            keepdims=True,
        )
        + 1e-6
    )

    return feature_np.astype(np.float32)


def resize_mask_nearest(
    mask: np.ndarray,
    size_hw: tuple[int, int],
) -> np.ndarray:
    height, width = size_hw

    return cv2.resize(
        mask.astype(np.int32),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int64)


def resize_float(
    mask: np.ndarray,
    size_hw: tuple[int, int],
) -> np.ndarray:
    height, width = size_hw

    return cv2.resize(
        mask.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)


def bbox_from_binary(
    mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)

    if xs.size == 0:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()) + 1,
        int(ys.max()) + 1,
    )


def sample_rows(
    array: np.ndarray,
    max_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if array.shape[0] <= max_count:
        return array

    indices = rng.choice(
        array.shape[0],
        size=max_count,
        replace=False,
    )

    return array[indices]


def spherical_kmeans(
    features: np.ndarray,
    n_clusters: int,
    seed: int,
    num_iter: int = KMEANS_ITERATIONS,
) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(
            "spherical_kmeans requires a non-empty [N, C] array"
        )

    rng = np.random.default_rng(seed)
    n_samples = features.shape[0]
    n_clusters = min(n_clusters, n_samples)

    centers = features[
        rng.choice(
            n_samples,
            size=n_clusters,
            replace=False,
        )
    ].copy()

    for _ in range(num_iter):
        labels = (
            features @ centers.T
        ).argmax(axis=1)

        updated: list[np.ndarray] = []

        for cluster_id in range(n_clusters):
            part = features[labels == cluster_id]

            if part.shape[0] == 0:
                center = features[
                    rng.integers(0, n_samples)
                ]
            else:
                center = part.mean(axis=0)

            center = center / (
                np.linalg.norm(center) + 1e-6
            )

            updated.append(
                center.astype(np.float32)
            )

        centers = np.stack(
            updated,
            axis=0,
        )

    return centers.astype(np.float32)


def append_positive_ints(
    target: list[int],
    values: Any,
) -> None:
    if not isinstance(values, list):
        return

    for value in values:
        try:
            class_id = int(value)
        except (TypeError, ValueError):
            continue

        if 0 < class_id < 255:
            target.append(class_id)


def read_class_ids(
    meta_dir: Path,
    method: str,
    full_records: list[dict[str, Any]],
) -> list[int]:
    label_meta_path = meta_dir / "label_meta.json"
    summary_path = (
        meta_dir
        / f"full_box_split_summary_{method}.json"
    )

    class_ids: list[int] = []

    if label_meta_path.exists():
        label_meta = load_json(label_meta_path)

        if isinstance(label_meta, dict):
            append_positive_ints(
                class_ids,
                label_meta.get("unique_labels", []),
            )
            append_positive_ints(
                class_ids,
                label_meta.get("class_ids", []),
            )

    if not class_ids and summary_path.exists():
        summary = load_json(summary_path)

        if isinstance(summary, dict):
            append_positive_ints(
                class_ids,
                summary.get("class_ids", []),
            )

    if not class_ids:
        for record in full_records:
            append_positive_ints(
                class_ids,
                record.get("class_ids", [1]),
            )

    result = sorted(set(class_ids))
    return result or [1]


def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.seed < 0:
        raise ValueError(
            f"--seed must be >= 0, got {args.seed}"
        )

    for name in (
        "k_fg",
        "k_bg",
        "max_fg_per_image",
        "max_bg_per_image",
        "max_total_features",
    ):
        value = int(getattr(args, name))

        if value <= 0:
            raise ValueError(
                f"--{name} must be positive, got {value}"
            )


def process_dataset(
    args: argparse.Namespace,
    dataset: str,
) -> None:
    rng = np.random.default_rng(args.seed)

    fold_root = (
        args.processed_root
        / dataset
        / args.fold
    )

    meta_dir = fold_root / "meta"

    manifest_path = meta_dir / "manifest.json"
    split_path = (
        meta_dir
        / f"full_box_split_{args.method}.json"
    )

    out_npz = (
        meta_dir
        / f"support_template_{args.method}.npz"
    )

    out_stats = (
        meta_dir
        / f"support_template_stats_{args.method}.json"
    )

    out_audit = (
        meta_dir
        / f"feature_extraction_audit_{args.method}.json"
    )

    for required_path in (
        manifest_path,
        split_path,
        args.checkpoint,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required input does not exist: {required_path}"
            )

    if (
        out_npz.exists()
        and out_stats.exists()
        and out_audit.exists()
        and not args.overwrite
    ):
        print(f"[SKIP] exists: {out_npz}")
        print(f"       audit : {out_audit}")
        return

    manifest = load_json(manifest_path)
    split_records = load_json(split_path)

    if not isinstance(manifest, list):
        raise TypeError(
            f"manifest.json must contain a list: {manifest_path}"
        )

    if not isinstance(split_records, list):
        raise TypeError(
            f"Full/Box split must contain a list: {split_path}"
        )

    manifest_by_slice = build_unique_manifest_index(
        manifest
    )

    full_records = [
        record
        for record in split_records
        if str(record.get("label_mode")) == "full"
    ]

    if not full_records:
        raise RuntimeError(
            f"No Full records found in {split_path}"
        )

    full_slice_names: list[str] = []

    for record in full_records:
        if not isinstance(record, dict):
            raise TypeError(
                "Each Full/Box record must be a dict"
            )

        full_slice_names.append(
            get_slice_name(record)
        )

    if len(full_slice_names) != len(
        set(full_slice_names)
    ):
        raise RuntimeError(
            "Duplicate Full slice names found in split file"
        )

    class_ids = read_class_ids(
        meta_dir,
        args.method,
        full_records,
    )

    print(
        f"[CLASS_IDS] {dataset}/{args.fold}: "
        f"{class_ids}"
    )

    device = torch.device(args.device)

    model = build_medsam(
        args.checkpoint,
        device,
    )

    foreground_features: dict[int, list[np.ndarray]] = {
        class_id: []
        for class_id in class_ids
    }

    background_features: dict[int, list[np.ndarray]] = {
        class_id: []
        for class_id in class_ids
    }

    shape_masks: dict[int, list[np.ndarray]] = {
        class_id: []
        for class_id in class_ids
    }

    per_image_stats: list[dict[str, Any]] = []
    missing_manifest_names: list[str] = []
    processed_full_slice_names: list[str] = []
    feature_dim: int | None = None
    feature_map_hw: tuple[int, int] | None = None

    for index, record in enumerate(
        full_records
    ):
        slice_name = get_slice_name(record)
        item = manifest_by_slice.get(slice_name)

        if item is None:
            missing_manifest_names.append(
                slice_name
            )
            print(
                f"[WARN] missing in manifest: "
                f"{slice_name}"
            )
            continue

        if item.get("split") != "train":
            raise RuntimeError(
                "A Full sample is not in train split: "
                f"{slice_name}"
            )

        image_path = resolve_path(
            fold_root,
            item,
            keys=(
                "teacher_img",
                "teacher_image",
                "img_teacher",
            ),
            fallback_dirs=(
                "teacher_npy/imgs",
            ),
        )

        gt_path = resolve_path(
            fold_root,
            item,
            keys=(
                "teacher_gt",
                "teacher_mask",
                "gt_teacher",
            ),
            fallback_dirs=(
                "teacher_npy/gts",
            ),
        )

        image = load_image_npy(image_path)
        gt = load_mask_npy(gt_path)

        if image.shape[:2] != gt.shape[:2]:
            raise ValueError(
                f"Image/GT shape mismatch for {slice_name}: "
                f"image={image.shape}, gt={gt.shape}"
            )

        feature = extract_encoder_feature(
            model,
            image,
            device,
        )

        current_feature_dim = int(
            feature.shape[-1]
        )

        current_feature_map_hw = (
            int(feature.shape[0]),
            int(feature.shape[1]),
        )

        if feature_dim is None:
            feature_dim = current_feature_dim
        elif feature_dim != current_feature_dim:
            raise RuntimeError(
                "Feature dimension changed across samples: "
                f"expected={feature_dim}, "
                f"found={current_feature_dim}, "
                f"slice={slice_name}"
            )

        if feature_map_hw is None:
            feature_map_hw = current_feature_map_hw
        elif feature_map_hw != current_feature_map_hw:
            raise RuntimeError(
                "Feature-map size changed across samples: "
                f"expected={feature_map_hw}, "
                f"found={current_feature_map_hw}, "
                f"slice={slice_name}"
            )

        feature_h, feature_w = (
            current_feature_map_hw
        )

        gt_feature = resize_mask_nearest(
            gt,
            (feature_h, feature_w),
        )

        sample_stat: dict[str, Any] = {
            "slice_name": slice_name,
            "image_path": str(image_path),
            "gt_path": str(gt_path),
            "teacher_image_hw": [
                int(image.shape[0]),
                int(image.shape[1]),
            ],
            "feature_map_hw": [
                feature_h,
                feature_w,
            ],
            "classes": {},
        }

        for class_id in class_ids:
            binary_mask = (
                gt == class_id
            ).astype(np.uint8)

            bbox = bbox_from_binary(
                binary_mask
            )

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                crop = binary_mask[
                    y1:y2,
                    x1:x2,
                ]

                if crop.size:
                    crop64 = np.clip(
                        resize_float(
                            crop,
                            SHAPE_TEMPLATE_SIZE,
                        ),
                        0.0,
                        1.0,
                    )

                    shape_masks[class_id].append(
                        crop64.astype(np.float32)
                    )

            fg_locations = np.where(
                gt_feature == class_id
            )

            bg_locations = np.where(
                gt_feature == 0
            )

            fg_array = feature[fg_locations]
            bg_array = feature[bg_locations]

            fg_sampled_count = 0
            bg_sampled_count = 0

            if fg_array.shape[0] > 0:
                sampled_fg = sample_rows(
                    fg_array,
                    args.max_fg_per_image,
                    rng,
                )

                foreground_features[
                    class_id
                ].append(sampled_fg)

                fg_sampled_count = int(
                    sampled_fg.shape[0]
                )

            if bg_array.shape[0] > 0:
                sampled_bg = sample_rows(
                    bg_array,
                    args.max_bg_per_image,
                    rng,
                )

                background_features[
                    class_id
                ].append(sampled_bg)

                bg_sampled_count = int(
                    sampled_bg.shape[0]
                )

            sample_stat["classes"][
                str(class_id)
            ] = {
                "gt_pixels_teacher": int(
                    binary_mask.sum()
                ),
                "fg_feature_tokens_available": int(
                    fg_array.shape[0]
                ),
                "bg_feature_tokens_available": int(
                    bg_array.shape[0]
                ),
                "fg_feature_tokens_sampled": int(
                    fg_sampled_count
                ),
                "bg_feature_tokens_sampled": int(
                    bg_sampled_count
                ),
                "has_shape": bbox is not None,
            }

        processed_full_slice_names.append(
            slice_name
        )

        per_image_stats.append(
            sample_stat
        )

        if (index + 1) % 20 == 0:
            print(
                f"[{dataset}] processed Full templates "
                f"{index + 1}/{len(full_records)}"
            )

    if feature_dim is None:
        raise RuntimeError(
            f"No valid Full sample was processed "
            f"for {dataset}"
        )

    if not processed_full_slice_names:
        raise RuntimeError(
            f"No Full sample was successfully processed "
            f"for {dataset}"
        )

    npz_data: dict[str, np.ndarray] = {
        "class_ids": np.asarray(
            class_ids,
            dtype=np.int64,
        )
    }

    stats: dict[str, Any] = {
        "template_version": 3,
        "dataset": dataset,
        "fold": args.fold,
        "method": args.method,
        "seed": args.seed,
        "checkpoint": str(
            args.checkpoint
        ),
        "model_type": MODEL_TYPE,
        "feature_dim": feature_dim,
        "feature_map_hw": (
            list(feature_map_hw)
            if feature_map_hw is not None
            else None
        ),
        "num_full_records": len(
            full_records
        ),
        "num_successfully_processed_full_records": len(
            processed_full_slice_names
        ),
        "num_missing_manifest": len(
            missing_manifest_names
        ),
        "missing_manifest_examples": (
            missing_manifest_names[:20]
        ),
        "class_ids": class_ids,
        "k_fg": args.k_fg,
        "k_bg": args.k_bg,
        "max_fg_per_image": (
            args.max_fg_per_image
        ),
        "max_bg_per_image": (
            args.max_bg_per_image
        ),
        "max_total_features": (
            args.max_total_features
        ),
        "shape_template_size": list(
            SHAPE_TEMPLATE_SIZE
        ),
        "clustering_iterations": (
            KMEANS_ITERATIONS
        ),
        "classes": {},
        "per_image_stats": (
            per_image_stats[:50]
        ),
    }

    audit_classes: dict[str, Any] = {}

    for class_id in class_ids:
        fg_parts = foreground_features[
            class_id
        ]

        bg_parts = background_features[
            class_id
        ]

        if not fg_parts:
            raise RuntimeError(
                "No foreground features collected "
                f"for class {class_id}"
            )

        if not bg_parts:
            raise RuntimeError(
                "No background features collected "
                f"for class {class_id}"
            )

        fg_collected = np.concatenate(
            fg_parts,
            axis=0,
        ).astype(np.float32)

        bg_collected = np.concatenate(
            bg_parts,
            axis=0,
        ).astype(np.float32)

        num_fg_collected = int(
            fg_collected.shape[0]
        )

        num_bg_collected = int(
            bg_collected.shape[0]
        )

        fg = sample_rows(
            fg_collected,
            args.max_total_features,
            rng,
        )

        bg = sample_rows(
            bg_collected,
            args.max_total_features,
            rng,
        )

        proto_fg = spherical_kmeans(
            fg,
            args.k_fg,
            seed=(
                args.seed
                + class_id * 11
            ),
            num_iter=KMEANS_ITERATIONS,
        )

        proto_bg = spherical_kmeans(
            bg,
            args.k_bg,
            seed=(
                args.seed
                + class_id * 17
            ),
            num_iter=KMEANS_ITERATIONS,
        )

        shape_a = (
            np.stack(
                shape_masks[class_id],
                axis=0,
            )
            .mean(axis=0)
            .astype(np.float32)
            if shape_masks[class_id]
            else np.zeros(
                SHAPE_TEMPLATE_SIZE,
                dtype=np.float32,
            )
        )

        npz_data[
            f"proto_fg_c{class_id}"
        ] = proto_fg

        npz_data[
            f"proto_bg_c{class_id}"
        ] = proto_bg

        npz_data[
            f"shape_A_c{class_id}"
        ] = shape_a

        class_stats = {
            "num_fg_features_collected": (
                num_fg_collected
            ),
            "num_bg_features_collected": (
                num_bg_collected
            ),
            "num_fg_features_used": int(
                fg.shape[0]
            ),
            "num_bg_features_used": int(
                bg.shape[0]
            ),
            "num_shape_masks": int(
                len(shape_masks[class_id])
            ),
            "proto_fg_shape": list(
                proto_fg.shape
            ),
            "proto_bg_shape": list(
                proto_bg.shape
            ),
            "shape_A_shape": list(
                shape_a.shape
            ),
            "shape_A_mean": float(
                shape_a.mean()
            ),
            "shape_A_min": float(
                shape_a.min()
            ),
            "shape_A_max": float(
                shape_a.max()
            ),
        }

        stats["classes"][
            str(class_id)
        ] = class_stats

        audit_classes[
            str(class_id)
        ] = {
            "foreground_features_collected": (
                num_fg_collected
            ),
            "background_features_collected": (
                num_bg_collected
            ),
            "foreground_features_used": int(
                fg.shape[0]
            ),
            "background_features_used": int(
                bg.shape[0]
            ),
            "shape_samples": int(
                len(shape_masks[class_id])
            ),
            "foreground_prototype_shape": list(
                proto_fg.shape
            ),
            "background_prototype_shape": list(
                proto_bg.shape
            ),
            "shape_template_shape": list(
                shape_a.shape
            ),
        }

    np.savez_compressed(
        out_npz,
        **npz_data,
    )

    save_json(
        stats,
        out_stats,
    )

    audit: dict[str, Any] = {
        "schema_version": 1,
        "dataset": dataset,
        "fold": args.fold,
        "method": args.method,
        "checkpoint": str(
            args.checkpoint
        ),
        "model_type": MODEL_TYPE,
        "feature_source": (
            "MedSAM ViT-B final image encoder embedding"
        ),
        "encoder_frozen": True,
        "encoder_mode": "eval",
        "gradient_tracking": False,
        "feature_layout": (
            "[H_feature, W_feature, C]"
        ),
        "feature_dim": int(
            feature_dim
        ),
        "feature_map_hw": (
            list(feature_map_hw)
            if feature_map_hw is not None
            else None
        ),
        "feature_normalization": (
            "per-token L2"
        ),
        "gt_to_feature_mapping": (
            "nearest-neighbor"
        ),
        "foreground_definition": (
            "resized_gt == class_id"
        ),
        "background_definition": (
            "resized_gt == 0 over the whole teacher image"
        ),
        "foreground_sampling": (
            "random_without_replacement"
        ),
        "background_sampling": (
            "random_without_replacement"
        ),
        "max_fg_per_image": int(
            args.max_fg_per_image
        ),
        "max_bg_per_image": int(
            args.max_bg_per_image
        ),
        "max_total_features": int(
            args.max_total_features
        ),
        "k_fg": int(
            args.k_fg
        ),
        "k_bg": int(
            args.k_bg
        ),
        "clustering_method": (
            "spherical_kmeans"
        ),
        "clustering_iterations": (
            KMEANS_ITERATIONS
        ),
        "clustering_initialization": (
            "seeded_random_feature_centers"
        ),
        "empty_cluster_handling": (
            "seeded_random_feature_replacement"
        ),
        "similarity": "cosine",
        "downstream_temperature": (
            DOWNSTREAM_TEMPERATURE
        ),
        "shape_template_construction": (
            "crop each class GT by its teacher-space tight bbox, "
            "bilinearly resize to 64x64, then average"
        ),
        "shape_template_size": list(
            SHAPE_TEMPLATE_SIZE
        ),
        "seed": int(
            args.seed
        ),
        "class_ids": class_ids,
        "num_full_records_in_split": len(
            full_records
        ),
        "num_successfully_processed_full_records": len(
            processed_full_slice_names
        ),
        "num_missing_manifest_records": len(
            missing_manifest_names
        ),
        "missing_manifest_examples": (
            missing_manifest_names[:20]
        ),
        "full_slice_names": sorted(
            full_slice_names
        ),
        "processed_full_slice_names": sorted(
            processed_full_slice_names
        ),
        "classes": audit_classes,
        "output_template_path": str(
            out_npz
        ),
        "output_stats_path": str(
            out_stats
        ),
        "feature_extraction_complete": (
            len(processed_full_slice_names)
            == len(full_records)
            and not missing_manifest_names
        ),
    }

    save_json(
        audit,
        out_audit,
    )

    print(
        f"[OK] support template built: "
        f"{dataset}/{args.fold}"
    )
    print(f"     out_npz   = {out_npz}")
    print(f"     out_stats = {out_stats}")
    print(f"     out_audit = {out_audit}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build foreground/background prototypes "
            "and a shape template from Full samples."
        )
    )

    parser.add_argument(
        "--processed_root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--datasets",
        required=True,
    )

    parser.add_argument(
        "--fold",
        default="fold_0",
    )

    parser.add_argument(
        "--method",
        required=True,
        help=(
            "Round-specific method name, for example "
            "idea1_iter_gt_iou_s2026_r0_full5"
        ),
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--k_fg",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--k_bg",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--max_fg_per_image",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--max_bg_per_image",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--max_total_features",
        type=int,
        default=200000,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        print(
            "[WARN] CUDA unavailable; "
            "falling back to CPU"
        )
        args.device = "cpu"

    datasets = [
        name.strip()
        for name in args.datasets.split(",")
        if name.strip()
    ]

    if not datasets:
        raise ValueError("--datasets is empty")

    for dataset in datasets:
        process_dataset(
            args,
            dataset,
        )


if __name__ == "__main__":
    main()
