from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_TYPE = "vit_b"

# ---------------------------------------------------------------------------
# Import from shared pipeline_common
# ---------------------------------------------------------------------------
try:
    from .pipeline_common import (
        METHOD_DEFAULT,
        is_3d_dataset,
        get_round_tag,
        make_run_id,
        build_fold_paths,
        validate_writable_path,
        save_json_atomic,
        check_git_dirty,
        IDEA1_WRITABLE_ROOT,
    )
    from .instance_utils import (
        bbox_iou,
        bbox_from_binary,
        match_instance_mask,
    )
except ImportError:
    from pipeline_common import (  # type: ignore[no-redef]
        METHOD_DEFAULT,
        is_3d_dataset,
        get_round_tag,
        make_run_id,
        build_fold_paths,
        validate_writable_path,
        save_json_atomic,
        check_git_dirty,
        IDEA1_WRITABLE_ROOT,
    )
    from instance_utils import (  # type: ignore[no-redef]
        bbox_iou,
        bbox_from_binary,
        match_instance_mask,
    )


# ============================================================================
# Helper functions (preserved from old 02)
# ============================================================================


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
    feature_np = np.transpose(feature_np, (1, 2, 0))

    # Per-token L2 normalization.
    feature_np /= (
        np.linalg.norm(feature_np, axis=-1, keepdims=True) + 1e-6
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


def resize_binary_mask_to_coverage_area(
    mask: np.ndarray,
    size_hw: tuple[int, int],
) -> np.ndarray:
    """Downsample a binary mask to a coverage map via area averaging.

    Each output value is the fraction of the corresponding input region
    where *mask* is non-zero, i.e. the area coverage in [0, 1].
    """
    height, width = size_hw

    return cv2.resize(
        mask.astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_AREA,
    ).astype(np.float32)


def resize_binary_shape_nearest(
    mask: np.ndarray,
    size_hw: tuple[int, int],
) -> np.ndarray:
    """Resize a binary shape crop with nearest-neighbour to preserve
    hard 0/1 boundaries (not area-averaged)."""
    height, width = size_hw

    return cv2.resize(
        mask.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)


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
    num_iter: int,
) -> np.ndarray:
    if features.ndim != 2 or features.shape[0] == 0:
        raise ValueError(
            "spherical_kmeans requires a non-empty [N, C] array"
        )

    rng = np.random.default_rng(seed)
    n_samples = features.shape[0]
    n_clusters = min(n_clusters, n_samples)

    centers = features[
        rng.choice(n_samples, size=n_clusters, replace=False)
    ].copy()

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
        meta_dir / f"full_box_split_summary_{method}.json"
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


def infer_round_tag_from_method(method: str) -> str:
    """Derive a *round_tag* from a method string like
    ``idea1_iter_gt_iou_s2026_r0_full5`` -> ``r00_full5``.
    """
    m = re.search(r"(r\d+_full\d+)", method)
    if m:
        raw = m.group(1)
        parts = raw.split("_")
        r_num = int(parts[0][1:])
        return f"r{r_num:02d}_{parts[1]}"
    return "unknown_round"






# ============================================================================
# Coverage-based token classification & background ring
# ============================================================================


def _compute_ring_mask(
    feature_h: int,
    feature_w: int,
    bbox_teacher: list[float],
    gt_h: int,
    gt_w: int,
    ring_width_tokens: int,
) -> np.ndarray:
    """Build a boolean ring mask in feature-map coordinates.

    The ring is the region *outside* the instance bounding box but *inside*
    the same bbox expanded by *ring_width_tokens* feature-map tokens.
    The original tight bbox interior is never modified.

    ring_width_tokens == 0 produces an all-False mask (ring disabled).
    """
    ring_f = int(ring_width_tokens)
    if ring_f <= 0:
        return np.zeros((feature_h, feature_w), dtype=bool)

    scale_h = feature_h / gt_h
    scale_w = feature_w / gt_w

    x1, y1, x2, y2 = (float(v) for v in bbox_teacher)

    # Interior bbox in feature space.
    ix1 = max(0, int(x1 * scale_w))
    iy1 = max(0, int(y1 * scale_h))
    ix2 = min(feature_w, int(np.ceil(x2 * scale_w)))
    iy2 = min(feature_h, int(np.ceil(y2 * scale_h)))

    # Expanded bbox (ring outer boundary).
    ex1 = max(0, ix1 - ring_f)
    ey1 = max(0, iy1 - ring_f)
    ex2 = min(feature_w, ix2 + ring_f)
    ey2 = min(feature_h, iy2 + ring_f)

    ring_mask = np.zeros((feature_h, feature_w), dtype=bool)
    ring_mask[ey1:ey2, ex1:ex2] = True

    # Subtract interior.
    if ix1 < ix2 and iy1 < iy2:
        ring_mask[iy1:iy2, ix1:ix2] = False

    return ring_mask


# ============================================================================
# Adaptive multi-shape clustering
# ============================================================================


def _adaptive_shape_clustering(
    shapes: np.ndarray,
    support_unit_ids: np.ndarray,
    Kmax: int,
    min_support: int,
    seed: int,
    num_iter: int,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Cluster flattened shape vectors.

    Parameters
    ----------
    shapes : (N, 4096) float32
        L2-normalised flattened 64x64 shape masks.
    support_unit_ids : (N,) str
        Independent support unit per shape (slice_name for 2D, case_id for 3D).
    Kmax : int
        Maximum number of clusters to try.
    min_support : int
        Minimum number of independent support units per cluster.
    seed : int
        Random seed.

    Returns
    -------
    best_K : int
    labels : (N,) int64
    centers : (best_K, 4096) float32
    """
    N = shapes.shape[0]

    if N == 0:
        raise ValueError("No shapes to cluster")

    max_possible_K = min(Kmax, N // min_support)
    if max_possible_K < 1:
        max_possible_K = 1

    best_result: tuple[int, np.ndarray, np.ndarray] | None = None
    best_score = -1.0

    for candidate_K in range(1, max_possible_K + 1):
        if candidate_K == 1:
            # Single cluster: mean is the centroid.
            center = shapes.mean(axis=0, keepdims=True)
            center = center / (np.linalg.norm(center, axis=1, keepdims=True) + 1e-6)
            centers = center.astype(np.float32)
            labels = np.zeros(N, dtype=np.int64)
        else:
            centers = spherical_kmeans(
                shapes,
                n_clusters=candidate_K,
                seed=seed + candidate_K,
                num_iter=num_iter,
            )
            labels = (shapes @ centers.T).argmax(axis=1).astype(np.int64)

        # Support constraint.
        ok = True
        for k in range(candidate_K):
            cluster_units = set(support_unit_ids[labels == k])
            if len(cluster_units) < min_support:
                ok = False
                break

        if not ok:
            continue

        # Score: average intra-cluster cosine similarity.
        sims = np.sum(shapes * centers[labels], axis=1)
        score = float(np.mean(sims))

        if score > best_score:
            best_score = score
            best_result = (candidate_K, labels.copy(), centers.copy())

    if best_result is None:
        # All K > 1 failed support constraints; fall back to K=1.
        center = shapes.mean(axis=0, keepdims=True)
        center = center / (np.linalg.norm(center, axis=1, keepdims=True) + 1e-6)
        return 1, np.zeros(N, dtype=np.int64), center.astype(np.float32)

    return best_result


def _build_cluster_templates(
    shapes: np.ndarray,
    labels: np.ndarray,
    K: int,
    shape_size: int,
) -> list[np.ndarray]:
    """Return per-cluster shape templates, each *(shape_size, shape_size) float32*."""
    templates: list[np.ndarray] = []
    for k in range(K):
        cluster_shapes = shapes[labels == k]
        if cluster_shapes.shape[0] > 0:
            tmpl = cluster_shapes.mean(axis=0).reshape(shape_size, shape_size)
        else:
            tmpl = np.zeros((shape_size, shape_size), dtype=np.float32)
        templates.append(np.clip(tmpl, 0.0, 1.0).astype(np.float32))
    return templates


def _compute_semantic_centers(
    fg_tokens_by_instance: list[np.ndarray],
    shape_labels: np.ndarray,
    K: int,
    feature_dim: int,
) -> np.ndarray:
    """Return per-cluster semantic centers, shape *(K, D)*, L2-normalised."""
    centers = np.zeros((K, feature_dim), dtype=np.float32)
    for k in range(K):
        cluster_tokens = [
            fg_tokens_by_instance[i]
            for i in range(len(fg_tokens_by_instance))
            if shape_labels[i] == k
        ]
        if cluster_tokens:
            all_tokens = np.concatenate(cluster_tokens, axis=0)
            center = all_tokens.mean(axis=0).astype(np.float32)
            center = center / (np.linalg.norm(center) + 1e-6)
            centers[k] = center
    return centers


# ============================================================================
# Main processing
# ============================================================================


def process_dataset(
    args: argparse.Namespace,
    dataset: str,
) -> None:
    started_at = time.time()
    rng = np.random.default_rng(args.seed)

    round_tag = args.round_tag or infer_round_tag_from_method(args.method)

    paths = build_fold_paths(
        args.processed_root,
        dataset,
        args.fold,
        method=args.method,
        round_tag=round_tag,
    )
    run_id = paths["run_id"]
    fold_root = paths["fold_root"]
    meta_dir = paths["meta_dir"]

    manifest_path = meta_dir / "manifest.json"
    prompts_path = fold_root / "prompts" / "prompts_train.json"
    split_path = paths["split_path"]

    out_npz = paths["support_path"]
    out_stats = paths["support_stats_path"]

    # ---- validate inputs exist ----
    for required_path in (manifest_path, split_path, args.checkpoint):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required input does not exist: {required_path}"
            )

    if not prompts_path.exists():
        raise FileNotFoundError(
            f"Prompts file does not exist: {prompts_path}"
        )

    # ---- overwrite check ----
    if not args.overwrite:
        for path in (out_npz, out_stats):
            if path.exists():
                raise FileExistsError(
                    f"Output already exists: {path}. Use --overwrite to replace."
                )

    # ---- load inputs ----
    manifest = load_json(manifest_path)
    split_records = load_json(split_path)
    prompts = load_json(prompts_path)

    if not isinstance(manifest, list):
        raise TypeError(
            f"manifest.json must contain a list: {manifest_path}"
        )
    if not isinstance(split_records, list):
        raise TypeError(
            f"Full/Box split must contain a list: {split_path}"
        )
    if not isinstance(prompts, dict):
        raise TypeError(
            f"Prompts must be a dict: {prompts_path}"
        )

    manifest_by_slice = build_unique_manifest_index(manifest)

    # ---- filter Full records ----
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
            raise TypeError("Each Full/Box record must be a dict")
        full_slice_names.append(get_slice_name(record))

    if len(full_slice_names) != len(set(full_slice_names)):
        raise RuntimeError(
            "Duplicate Full slice names found in split file"
        )

    class_ids = read_class_ids(meta_dir, args.method, full_records)

    print(
        f"[CLASS_IDS] {dataset}/{args.fold}: {class_ids}"
    )
    print(f"[ROUND_TAG] {round_tag}")
    print(f"[RUN_ID] {run_id}")

    is_3d = is_3d_dataset(dataset)

    # ---- build MedSAM encoder ----
    device = torch.device(args.device)
    model = build_medsam(args.checkpoint, device)

    # ---- per-class collectors ----
    # fg/bg bank: class_id -> list of per-instance token arrays
    fg_bank_parts: dict[int, list[np.ndarray]] = {
        c: [] for c in class_ids
    }
    bg_bank_parts: dict[int, list[np.ndarray]] = {
        c: [] for c in class_ids
    }

    # Shape: class_id -> list of (shape_size, shape_size) float32
    shape_list: dict[int, list[np.ndarray]] = {
        c: [] for c in class_ids
    }
    # Per-shape metadata for clustering.
    shape_support_units: dict[int, list[str]] = {
        c: [] for c in class_ids
    }
    # Per-shape -> index into fg_bank_parts for semantic center.
    shape_instance_fg_index: dict[int, list[int]] = {
        c: [] for c in class_ids
    }

    # Per-class fallback tracking.
    fg_fallback_instance_count_c: dict[int, int] = {
        c: 0 for c in class_ids
    }
    fg_fallback_token_count_c: dict[int, int] = {
        c: 0 for c in class_ids
    }

    # 3D per-case tracking.
    case_fg_parts: dict[str, dict[int, list[np.ndarray]]] = {}
    case_bg_parts: dict[str, dict[int, list[np.ndarray]]] = {}

    per_instance_stats: list[dict[str, Any]] = []
    missing_manifest_names: list[str] = []
    empty_target_count = 0
    skipped_or_invalid_count = 0
    total_instance_count = 0
    zero_bg_instance_count = 0
    instances_with_bg_tokens = 0
    instances_without_bg_tokens = 0
    fg_fallback_instance_count = 0
    fg_fallback_token_count = 0
    fg_fallback_coverage_all: list[float] = []
    total_bg_ring_candidate_tokens = 0
    total_valid_global_bg_tokens = 0
    global_non_bg_coverage_ring_sum = 0.0
    global_non_bg_coverage_ring_count = 0
    global_non_bg_coverage_ring_min: float | None = None
    global_non_bg_coverage_ring_max: float | None = None
    processed_full_slice_names: list[str] = []
    feature_dim: int | None = None
    feature_map_hw: tuple[int, int] | None = None

    # ---- iterate Full records ----
    for index, record in enumerate(full_records):
        slice_name = get_slice_name(record)
        item = manifest_by_slice.get(slice_name)

        if item is None:
            missing_manifest_names.append(slice_name)
            print(f"[WARN] missing in manifest: {slice_name}")
            skipped_or_invalid_count += 1
            continue

        if item.get("split") != "train":
            raise RuntimeError(
                f"A Full sample is not in train split: {slice_name}"
            )

        # Resolve image / GT paths.
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

        # Get prompt instances for this slice.
        prompt_meta = prompts.get(slice_name)
        if prompt_meta is None:
            print(
                f"[WARN] missing in prompts: {slice_name}"
            )
            skipped_or_invalid_count += 1
            continue

        instances = prompt_meta.get("instances")
        if not isinstance(instances, list) or not instances:
            # Verify GT is genuinely empty before skipping.
            gt = load_mask_npy(gt_path)
            fg_mask = gt != 0
            fg_values = gt[fg_mask]
            fg_count = int((fg_values != 255).sum())
            if fg_count > 0:
                raise RuntimeError(
                    f"Prompt has 0 instances but GT has "
                    f"{fg_count} foreground pixels for: "
                    f"{slice_name}. Data inconsistency — "
                    f"prompt and GT diverge."
                )
            print(
                f"[WARN] no instances in prompt for: {slice_name} "
                f"(GT verified: 0 foreground pixels)"
            )
            continue

        case_id = str(item.get("case_id", "unknown_case"))

        # Load image and GT.
        image = load_image_npy(image_path)
        gt = load_mask_npy(gt_path)

        if image.shape[:2] != gt.shape[:2]:
            raise ValueError(
                f"Image/GT shape mismatch for {slice_name}: "
                f"image={image.shape}, gt={gt.shape}"
            )

        gt_h, gt_w = gt.shape

        # Extract encoder features.
        feature = extract_encoder_feature(model, image, device)

        current_feature_dim = int(feature.shape[-1])
        current_feature_hw = (
            int(feature.shape[0]),
            int(feature.shape[1]),
        )

        if feature_dim is None:
            feature_dim = current_feature_dim
        elif feature_dim != current_feature_dim:
            raise RuntimeError(
                "Feature dimension changed across samples: "
                f"expected={feature_dim}, found={current_feature_dim}, "
                f"slice={slice_name}"
            )

        if feature_map_hw is None:
            feature_map_hw = current_feature_hw
        elif feature_map_hw != current_feature_hw:
            raise RuntimeError(
                "Feature-map size changed across samples: "
                f"expected={feature_map_hw}, found={current_feature_hw}, "
                f"slice={slice_name}"
            )

        feature_h, feature_w = current_feature_hw

        slice_stat: dict[str, Any] = {
            "slice_name": slice_name,
            "case_id": case_id,
            "image_path": str(image_path),
            "gt_path": str(gt_path),
            "teacher_image_hw": [gt_h, gt_w],
            "feature_map_hw": [feature_h, feature_w],
            "instances": [],
        }

        # Global non-background coverage (computed once per image).
        # Everything that is not 0 is treated as non-reliable-background:
        #   - current instance foreground
        #   - other instances (same or different class)
        #   - foreground classes not in class_ids
        #   - ignore label 255
        global_non_bg_mask = (gt != 0)
        global_non_bg_coverage = resize_binary_mask_to_coverage_area(
            global_non_bg_mask.astype(np.float32),
            (feature_h, feature_w),
        )
        valid_global_bg_mask = (
            global_non_bg_coverage <= args.max_global_non_bg_coverage
        )

        # ---- process each instance ----
        for inst in instances:
            label_id = int(inst["label_id"])
            if label_id not in class_ids:
                continue

            bbox = inst.get("bbox_teacher") or inst.get("bbox")
            if bbox is None:
                raise ValueError(
                    f"Instance missing bbox for label_id={label_id} "
                    f"in slice={slice_name}"
                )
            bbox = [float(v) for v in bbox]

            component_id = inst.get("component_id")
            if component_id is not None:
                component_id = int(component_id)

            total_instance_count += 1

            # Match instance GT.
            try:
                inst_mask = match_instance_mask(
                    gt, label_id, bbox, component_id
                )
            except ValueError as exc:
                raise RuntimeError(
                    f"Instance matching failed for label_id={label_id}, "
                    f"component_id={component_id}, slice={slice_name}: {exc}"
                ) from exc

            if not inst_mask.any():
                empty_target_count += 1
                raise RuntimeError(
                    f"Empty target for label_id={label_id}, "
                    f"slice={slice_name}"
                )

            # Coverage in feature space (area-averaged).
            inst_coverage = resize_binary_mask_to_coverage_area(
                inst_mask.astype(np.float32),
                (feature_h, feature_w),
            )

            # Token classification.
            # Foreground: instance-level coverage (unchanged).
            fg_mask = inst_coverage >= args.fg_coverage_threshold

            # Background ring (feature-token units, does not modify bbox).
            ring_mask = _compute_ring_mask(
                feature_h, feature_w,
                bbox, gt_h, gt_w,
                args.bg_ring_width_tokens,
            )

            # Reliable background = ring AND global non-bg coverage <= max.
            final_bg_mask = valid_global_bg_mask & ring_mask

            # Ring-level global-non-bg-coverage statistics.
            ring_indices = np.where(ring_mask)
            ring_count = int(ring_indices[0].size)
            total_bg_ring_candidate_tokens += ring_count
            if ring_count > 0:
                ring_coverage_values = global_non_bg_coverage[ring_indices]
                global_non_bg_coverage_ring_sum += float(ring_coverage_values.sum())
                global_non_bg_coverage_ring_count += ring_count
                cmin = float(ring_coverage_values.min())
                cmax = float(ring_coverage_values.max())
                if global_non_bg_coverage_ring_min is None or cmin < global_non_bg_coverage_ring_min:
                    global_non_bg_coverage_ring_min = cmin
                if global_non_bg_coverage_ring_max is None or cmax > global_non_bg_coverage_ring_max:
                    global_non_bg_coverage_ring_max = cmax

            # Sample tokens.
            fg_indices = np.where(fg_mask)
            bg_indices = np.where(final_bg_mask)

            fg_tokens = feature[fg_indices]
            bg_tokens = feature[bg_indices]

            fg_available = int(fg_tokens.shape[0])
            bg_available = int(bg_tokens.shape[0])
            total_valid_global_bg_tokens += bg_available

            fg_source = "primary"
            fg_fallback_coverage = None
            if fg_available == 0:
                # Deterministic fallback: select cells where
                # inst_coverage > 0 within the instance bbox,
                # sorted by coverage desc → row asc → col asc.
                fx1 = max(0, int(bbox[0] * feature_w / gt_w))
                fy1 = max(0, int(bbox[1] * feature_h / gt_h))
                fx2 = min(feature_w, int(np.ceil(bbox[2] * feature_w / gt_w)))
                fy2 = min(feature_h, int(np.ceil(bbox[3] * feature_h / gt_h)))

                bbox_mask_f = np.zeros(
                    (feature_h, feature_w), dtype=bool
                )
                if fy2 > fy1 and fx2 > fx1:
                    bbox_mask_f[fy1:fy2, fx1:fx2] = True

                candidate_mask = (inst_coverage > 0) & bbox_mask_f
                cand_rows, cand_cols = np.where(candidate_mask)
                n_candidates = int(cand_rows.size)


                # Robust fallback for small/thin organs:
                # token-level bbox masks can miss all positive coverage cells.
                # If bbox-constrained fallback is empty, use all positive
                # instance-coverage tokens rather than aborting bank construction.
                if n_candidates == 0:
                    candidate_mask = inst_coverage > 0
                    cand_rows, cand_cols = np.where(candidate_mask)
                    n_candidates = int(len(cand_rows))

                if n_candidates == 0:
                    max_cov = float(inst_coverage.max())
                    raise RuntimeError(
                        f"Zero foreground tokens after coverage "
                        f"filtering and zero fallback candidates "
                        f"(max inst_coverage={max_cov:.4f}) "
                        f"for label_id={label_id}, slice={slice_name}"
                    )

                cand_coverage = inst_coverage[cand_rows, cand_cols]
                # lexsort: coverage desc, row asc, col asc.
                order = np.lexsort((
                    cand_cols,       # col asc  (tie-breaker 3)
                    cand_rows,       # row asc  (tie-breaker 2)
                    -cand_coverage,  # coverage desc (primary)
                ))
                cand_rows = cand_rows[order]
                cand_cols = cand_cols[order]
                cand_coverage = cand_coverage[order]

                n_select = min(
                    n_candidates, args.max_fg_fallback_tokens
                )
                fallback_rows = cand_rows[:n_select]
                fallback_cols = cand_cols[:n_select]

                fg_tokens = feature[fallback_rows, fallback_cols]
                fg_available = int(fg_tokens.shape[0])
                fg_source = "fallback"
                fg_fallback_coverage = [
                    float(cand_coverage[i]) for i in range(n_select)
                ]
                fg_fallback_instance_count += 1
                fg_fallback_token_count += fg_available
                fg_fallback_coverage_all.extend(fg_fallback_coverage)
                fg_fallback_instance_count_c[label_id] += 1
                fg_fallback_token_count_c[label_id] += fg_available

            sampled_fg = sample_rows(
                fg_tokens, args.max_fg_per_instance, rng
            )
            if bg_available > 0:
                sampled_bg = sample_rows(
                    bg_tokens, args.max_bg_per_instance, rng
                )
                instances_with_bg_tokens += 1
            else:
                sampled_bg = np.zeros((0, feature_dim), dtype=np.float32)
                zero_bg_instance_count += 1
                instances_without_bg_tokens += 1

            fg_sampled = int(sampled_fg.shape[0])
            bg_sampled = int(sampled_bg.shape[0])

            # Record per-instance fg index for semantic center computation.
            fg_instance_idx = len(fg_bank_parts[label_id])

            # Store.
            if fg_sampled > 0:
                fg_bank_parts[label_id].append(sampled_fg)
                shape_instance_fg_index[label_id].append(fg_instance_idx)
            if bg_sampled > 0:
                bg_bank_parts[label_id].append(sampled_bg)

            # Shape extraction.
            shape_bbox = bbox_from_binary(inst_mask)
            has_shape = shape_bbox is not None
            if has_shape:
                sx1, sy1, sx2, sy2 = shape_bbox
                shape_crop = inst_mask[sy1:sy2, sx1:sx2].astype(np.float32)
                shape64 = np.clip(
                    resize_binary_shape_nearest(shape_crop, (args.shape_size, args.shape_size)),
                    0.0, 1.0,
                )
                shape_list[label_id].append(shape64.astype(np.float32))
                # Support unit: case_id for 3D, slice_name for 2D.
                shape_support_units[label_id].append(
                    case_id if is_3d else slice_name
                )

            # 3D per-case tracking.
            if is_3d:
                case_fg_parts.setdefault(case_id, {}).setdefault(
                    label_id, []
                ).append(sampled_fg)
                case_bg_parts.setdefault(case_id, {}).setdefault(
                    label_id, []
                ).append(sampled_bg)

            slice_stat["instances"].append({
                "label_id": label_id,
                "component_id": component_id,
                "bbox_teacher": bbox,
                "fg_tokens_available": fg_available,
                "bg_tokens_available": bg_available,
                "fg_tokens_sampled": fg_sampled,
                "bg_tokens_sampled": bg_sampled,
                "has_shape": has_shape,
                "fg_source": fg_source,
                "fg_fallback_coverage": fg_fallback_coverage,
            })

        processed_full_slice_names.append(slice_name)
        per_instance_stats.append(slice_stat)

        if (index + 1) % 20 == 0:
            print(
                f"[{dataset}] processed Full records "
                f"{index + 1}/{len(full_records)}"
            )

    # ---- post-loop validation ----
    if feature_dim is None:
        raise RuntimeError(
            f"No valid Full sample was processed for {dataset}"
        )
    if not processed_full_slice_names:
        raise RuntimeError(
            f"No Full sample was successfully processed for {dataset}"
        )
    if skipped_or_invalid_count != 0:
        raise RuntimeError(
            f"skipped_or_invalid_count={skipped_or_invalid_count}, "
            f"must be 0. Check missing manifest/prompt entries above."
        )

    # ---- 3D per-case capping ----
    if is_3d:
        for label_id in class_ids:
            num_cases = len(case_fg_parts)
            if num_cases == 0:
                continue
            per_case_cap = max(
                128,
                int(np.ceil(args.max_fg_per_class / num_cases)),
            )

            # Collect per-case capped FG.
            new_fg_parts: list[np.ndarray] = []
            for case_id, class_dict in case_fg_parts.items():
                parts = class_dict.get(label_id, [])
                if not parts:
                    continue
                case_all = np.concatenate(parts, axis=0)
                capped = sample_rows(case_all, per_case_cap, rng)
                if capped.shape[0] > 0:
                    new_fg_parts.append(capped)
            if new_fg_parts:
                fg_bank_parts[label_id] = new_fg_parts

            # Same for BG.
            new_bg_parts: list[np.ndarray] = []
            for case_id, class_dict in case_bg_parts.items():
                parts = class_dict.get(label_id, [])
                if not parts:
                    continue
                case_all = np.concatenate(parts, axis=0)
                capped = sample_rows(case_all, per_case_cap, rng)
                if capped.shape[0] > 0:
                    new_bg_parts.append(capped)
            if new_bg_parts:
                bg_bank_parts[label_id] = new_bg_parts

    # ---- per-class bank capping + shape clustering ----
    npz_data: dict[str, np.ndarray] = {
        "class_ids": np.asarray(class_ids, dtype=np.int64),
        "feature_dim": np.array(feature_dim, dtype=np.int64),
        "shape_size": np.array([args.shape_size, args.shape_size], dtype=np.int64),
        "method": np.array(args.method),
        "round_tag": np.array(round_tag),
        "run_id": np.array(run_id),
    }

    stats: dict[str, Any] = {
        "schema_version": 1,
        "support_stats_schema_version": 2,
        "method": args.method,
        "round_tag": round_tag,
        "run_id": run_id,
        "dataset": dataset,
        "fold": args.fold,
        "seed": args.seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_exists": args.checkpoint.is_file(),
        "git_dirty": check_git_dirty(REPO_ROOT),
        "device": str(device),
        "model_type": MODEL_TYPE,
        "feature_dim": feature_dim,
        "feature_height": feature_map_hw[0],
        "feature_width": feature_map_hw[1],
        "shape_size": [args.shape_size, args.shape_size],
        "num_full_records": len(full_records),
        "num_full_instances": total_instance_count,
        "num_full_cases": len(case_fg_parts) if is_3d else len(
            set(str(manifest_by_slice.get(get_slice_name(r), {}).get("case_id", ""))
                for r in full_records)
        ),
        "skipped_or_invalid_count": skipped_or_invalid_count,
        "num_empty_targets": empty_target_count,
        "zero_bg_instance_count": zero_bg_instance_count,
        "instances_with_bg_tokens": instances_with_bg_tokens,
        "instances_without_bg_tokens": instances_without_bg_tokens,
        "fg_fallback_instance_count": fg_fallback_instance_count,
        "fg_fallback_token_count": fg_fallback_token_count,
        "instances_with_primary_fg": total_instance_count - fg_fallback_instance_count,
        "instances_with_fallback_fg": fg_fallback_instance_count,
        "fg_primary_coverage_threshold": args.fg_coverage_threshold,
        "fg_fallback_coverage_min": float(min(fg_fallback_coverage_all)) if fg_fallback_coverage_all else None,
        "fg_fallback_coverage_mean": float(np.mean(fg_fallback_coverage_all)) if fg_fallback_coverage_all else None,
        "fg_fallback_coverage_max": float(max(fg_fallback_coverage_all)) if fg_fallback_coverage_all else None,
        "class_ids": class_ids,
        "fg_coverage_threshold": args.fg_coverage_threshold,
        "max_global_non_bg_coverage": args.max_global_non_bg_coverage,
        "bg_ring_width_tokens": args.bg_ring_width_tokens,
        "total_bg_ring_candidate_tokens": total_bg_ring_candidate_tokens,
        "total_valid_global_bg_tokens": total_valid_global_bg_tokens,
        "bg_ring_valid_ratio": (
            total_valid_global_bg_tokens
            / max(total_bg_ring_candidate_tokens, 1)
        ),
        "global_non_bg_coverage_ring_min": (
            float(global_non_bg_coverage_ring_min)
            if global_non_bg_coverage_ring_min is not None
            else None
        ),
        "global_non_bg_coverage_ring_mean": (
            float(global_non_bg_coverage_ring_sum
                  / max(global_non_bg_coverage_ring_count, 1))
            if global_non_bg_coverage_ring_count > 0
            else None
        ),
        "global_non_bg_coverage_ring_max": (
            float(global_non_bg_coverage_ring_max)
            if global_non_bg_coverage_ring_max is not None
            else None
        ),
        "max_fg_per_instance": args.max_fg_per_instance,
        "max_bg_per_instance": args.max_bg_per_instance,
        "max_fg_per_class": args.max_fg_per_class,
        "max_bg_per_class": args.max_bg_per_class,
        "shape_size": args.shape_size,
        "kmax_shape": args.kmax_shape,
        "cluster_max_iter": args.cluster_max_iter,
        "min_cluster_support": 2,
        "max_fg_fallback_tokens": args.max_fg_fallback_tokens,
        "classes": {},
        "per_instance_stats": per_instance_stats[:50],
        "input_paths": {
            "processed_root": str(args.processed_root),
            "manifest": str(manifest_path),
            "split": str(split_path),
            "prompts": str(prompts_path),
        },
        "output_paths": {
            "npz": str(out_npz),
            "stats": str(out_stats),
        },
    }

    for class_id in class_ids:
        # --- FG bank ---
        fg_parts = fg_bank_parts.get(class_id, [])
        if not fg_parts:
            raise RuntimeError(
                f"No foreground features collected for class {class_id}"
            )

        fg_all = np.concatenate(fg_parts, axis=0).astype(np.float32)
        fg_bank = sample_rows(fg_all, args.max_fg_per_class, rng)

        # --- BG bank ---
        bg_parts = bg_bank_parts.get(class_id, [])
        if bg_parts:
            bg_all = np.concatenate(bg_parts, axis=0).astype(np.float32)
            bg_bank = sample_rows(bg_all, args.max_bg_per_class, rng)
        else:
            raise RuntimeError(
                f"Empty background bank for class {class_id} in "
                f"{dataset}/{args.fold}, round_tag={round_tag}. "
                f"No reliable global-background tokens were collected."
            )

        # --- Shape clustering ---
        shapes = shape_list.get(class_id, [])
        supp_units = shape_support_units.get(class_id, [])
        fg_indices_for_class = shape_instance_fg_index.get(class_id, [])

        if shapes:
            shape_matrix = np.stack(shapes, axis=0).reshape(len(shapes), -1).astype(np.float32)
            # L2 normalize each shape.
            shape_norms = np.linalg.norm(shape_matrix, axis=1, keepdims=True) + 1e-6
            shape_matrix_norm = shape_matrix / shape_norms

            supp_arr = np.array(supp_units, dtype=str)

            K_shape, shape_labels, shape_centers = _adaptive_shape_clustering(
                shape_matrix_norm,
                supp_arr,
                args.kmax_shape,
                2,
                args.seed + class_id * 7,
                num_iter=args.cluster_max_iter,
            )

            shape_templates = _build_cluster_templates(
                shape_matrix, shape_labels, K_shape,
                shape_size=args.shape_size,
            )

            # Collect per-instance FG tokens for semantic center.
            inst_fg_list: list[np.ndarray] = []
            for idx in fg_indices_for_class:
                if idx < len(fg_parts):
                    inst_fg_list.append(fg_parts[idx])
                else:
                    inst_fg_list.append(
                        np.zeros((0, feature_dim), dtype=np.float32)
                    )

            semantic_centers = _compute_semantic_centers(
                inst_fg_list, shape_labels, K_shape, feature_dim,
            )

            cluster_instance_counts = np.array(
                [int((shape_labels == k).sum()) for k in range(K_shape)],
                dtype=np.int64,
            )
            cluster_support_counts = np.array(
                [len(set(supp_arr[shape_labels == k])) for k in range(K_shape)],
                dtype=np.int64,
            )
            candidate_Ks: list[int] = [K_shape]
        else:
            K_shape = 1
            shape_templates = [np.zeros((args.shape_size, args.shape_size), dtype=np.float32)]
            semantic_centers = np.zeros((1, feature_dim), dtype=np.float32)
            cluster_instance_counts = np.array([0], dtype=np.int64)
            cluster_support_counts = np.array([0], dtype=np.int64)
            candidate_Ks = [1]

        shape_templates_arr = np.stack(shape_templates, axis=0).astype(np.float32)

        # --- Store new NPZ keys ---
        npz_data[f"bank_fg_c{class_id}"] = fg_bank
        npz_data[f"bank_bg_c{class_id}"] = bg_bank
        npz_data[f"shape_templates_c{class_id}"] = shape_templates_arr
        npz_data[f"shape_semantic_centers_c{class_id}"] = semantic_centers
        npz_data[f"shape_cluster_instance_counts_c{class_id}"] = cluster_instance_counts
        npz_data[f"shape_cluster_support_counts_c{class_id}"] = cluster_support_counts

        # --- Per-class stats ---
        cluster_support_unit_ids: list[list[str]] = []
        if shapes:
            supp_arr = np.array(supp_units, dtype=str)
            for k in range(K_shape):
                cluster_support_unit_ids.append(
                    sorted(set(supp_arr[shape_labels == k]))
                )

        stats["classes"][str(class_id)] = {
            "num_instances": len(shapes),
            "num_support_units": len(set(supp_units)) if supp_units else 0,
            "fg_tokens_collected": int(fg_all.shape[0]),
            "bg_tokens_collected": int(bg_all.shape[0]) if bg_parts else 0,
            "fg_tokens_bank": int(fg_bank.shape[0]),
            "bg_tokens_bank": int(bg_bank.shape[0]),
            "fg_fallback_instance_count": fg_fallback_instance_count_c.get(class_id, 0),
            "fg_fallback_token_count": fg_fallback_token_count_c.get(class_id, 0),
            "K_shape": K_shape,
            "candidate_Ks": candidate_Ks,
            "cluster_instance_counts": cluster_instance_counts.tolist(),
            "cluster_support_counts": cluster_support_counts.tolist(),
            "cluster_support_unit_ids": cluster_support_unit_ids,
            "cluster_template_means": [
                float(tmpl.mean()) for tmpl in shape_templates
            ],
            "cluster_template_maxs": [
                float(tmpl.max()) for tmpl in shape_templates
            ],
        }

    # ---- save NPZ ----
    validate_writable_path(out_npz)
    import tempfile
    # Write to a sibling temporary file then atomically replace.
    fd, tmp_name = tempfile.mkstemp(
        suffix=".npz",
        prefix=".tmp_",
        dir=str(out_npz.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        np.savez_compressed(tmp_path, **npz_data)
        os.replace(tmp_path, out_npz)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    # ---- save stats ----
    stats["elapsed_seconds"] = round(time.time() - started_at, 3)
    save_json_atomic(stats, out_stats)

    print(
        f"[OK] support bank + templates built: "
        f"{dataset}/{args.fold}"
    )
    print(f"     out_npz   = {out_npz}")
    print(f"     out_stats = {out_stats}")
    print(f"     elapsed   = {stats['elapsed_seconds']:.1f}s")


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build foreground/background feature banks and adaptive "
            "multi-shape templates from Full samples."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required.
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
        "--method",
        required=True,
        help=(
            "Round-specific method name, for example "
            "idea1_iter_gt_iou_s2026_r0_full5"
        ),
    )

    # Optional.
    parser.add_argument(
        "--fold",
        default="fold_0",
    )
    parser.add_argument(
        "--round_tag",
        default=None,
        help=(
            "Round tag like 'r00_full5'. "
            "If not provided, derived from --method via regex."
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
        "--overwrite",
        action="store_true",
    )

    # Bank parameters.
    bank_group = parser.add_argument_group("Feature Bank")
    bank_group.add_argument(
        "--fg_coverage_threshold",
        type=float,
        default=0.90,
        help="Minimum coverage for foreground tokens.",
    )
    bank_group.add_argument(
        "--max_global_non_bg_coverage",
        type=float,
        default=0.10,
        help=(
            "Maximum allowed global non-background area fraction "
            "for a reliable background token. "
            "0.10 means at most 10%% of the feature-token region "
            "may be occupied by any foreground (any class/instance) "
            "or ignore (255) pixels."
        ),
    )
    bank_group.add_argument(
        "--max_fg_per_instance",
        type=int,
        default=128,
        help="Max foreground tokens sampled per instance.",
    )
    bank_group.add_argument(
        "--max_bg_per_instance",
        type=int,
        default=128,
        help="Max background tokens sampled per instance.",
    )
    bank_group.add_argument(
        "--max_fg_per_class",
        type=int,
        default=4096,
        help="Max foreground tokens in bank per class.",
    )
    bank_group.add_argument(
        "--max_bg_per_class",
        type=int,
        default=4096,
        help="Max background tokens in bank per class.",
    )
    bank_group.add_argument(
        "--bg_ring_width_tokens",
        type=int,
        default=1,
        help=(
            "Width of the exterior local-background ring in "
            "feature-map tokens. This does not modify the "
            "original tight box prompt. 0 disables the ring."
        ),
    )
    bank_group.add_argument(
        "--max_fg_fallback_tokens",
        type=int,
        default=4,
        help=(
            "Maximum fallback FG tokens when primary coverage "
            "filtering yields zero tokens. Fallback selects "
            "cells with inst_coverage > 0, sorted "
            "deterministically."
        ),
    )

    # Shape clustering parameters.
    shape_group = parser.add_argument_group("Shape Clustering")
    shape_group.add_argument(
        "--shape_size",
        type=int,
        default=64,
        help="Shape template size (square).",
    )
    shape_group.add_argument(
        "--kmax_shape",
        type=int,
        default=5,
        help="Maximum number of shape clusters per class.",
    )
    shape_group.add_argument(
        "--cluster_max_iter",
        type=int,
        default=25,
        help="Maximum k-means iterations.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError(f"--seed must be >= 0, got {args.seed}")

    for name in (
        "max_fg_per_instance",
        "max_bg_per_instance",
        "max_fg_per_class",
        "max_bg_per_class",
        "shape_size",
        "kmax_shape",
        "cluster_max_iter",
        "max_fg_fallback_tokens",
    ):
        value = int(getattr(args, name))
        if value <= 0:
            raise ValueError(f"--{name} must be positive, got {value}")

    if args.bg_ring_width_tokens < 0:
        raise ValueError(
            f"--bg_ring_width_tokens must be >= 0, "
            f"got {args.bg_ring_width_tokens}"
        )

    if not (0.0 <= args.fg_coverage_threshold <= 1.0):
        raise ValueError(
            f"--fg_coverage_threshold must be in [0, 1], "
            f"got {args.fg_coverage_threshold}"
        )
    if not (0.0 <= args.max_global_non_bg_coverage <= 1.0):
        raise ValueError(
            f"--max_global_non_bg_coverage must be in [0, 1], "
            f"got {args.max_global_non_bg_coverage}"
        )



def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    if (
        args.device.startswith("cuda")
        and not torch.cuda.is_available()
    ):
        print("[WARN] CUDA unavailable; falling back to CPU")
        args.device = "cpu"

    datasets = [
        name.strip()
        for name in args.datasets.split(",")
        if name.strip()
    ]

    if not datasets:
        raise ValueError("--datasets is empty")

    for dataset in datasets:
        process_dataset(args, dataset)


if __name__ == "__main__":
    main()
