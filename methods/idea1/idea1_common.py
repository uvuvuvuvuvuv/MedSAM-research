#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared contracts and utilities for Idea1 hard-Full MedSAM fine-tuning.

The module intentionally has no dependency on the historical SAC implementation.
It reads the frozen baseline manifest/prompt/geometry contracts and writes only to
an Idea1-specific workspace.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

METHOD_DEFAULT = "idea1_hard_full_medsam_ft"
KNOWN_3D_DATASETS = {"btcv", "synapse", "acdc", "prostate158"}
IGNORE_LABEL = 255


def _coerce_spacing3(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        values = [float(value[0]), float(value[1]), float(value[2])]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) and v > 0 for v in values):
        return None
    return values


def resolve_spacing_zyx(
    item: Mapping[str, Any], geometry_item: Mapping[str, Any] | None = None
) -> tuple[list[float], str]:
    """Resolve spacing to (z,y,x), matching the formal eval_3d fallback contract.

    Precedence:
      1. spacing_zyx: already in (z,y,x)
      2. spacing_xyz: reverse to (z,y,x)
      3. legacy spacing: interpreted as (x,y,z), then reversed
    """
    geometry_item = geometry_item or {}
    for source_obj, prefix in ((item, "manifest"), (geometry_item, "geometry")):
        spacing = _coerce_spacing3(source_obj.get("spacing_zyx"))
        if spacing is not None:
            return spacing, f"{prefix}.spacing_zyx"
        spacing = _coerce_spacing3(source_obj.get("spacing_xyz"))
        if spacing is not None:
            return list(reversed(spacing)), f"{prefix}.spacing_xyz"
        spacing = _coerce_spacing3(source_obj.get("spacing"))
        if spacing is not None:
            return list(reversed(spacing)), f"{prefix}.spacing_legacy_xyz"
    raise ContractError("No valid spacing_zyx / spacing_xyz / legacy spacing found")


def normalize_workspace_spacing_metadata(fold_root: Path) -> dict[str, Any]:
    """Write explicit spacing_xyz and spacing_zyx into the copied Idea1 metadata only."""
    manifest_path = fold_root / "meta" / "manifest.json"
    geometry_path = fold_root / "meta" / "geometry_meta.json"
    manifest = load_json(manifest_path)
    geometry = load_json(geometry_path)
    if not isinstance(manifest, list) or not isinstance(geometry, dict):
        raise ContractError(f"Invalid metadata schema under {fold_root}")

    source_counts: dict[str, int] = {}
    normalized = 0
    for item in manifest:
        if not isinstance(item, dict):
            raise ContractError("Manifest record must be an object")
        name = get_slice_name(item)
        geom = geometry.get(name, {})
        if not isinstance(geom, dict):
            geom = {}
        spacing_zyx, source = resolve_spacing_zyx(item, geom)
        spacing_xyz = list(reversed(spacing_zyx))
        item["spacing_xyz"] = spacing_xyz
        item["spacing_zyx"] = spacing_zyx
        if name in geometry and isinstance(geometry[name], dict):
            geometry[name]["spacing_xyz"] = spacing_xyz
            geometry[name]["spacing_zyx"] = spacing_zyx
        source_counts[source] = source_counts.get(source, 0) + 1
        normalized += 1

    atomic_save_json(manifest, manifest_path)
    atomic_save_json(geometry, geometry_path)
    return {"num_records": normalized, "source_counts": source_counts}


class ContractError(RuntimeError):
    """Raised when a frozen-baseline or Idea1 contract is violated."""


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")
        tmp = Path(f.name)
    os.replace(tmp, path)


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as f:
        f.write(text)
        tmp = Path(f.name)
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path, fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root: Path, include_suffixes: set[str] | None = None) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        if include_suffixes is not None and path.suffix not in include_suffixes:
            continue
        result[str(path.relative_to(root))] = sha256_file(path)
    return result


def get_slice_name(item: Mapping[str, Any]) -> str:
    value = item.get("slice_name")
    if value:
        return Path(str(value)).name
    for key in ("teacher_img", "student_img", "native_gt", "teacher_gt", "student_gt"):
        value = item.get(key)
        if value:
            return Path(str(value)).name
    raise ContractError(f"Cannot infer slice_name from manifest keys={sorted(item.keys())}")


def get_case_id(item: Mapping[str, Any], required: bool = False) -> str:
    value = item.get("case_id")
    if value is None or not str(value).strip():
        if required:
            raise ContractError(f"Missing case_id for {get_slice_name(item)}")
        return ""
    return str(value)


def resolve_path(fold_root: Path, item: Mapping[str, Any], key: str) -> Path:
    value = item.get(key)
    if value is None or str(value).strip() == "":
        raise ContractError(f"Manifest item {get_slice_name(item)} misses key={key}")
    path = Path(str(value))
    if not path.is_absolute():
        path = fold_root / path
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def load_manifest(fold_root: Path) -> list[dict[str, Any]]:
    path = fold_root / "meta" / "manifest.json"
    obj = load_json(path)
    if not isinstance(obj, list):
        raise ContractError(f"manifest.json must be a list: {path}")
    result = [dict(x) for x in obj]
    names = [get_slice_name(x) for x in result]
    if len(names) != len(set(names)):
        raise ContractError("manifest contains duplicate slice_name values")
    return result


def manifest_index(manifest: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {get_slice_name(x): dict(x) for x in manifest}


def load_prompts(fold_root: Path, prompt_name: str = "prompts_train.json") -> dict[str, Any]:
    candidates = [fold_root / "prompts" / prompt_name, fold_root / prompt_name]
    for path in candidates:
        if path.exists():
            obj = load_json(path)
            if not isinstance(obj, dict):
                raise ContractError(f"Prompt JSON must be an object: {path}")
            return obj
    raise FileNotFoundError(f"Prompt JSON not found. Tried: {candidates}")


def load_geometry(fold_root: Path) -> dict[str, Any]:
    path = fold_root / "meta" / "geometry_meta.json"
    obj = load_json(path)
    if not isinstance(obj, dict):
        raise ContractError(f"geometry_meta.json must be an object: {path}")
    return obj


def load_split_meta(fold_root: Path) -> dict[str, Any]:
    path = fold_root / "meta" / "split_meta.json"
    if not path.exists():
        return {}
    obj = load_json(path)
    return dict(obj) if isinstance(obj, dict) else {}


def infer_is_3d(dataset: str, split_meta: Mapping[str, Any] | None = None) -> bool:
    if dataset.strip().lower() in KNOWN_3D_DATASETS:
        return True
    split_meta = split_meta or {}
    return any(bool(split_meta.get(k)) for k in ("is_3d", "volume_level_split", "has_volume"))


def train_items(manifest: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(x) for x in manifest if str(x.get("split", "")).lower() == "train"]


def test_items(manifest: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(x) for x in manifest if str(x.get("split", "")).lower() == "test"]


def prompt_instances(meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    if bool(meta.get("empty_slice", False)):
        return []
    instances = meta.get("instances")
    if isinstance(instances, list):
        return [dict(x) for x in instances]
    if isinstance(meta.get("bboxes"), list):
        return [
            {"bbox_teacher": list(box), "bbox": list(box), "label_id": 1, "component_id": idx + 1}
            for idx, box in enumerate(meta["bboxes"])
        ]
    if meta.get("bbox") is not None:
        return [{"bbox_teacher": list(meta["bbox"]), "bbox": list(meta["bbox"]), "label_id": 1, "component_id": 1}]
    return []


def instance_teacher_box(instance: Mapping[str, Any]) -> list[float]:
    for key in ("bbox_teacher", "bbox"):
        if instance.get(key) is not None:
            values = [float(x) for x in instance[key]]
            if len(values) != 4:
                raise ContractError(f"Invalid {key}: {values}")
            return values
    raise ContractError(f"Prompt instance misses bbox_teacher/bbox: {instance}")


def instance_native_box(instance: Mapping[str, Any]) -> list[float] | None:
    value = instance.get("bbox_native")
    if value is None:
        return None
    values = [float(x) for x in value]
    if len(values) != 4:
        raise ContractError(f"Invalid bbox_native: {values}")
    return values


def foreground_slice_names(prompts: Mapping[str, Any], valid_names: set[str]) -> list[str]:
    result = []
    for name in sorted(valid_names):
        meta = prompts.get(name)
        if isinstance(meta, Mapping) and len(prompt_instances(meta)) > 0:
            result.append(name)
    return result


def build_case_to_slices(items: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for item in items:
        result[get_case_id(item, required=True)].append(get_slice_name(item))
    for case_id in result:
        result[case_id] = sorted(result[case_id])
    return dict(sorted(result.items()))


def compute_3d_budget(
    num_train_cases: int,
    ratio: float = 0.05,
) -> int:
    """Return the integer case count nearest to the target ratio."""
    if num_train_cases <= 0:
        raise ValueError(
            "num_train_cases must be positive"
        )

    raw = (
        float(num_train_cases)
        * float(ratio)
    )

    return max(
        1,
        int(math.floor(raw + 0.5)),
    )


def seeded_sample(values: Sequence[str], count: int, seed: int) -> list[str]:
    ordered = sorted(str(x) for x in values)
    if count < 0 or count > len(ordered):
        raise ValueError(f"count={count} outside [0,{len(ordered)}]")
    rng = random.Random(int(seed))
    rng.shuffle(ordered)
    return sorted(ordered[:count])


def validate_partition(train_names: set[str], full_names: set[str], box_names: set[str]) -> None:
    overlap = full_names & box_names
    if overlap:
        raise ContractError(f"Full/Box overlap: {sorted(overlap)[:10]}")
    missing = train_names - (full_names | box_names)
    extra = (full_names | box_names) - train_names
    if missing or extra:
        raise ContractError(
            f"Full+Box partition mismatch: missing={sorted(missing)[:10]}, extra={sorted(extra)[:10]}"
        )


def load_selection(path: Path) -> dict[str, Any]:
    obj = load_json(path)
    if not isinstance(obj, dict):
        raise ContractError(f"Selection must be a JSON object: {path}")
    return dict(obj)


def full_and_box_from_selection(
    selection: Mapping[str, Any], manifest: Sequence[Mapping[str, Any]], is_3d: bool
) -> tuple[set[str], set[str], set[str]]:
    train = train_items(manifest)
    train_names = {get_slice_name(x) for x in train}
    if is_3d:
        case_to_slices = build_case_to_slices(train)
        full_cases = {str(x) for x in selection.get("cumulative_full_case_ids", [])}
        unknown = full_cases - set(case_to_slices)
        if unknown:
            raise ContractError(f"Unknown Full case IDs: {sorted(unknown)}")
        full = {name for case_id in full_cases for name in case_to_slices[case_id]}
    else:
        full_cases = set()
        full = {str(x) for x in selection.get("cumulative_full_slice_names", [])}
    box = train_names - full
    validate_partition(train_names, full, box)
    return full, box, full_cases


def copy_or_symlink(src: Path, dst: Path, mode: str, overwrite: bool = False) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            raise FileExistsError(dst)
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
    elif mode == "copy":
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported mode={mode}; expected symlink/copy")


def load_npy_2d(path: Path) -> np.ndarray:
    array = np.load(path)
    array = np.asarray(array)
    while array.ndim > 2 and 1 in array.shape:
        array = np.squeeze(array)
    if array.ndim == 3 and array.shape[-1] in (1, 3):
        array = array[..., 0]
    if array.ndim != 2:
        raise ContractError(f"Expected 2D array, got {array.shape}: {path}")
    return array


def load_rgb_uint8(path: Path) -> np.ndarray:
    image = np.load(path)
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    elif image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ContractError(f"Expected HWC RGB image, got {image.shape}: {path}")
    image = image.astype(np.float32)
    if image.max(initial=0.0) <= 1.5:
        image *= 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def box_mask(box: Sequence[float], height: int, width: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(v) for v in box]
    xa = max(0, min(width - 1, int(math.floor(x1))))
    ya = max(0, min(height - 1, int(math.floor(y1))))
    xb = max(0, min(width, int(math.ceil(x2)) + 1))
    yb = max(0, min(height, int(math.ceil(y2)) + 1))
    out = np.zeros((height, width), dtype=bool)
    if xb > xa and yb > ya:
        out[ya:yb, xa:xb] = True
    return out


def binary_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(pred, gt).sum() / union)


def largest_component(mask: np.ndarray, connectivity: int = 8) -> np.ndarray:
    import cv2

    mask_u8 = np.asarray(mask, dtype=np.uint8)
    if mask_u8.max(initial=0) == 0:
        return mask_u8.astype(bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=connectivity)
    if n <= 1:
        return mask_u8.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = int(np.argmax(areas)) + 1
    return labels == keep


def geometry_item(geometry: Mapping[str, Any], slice_name: str) -> dict[str, Any]:
    item = geometry.get(slice_name)
    if item is None:
        item = geometry.get(Path(slice_name).stem)
    if not isinstance(item, Mapping):
        raise ContractError(f"geometry_meta misses {slice_name}")
    return dict(item)


def _transform_spec(geom: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = geom.get(key)
    if not isinstance(value, Mapping):
        raise ContractError(f"geometry item misses transform={key}")
    return dict(value)


def map_native_mask_to_target(mask: np.ndarray, geom: Mapping[str, Any], transform_key: str) -> np.ndarray:
    import cv2

    spec = _transform_spec(geom, transform_key)
    target_h = int(spec.get("target_h", 0))
    target_w = int(spec.get("target_w", 0))
    if target_h <= 0 or target_w <= 0:
        fallback_key = "teacher_hw" if "teacher" in transform_key else "student_hw"
        fallback = geom.get(fallback_key, [])
        if isinstance(fallback, Sequence) and len(fallback) >= 2:
            target_h, target_w = int(fallback[0]), int(fallback[1])
    if target_h <= 0 or target_w <= 0:
        raise ContractError(f"Invalid target size in {transform_key}: {spec}")

    scale_x = float(spec.get("scale_x", spec.get("scale", 1.0)))
    scale_y = float(spec.get("scale_y", spec.get("scale", 1.0)))
    offset_x = int(round(float(spec.get("offset_x", 0.0))))
    offset_y = int(round(float(spec.get("offset_y", 0.0))))
    resized_w = int(spec.get("resized_w", round(mask.shape[1] * scale_x)))
    resized_h = int(spec.get("resized_h", round(mask.shape[0] * scale_y)))
    resized_w = max(1, resized_w)
    resized_h = max(1, resized_h)
    resized = cv2.resize(mask.astype(np.uint8), (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
    output = np.zeros((target_h, target_w), dtype=np.uint8)
    x1, y1 = max(0, offset_x), max(0, offset_y)
    x2, y2 = min(target_w, offset_x + resized_w), min(target_h, offset_y + resized_h)
    if x2 <= x1 or y2 <= y1:
        raise ContractError(f"Transform places mask outside target: {spec}")
    sx1, sy1 = x1 - offset_x, y1 - offset_y
    output[y1:y2, x1:x2] = resized[sy1 : sy1 + (y2 - y1), sx1 : sx1 + (x2 - x1)]
    return output


def map_teacher_mask_to_native(mask: np.ndarray, geom: Mapping[str, Any]) -> np.ndarray:
    import cv2

    spec = _transform_spec(geom, "native_to_teacher")
    orig_h = int(spec.get("orig_h", geom.get("orig_h", 0)))
    orig_w = int(spec.get("orig_w", geom.get("orig_w", 0)))
    if orig_h <= 0 or orig_w <= 0:
        native_hw = geom.get("native_hw", [])
        if isinstance(native_hw, Sequence) and len(native_hw) >= 2:
            orig_h, orig_w = int(native_hw[0]), int(native_hw[1])
    if orig_h <= 0 or orig_w <= 0:
        raise ContractError(f"Cannot infer native size: {geom}")

    scale_x = float(spec.get("scale_x", spec.get("scale", 1.0)))
    scale_y = float(spec.get("scale_y", spec.get("scale", 1.0)))
    offset_x = int(round(float(spec.get("offset_x", 0.0))))
    offset_y = int(round(float(spec.get("offset_y", 0.0))))
    resized_w = int(spec.get("resized_w", round(orig_w * scale_x)))
    resized_h = int(spec.get("resized_h", round(orig_h * scale_y)))
    x1, y1 = max(0, offset_x), max(0, offset_y)
    x2, y2 = min(mask.shape[1], offset_x + resized_w), min(mask.shape[0], offset_y + resized_h)
    cropped = np.asarray(mask)[y1:y2, x1:x2]
    if cropped.size == 0:
        raise ContractError(f"Empty teacher crop from geometry: {spec}")
    return cv2.resize(cropped.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)


def map_native_box_to_target(box: Sequence[float], geom: Mapping[str, Any], transform_key: str) -> list[float]:
    spec = _transform_spec(geom, transform_key)
    sx = float(spec.get("scale_x", spec.get("scale", 1.0)))
    sy = float(spec.get("scale_y", spec.get("scale", 1.0)))
    ox = float(spec.get("offset_x", 0.0))
    oy = float(spec.get("offset_y", 0.0))
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1 * sx + ox, y1 * sy + oy, x2 * sx + ox, y2 * sy + oy]


def component_mask_native(
    gt_native: np.ndarray,
    label_id: int,
    component_id: int | None,
    native_box: Sequence[float] | None,
    connectivity: int = 8,
) -> np.ndarray:
    import cv2

    label_mask = (gt_native.astype(np.int64) == int(label_id)).astype(np.uint8)
    if label_mask.max(initial=0) == 0:
        raise ContractError(f"GT has no label_id={label_id}")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(label_mask, connectivity=connectivity)
    candidate_ids = list(range(1, n))
    if component_id is not None and component_id in candidate_ids:
        return labels == int(component_id)
    if native_box is not None:
        target_box = [float(x) for x in native_box]
        best_id = None
        best_iou = -1.0
        target_bm = box_mask(target_box, gt_native.shape[0], gt_native.shape[1])
        for cc_id in candidate_ids:
            current = labels == cc_id
            ys, xs = np.where(current)
            if len(xs) == 0:
                continue
            current_box = [xs.min(), ys.min(), xs.max(), ys.max()]
            score = binary_iou(box_mask(current_box, *gt_native.shape), target_bm)
            if np.isfinite(score) and score > best_iou:
                best_iou = score
                best_id = cc_id
        if best_id is not None:
            return labels == int(best_id)
    if len(candidate_ids) == 1:
        return labels == candidate_ids[0]
    raise ContractError(
        f"Cannot uniquely match label={label_id}, component_id={component_id}, box={native_box}; components={candidate_ids}"
    )


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def current_timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class RoundPaths:
    fold_root: Path
    method: str
    round_index: int

    @property
    def root(self) -> Path:
        return self.fold_root / "rounds" / self.method / f"round_{self.round_index:02d}"

    @property
    def selection_dir(self) -> Path:
        return self.root / "selection"

    @property
    def pair_dir(self) -> Path:
        return self.root / "finetune_pairs"

    @property
    def teacher_dir(self) -> Path:
        return self.root / "teacher"

    @property
    def diagnosis_dir(self) -> Path:
        return self.root / "diagnosis"

    @property
    def state_path(self) -> Path:
        return self.root / "round_state.json"


def selection_path(fold_root: Path, method: str, round_index: int) -> Path:
    return RoundPaths(fold_root, method, round_index).selection_dir / "selection.json"


def ensure_method_not_baseline(method: str) -> None:
    if method in {"baseline", "upper", "boxonly", "frozen_baseline_v1"}:
        raise ContractError(f"Unsafe method name for writable Idea1 outputs: {method}")


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_output_isolated(output: Path, frozen_roots: Iterable[Path]) -> None:
    for root in frozen_roots:
        if is_relative_to(output, root):
            raise ContractError(f"Idea1 output must not be inside frozen root: output={output}, frozen={root}")
