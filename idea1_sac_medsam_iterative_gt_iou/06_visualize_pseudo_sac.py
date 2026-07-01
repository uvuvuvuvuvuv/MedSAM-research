#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAC-MedSAM pseudo-label visualization and diagnosis for the cleaned mainline.

Current pipeline assumptions:
1) split metadata: meta/full_box_split_<method>.json;
2) prompts: prompts/prompts_train.json (teacher-space boxes by default);
3) geometry: meta/geometry_meta.json;
4) frozen baseline pseudo labels: pseudo_student/<baseline_pseudo_name>;
5) SAC pseudo labels: pseudo_student/<sac_pseudo_name>;
6) evaluation and ranking are performed in student space against student_gt;
7) figures may be rendered in student space or reconstructed native space.

The script supports both full runs and partial smoke runs. Full-supervision samples are
shown only as references by default; aggregate method comparison is box-only so copied
full GT masks do not inflate SAC quality.

Label convention:
    0      background
    1..K   foreground class IDs
    255    unknown / ignore
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


# Color-blind-aware fixed palette. Class ID -> RGB.
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    1: (230, 159, 0),
    2: (86, 180, 233),
    3: (0, 158, 115),
    4: (240, 228, 66),
    5: (0, 114, 178),
    6: (213, 94, 0),
    7: (204, 121, 167),
    8: (127, 201, 127),
    9: (190, 174, 212),
    10: (253, 192, 134),
    11: (255, 255, 153),
    12: (56, 108, 176),
    13: (240, 2, 127),
    14: (191, 91, 23),
    15: (102, 102, 102),
}

ERROR_COLORS: Dict[str, Tuple[int, int, int]] = {
    "tp": (0, 180, 0),          # correct foreground
    "fp": (220, 40, 40),       # foreground on GT background
    "fn": (40, 100, 230),      # missed foreground
    "confusion": (220, 0, 180),# wrong foreground class
    "unknown": (170, 170, 170),
}

KNOWN_3D_DATASETS: Set[str] = {"acdc", "prostate158", "btcv", "synapse"}


FALLBACK_LABEL_NAMES: Dict[str, Dict[int, str]] = {
    "acdc": {1: "RV", 2: "MYO", 3: "LV"},
    "prostate158": {1: "Central gland", 2: "Peripheral zone"},
    "synapse": {
        1: "Aorta", 2: "Gallbladder", 3: "Left kidney", 4: "Right kidney",
        5: "Liver", 6: "Pancreas", 7: "Spleen", 8: "Stomach",
    },
    "btcv": {
        1: "Spleen", 2: "Right kidney", 3: "Left kidney", 4: "Gallbladder",
        5: "Esophagus", 6: "Liver", 7: "Stomach", 8: "Aorta",
        9: "IVC", 10: "Portal/splenic veins", 11: "Pancreas",
        12: "Right adrenal", 13: "Left adrenal",
    },
}


@dataclass
class SampleRecord:
    slice_name: str
    split: str
    case_id: str
    slice_idx: int
    image_path: Path
    native_image_path: Optional[Path]
    gt_path: Path
    baseline_path: Path
    sac_path: Path
    label_mode: str
    prompt_meta: Optional[dict]
    geometry_meta: Optional[dict]


@dataclass
class RenderedSample:
    record: SampleRecord
    image_rgb: np.ndarray
    gt: np.ndarray
    baseline: np.ndarray
    sac: np.ndarray
    boxes_student: List[Tuple[np.ndarray, int]]
    class_names: Dict[int, str]
    metrics: Dict[str, Any]
    display_mode: str
    image_source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize frozen baseline and SAC-MedSAM tri-state pseudo labels "
            "for the cleaned full/box training pipeline."
        )
    )
    parser.add_argument(
        "--processed-root", "--processed_root",
        dest="processed_root",
        type=Path,
        default=Path("/storage/baiyuting/data/out_data_idea1/MedSAM-main/data/processed"),
        help="Root containing <dataset>/<fold>.",
    )
    parser.add_argument(
        "--baseline-processed-root", "--baseline_processed_root",
        dest="baseline_processed_root",
        type=Path,
        default=None,
        help="Optional separate root containing the frozen baseline pseudo labels.",
    )
    parser.add_argument(
        "--output-root", "--output_root",
        dest="output_root",
        type=Path,
        default=Path(
            "/storage/baiyuting/data/out_data_idea1/visualization/"
            "pseudo_quality/idea1_sac_medsam_final"
        ),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=(
            "kvasirseg,cvc_clinicdb,tn3k,tg3k,ddti,otu_2d,ph2,"
            "acdc,prostate158,btcv,synapse"
        ),
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--fold", type=str, default="fold_0")
    parser.add_argument("--method", type=str, default="idea1_sac_medsam_final")
    parser.add_argument(
        "--baseline-pseudo-name", "--baseline_pseudo_name",
        dest="baseline_pseudo_name",
        type=str,
        default="tri_train",
        help="Directory name under pseudo_student for the frozen baseline.",
    )
    parser.add_argument(
        "--sac-pseudo-name", "--sac_pseudo_name",
        dest="sac_pseudo_name",
        type=str,
        default=None,
        help=(
            "Directory name under pseudo_student for SAC. "
            "Defaults to tri_train_<method>."
        ),
    )
    parser.add_argument(
        "--per-group", "--per_group",
        dest="per_group",
        type=int,
        default=5,
        help="Number of selected samples per diagnostic group.",
    )
    parser.add_argument(
        "--max-per-case", "--max_per_case",
        dest="max_per_case",
        type=int,
        default=3,
        help="Maximum selected slices from one 3D case across all groups.",
    )
    parser.add_argument(
        "--prompt-space", "--prompt_space",
        dest="prompt_space",
        choices=["teacher", "native", "student"],
        default="teacher",
        help="Coordinate space of stored prompt boxes. Current pipeline uses teacher.",
    )
    parser.add_argument(
        "--display-mode", "--display_mode",
        dest="display_mode",
        choices=["student", "native", "native_full"],
        default="native",
        help=(
            "student: padded 352x352 training canvas; "
            "native/native_full: remove student padding and restore native resolution."
        ),
    )
    parser.add_argument(
        "--auto-grayscale", "--auto_grayscale",
        dest="auto_grayscale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collapse near-grayscale RGB medical images for display only.",
    )
    parser.add_argument(
        "--grayscale-datasets", "--grayscale_datasets",
        dest="grayscale_datasets",
        type=str,
        default="",
        help="Comma-separated datasets forced to grayscale for visualization only.",
    )
    parser.add_argument("--overlay-alpha", "--overlay_alpha", dest="overlay_alpha", type=float, default=0.58)
    parser.add_argument("--error-alpha", "--error_alpha", dest="error_alpha", type=float, default=0.62)
    parser.add_argument("--dpi", type=int, default=220)
    parser.add_argument("--sheet-dpi", "--sheet_dpi", dest="sheet_dpi", type=int, default=180)
    parser.add_argument(
        "--include-full", "--include_full",
        dest="include_full",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render a full-supervision reference group.",
    )
    parser.add_argument(
        "--render-individual", "--render_individual",
        dest="render_individual",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--render-contact-sheets", "--render_contact_sheets",
        dest="render_contact_sheets",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--strict",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Fail on malformed paths, labels, shapes, prompts, or geometry. "
            "Use --allow-partial for smoke runs instead of disabling strict geometry."
        ),
    )
    parser.add_argument(
        "--allow-partial", "--allow_partial",
        dest="allow_partial",
        action="store_true",
        help=(
            "Skip train samples whose baseline/SAC pseudo file is absent. "
            "Intended for max_samples smoke outputs."
        ),
    )
    parser.add_argument(
        "--only-box", "--only_box",
        dest="only_box",
        action="store_true",
        help="Analyze only box-labeled samples and omit full references.",
    )
    parser.add_argument(
        "--summary-scope", "--summary_scope",
        dest="summary_scope",
        choices=["box", "all"],
        default="box",
        help=(
            "Rows used for aggregate comparison. box is recommended because SAC full "
            "samples are copied GT masks."
        ),
    )
    parser.add_argument(
        "--max-samples-scan", "--max_samples_scan", "--max-valid-samples",
        dest="max_samples_scan",
        type=int,
        default=-1,
        help="Maximum number of valid records to inspect; -1 scans all available records.",
    )
    parser.add_argument(
        "--max-warning-examples", "--max_warning_examples",
        dest="max_warning_examples",
        type=int,
        default=40,
        help="Maximum detailed missing-file warnings written to warnings.log.",
    )
    parser.add_argument(
        "--validate-labels", "--validate_labels",
        dest="validate_labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require GT/baseline/SAC values to be in {0, class IDs, 255}.",
    )
    args = parser.parse_args()
    if args.display_mode == "native_full":
        args.display_mode = "native"
    if args.sac_pseudo_name is None:
        args.sac_pseudo_name = f"tri_train_{args.method}"
    if args.only_box:
        args.include_full = False
        args.summary_scope = "box"
    if args.max_samples_scan == 0:
        args.max_samples_scan = -1
    return args


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_path(fold_root: Path, value: Any) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else fold_root / p


def normalize_manifest(raw: Any) -> List[dict]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if "samples" in raw and isinstance(raw["samples"], list):
            return raw["samples"]
        out = []
        for key, value in raw.items():
            if isinstance(value, dict):
                rec = dict(value)
                rec.setdefault("slice_name", key)
                out.append(rec)
        return out
    raise TypeError(f"Unsupported manifest type: {type(raw).__name__}")


def normalize_named_records(raw: Any, key_name: str = "slice_name") -> Dict[str, dict]:
    if raw is None:
        return {}
    if isinstance(raw, list):
        out: Dict[str, dict] = {}
        for rec in raw:
            if not isinstance(rec, dict):
                continue
            name = rec.get(key_name) or rec.get("filename") or rec.get("name")
            if name:
                out[str(name)] = rec
        return out
    if isinstance(raw, dict):
        if "samples" in raw and isinstance(raw["samples"], list):
            return normalize_named_records(raw["samples"], key_name)
        out = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                rec = dict(value)
                rec.setdefault(key_name, key)
                out[str(key)] = rec
        return out
    return {}


def find_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def find_prompt_file(fold_root: Path, method: str) -> Optional[Path]:
    # 04_generate_pseudo_sac.py consumes prompts/prompts_train.json.
    return find_existing([
        fold_root / "prompts" / "prompts_train.json",
        fold_root / "prompts" / f"prompts_train_{method}.json",
        fold_root / "prompts_train.json",
        fold_root / f"prompts_train_{method}.json",
    ])


def find_full_box_file(fold_root: Path, method: str) -> Optional[Path]:
    return find_existing([
        fold_root / "meta" / f"full_box_split_{method}.json",
        fold_root / "meta" / "full_box_split.json",
    ])


def parse_label_names(dataset: str, label_meta_path: Path, observed_classes: Set[int]) -> Dict[int, str]:
    names: Dict[int, str] = {}
    if label_meta_path.exists():
        try:
            raw = load_json(label_meta_path)
            candidates = []
            if isinstance(raw, dict):
                for key in ["classes", "labels", "label_names", "class_names", "id_to_name"]:
                    if key in raw:
                        candidates.append(raw[key])
                candidates.append(raw)
            else:
                candidates.append(raw)

            for obj in candidates:
                if isinstance(obj, dict):
                    for k, v in obj.items():
                        try:
                            cid = int(k)
                        except (TypeError, ValueError):
                            continue
                        if isinstance(v, dict):
                            name = v.get("name") or v.get("label_name") or v.get("class_name")
                        else:
                            name = str(v)
                        if name:
                            names[cid] = str(name)
                elif isinstance(obj, list):
                    for idx, v in enumerate(obj):
                        if isinstance(v, dict):
                            cid = int(v.get("id", v.get("label_id", idx)))
                            name = v.get("name") or v.get("label_name") or v.get("class_name")
                        else:
                            cid = idx
                            name = str(v)
                        if cid > 0 and name:
                            names[cid] = str(name)
        except Exception:
            pass

    for cid, name in FALLBACK_LABEL_NAMES.get(dataset.lower(), {}).items():
        names.setdefault(cid, name)
    for cid in observed_classes:
        names.setdefault(cid, f"Class {cid}")
    return names


def load_array_2d(path: Path, role: str) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr)
    arr = np.squeeze(arr)
    if role == "image":
        return arr
    if arr.ndim != 2:
        raise ValueError(f"{role} must be 2D after squeeze, got {arr.shape}: {path}")
    return arr.astype(np.uint8, copy=False)


def _canonicalize_image_array(arr: np.ndarray) -> np.ndarray:
    """Convert an image array to HxW or HxWxC without reshaping spatial data.

    Ambiguous multi-frame arrays are reduced to one frame/channel instead of
    being flattened or tiled. This avoids accidental side-by-side duplication.
    """
    x = np.asarray(arr)
    x = np.squeeze(x)

    # Remove batch-like dimensions conservatively.
    while x.ndim > 3:
        x = x[0]
        x = np.squeeze(x)

    if x.ndim == 2:
        return x
    if x.ndim != 3:
        raise ValueError(f"Unsupported image shape after squeeze: {x.shape}")

    # Unambiguous HWC / CHW.
    if x.shape[-1] in (1, 3, 4):
        return x[..., :3] if x.shape[-1] > 1 else x[..., 0]
    if x.shape[0] in (1, 3, 4):
        y = np.transpose(x[:3], (1, 2, 0))
        return y[..., 0] if y.shape[-1] == 1 else y

    # Two-channel arrays are commonly frames/modalities, not RGB.
    if x.shape[0] == 2 and x.shape[1] > 8 and x.shape[2] > 8:
        return x[0]
    if x.shape[-1] == 2 and x.shape[0] > 8 and x.shape[1] > 8:
        return x[..., 0]

    # N,H,W or H,W,N: choose one frame, never reshape/tile.
    smallest_axis = int(np.argmin(x.shape))
    if smallest_axis == 0:
        return x[0]
    if smallest_axis == 2:
        return x[..., 0]

    # Last conservative fallback.
    return x[0]


def _near_grayscale_rgb(rgb: np.ndarray) -> bool:
    x = np.asarray(rgb, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return True
    vals = x[finite]
    vmin, vmax = float(vals.min()), float(vals.max())
    scale = 1.0 if vmax <= 1.5 and vmin >= -0.1 else 255.0
    channel_range = np.nanmax(x, axis=-1) - np.nanmin(x, axis=-1)
    threshold = 10.0 / 255.0 * scale
    color_fraction = float(np.mean(channel_range > threshold))
    mean_diff = float(np.nanmean(channel_range))
    return color_fraction < 0.01 and mean_diff < (5.0 / 255.0 * scale)


def normalize_image_to_rgb(
    arr: np.ndarray,
    force_grayscale: bool = False,
    auto_grayscale: bool = True,
) -> np.ndarray:
    """Safely normalize an image for display.

    Key safeguards:
    - never reshapes or tiles ambiguous multi-frame data;
    - near-grayscale RGB is normalized once as grayscale;
    - true RGB uses one shared intensity transform, not independent per-channel
      stretching that can create artificial colored speckles.
    """
    x = _canonicalize_image_array(arr)

    if x.ndim == 2:
        gray = robust_uint8(x)
        return np.repeat(gray[..., None], 3, axis=-1)

    if x.ndim != 3 or x.shape[-1] < 3:
        raise ValueError(f"Unsupported canonical image shape: {x.shape}")

    rgb = np.asarray(x[..., :3])
    if force_grayscale or (auto_grayscale and _near_grayscale_rgb(rgb)):
        gray = robust_uint8(np.nanmean(rgb.astype(np.float32), axis=-1))
        return np.repeat(gray[..., None], 3, axis=-1)

    if rgb.dtype == np.uint8:
        return rgb.copy()

    xf = rgb.astype(np.float32)
    finite = np.isfinite(xf)
    if not finite.any():
        return np.zeros(xf.shape, dtype=np.uint8)
    vals = xf[finite]
    lo_raw, hi_raw = float(vals.min()), float(vals.max())
    if hi_raw <= 1.5 and lo_raw >= -0.1:
        out = np.clip(xf, 0.0, 1.0) * 255.0
        out[~finite] = 0
        return out.astype(np.uint8)

    # Shared robust normalization preserves genuine channel relationships.
    lo, hi = np.percentile(vals, [1.0, 99.0])
    if hi <= lo + 1e-6:
        lo, hi = lo_raw, hi_raw
    if hi <= lo + 1e-6:
        return np.zeros(xf.shape, dtype=np.uint8)
    out = (xf - lo) / (hi - lo)
    out[~finite] = 0
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def robust_uint8(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros(x.shape, dtype=np.uint8)
    vals = x[finite]
    lo, hi = np.percentile(vals, [1.0, 99.0])
    if hi <= lo + 1e-6:
        lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo + 1e-6:
        return np.zeros(x.shape, dtype=np.uint8)
    y = (x - lo) / (hi - lo)
    y[~finite] = 0
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def _geometry_transforms(
    geom: Optional[dict],
    student_hw: Tuple[int, int],
) -> Tuple[dict, dict]:
    sh, sw = student_hw
    g = geom or {}
    teacher = dict(g.get("native_to_teacher") or g.get("teacher") or {})
    student = dict(g.get("native_to_student") or g.get("student") or {})

    teacher.setdefault("orig_h", g.get("orig_h", sh))
    teacher.setdefault("orig_w", g.get("orig_w", sw))
    teacher.setdefault("target_h", (g.get("teacher_hw") or [1024, 1024])[0])
    teacher.setdefault("target_w", (g.get("teacher_hw") or [1024, 1024])[1])
    teacher.setdefault("new_h", g.get("new_h", teacher["target_h"]))
    teacher.setdefault("new_w", g.get("new_w", teacher["target_w"]))
    teacher.setdefault("offset_x", 0)
    teacher.setdefault("offset_y", 0)

    student.setdefault("orig_h", g.get("orig_h", sh))
    student.setdefault("orig_w", g.get("orig_w", sw))
    student.setdefault("target_h", (g.get("student_hw") or [sh, sw])[0])
    student.setdefault("target_w", (g.get("student_hw") or [sh, sw])[1])
    student.setdefault("new_h", sh)
    student.setdefault("new_w", sw)
    student.setdefault("offset_x", 0)
    student.setdefault("offset_y", 0)
    return teacher, student


def _student_valid_crop(
    arr: np.ndarray,
    geom: Optional[dict],
) -> np.ndarray:
    h, w = arr.shape[:2]
    _, student = _geometry_transforms(geom, (h, w))
    x0 = int(round(float(student.get("offset_x", 0))))
    y0 = int(round(float(student.get("offset_y", 0))))
    nw = int(round(float(student.get("new_w", w))))
    nh = int(round(float(student.get("new_h", h))))
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x0 + max(nw, 1)))
    y1 = max(y0 + 1, min(h, y0 + max(nh, 1)))
    return arr[y0:y1, x0:x1]


def student_image_to_native(
    image_rgb: np.ndarray,
    geom: Optional[dict],
) -> np.ndarray:
    crop = _student_valid_crop(image_rgb, geom)
    _, student = _geometry_transforms(geom, image_rgb.shape[:2])
    oh = int(round(float(student.get("orig_h", crop.shape[0]))))
    ow = int(round(float(student.get("orig_w", crop.shape[1]))))
    return cv2.resize(crop, (max(ow, 1), max(oh, 1)), interpolation=cv2.INTER_LINEAR)


def student_mask_to_native(
    mask: np.ndarray,
    geom: Optional[dict],
) -> np.ndarray:
    crop = _student_valid_crop(mask, geom)
    _, student = _geometry_transforms(geom, mask.shape[:2])
    oh = int(round(float(student.get("orig_h", crop.shape[0]))))
    ow = int(round(float(student.get("orig_w", crop.shape[1]))))
    return cv2.resize(crop, (max(ow, 1), max(oh, 1)), interpolation=cv2.INTER_NEAREST).astype(np.uint8)


def teacher_box_to_native(
    box: np.ndarray,
    geom: dict,
    student_hw: Tuple[int, int],
) -> np.ndarray:
    teacher, student = _geometry_transforms(geom, student_hw)
    tw = float(teacher["new_w"])
    th = float(teacher["new_h"])
    tx = float(teacher.get("offset_x", 0))
    ty = float(teacher.get("offset_y", 0))
    ow = float(student["orig_w"])
    oh = float(student["orig_h"])
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    return np.array([
        (x1 - tx) * ow / max(tw, 1.0),
        (y1 - ty) * oh / max(th, 1.0),
        (x2 - tx) * ow / max(tw, 1.0),
        (y2 - ty) * oh / max(th, 1.0),
    ], dtype=np.float32)


def student_box_to_native(
    box: np.ndarray,
    geom: dict,
    student_hw: Tuple[int, int],
) -> np.ndarray:
    _, student = _geometry_transforms(geom, student_hw)
    sw = float(student["new_w"])
    sh = float(student["new_h"])
    sx = float(student.get("offset_x", 0))
    sy = float(student.get("offset_y", 0))
    ow = float(student["orig_w"])
    oh = float(student["orig_h"])
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    return np.array([
        (x1 - sx) * ow / max(sw, 1.0),
        (y1 - sy) * oh / max(sh, 1.0),
        (x2 - sx) * ow / max(sw, 1.0),
        (y2 - sy) * oh / max(sh, 1.0),
    ], dtype=np.float32)


def map_boxes_to_native(
    instances: Sequence[Tuple[np.ndarray, int]],
    geom: Optional[dict],
    student_hw: Tuple[int, int],
    prompt_space: str,
) -> List[Tuple[np.ndarray, int]]:
    if geom is None:
        raise ValueError("geometry_meta is required for native display")
    _, student = _geometry_transforms(geom, student_hw)
    oh, ow = int(student["orig_h"]), int(student["orig_w"])
    out: List[Tuple[np.ndarray, int]] = []
    for box, cid in instances:
        if prompt_space == "teacher":
            mapped = teacher_box_to_native(box, geom, student_hw)
        elif prompt_space == "student":
            mapped = student_box_to_native(box, geom, student_hw)
        else:
            mapped = box.astype(np.float32).copy()
        mapped[0::2] = np.clip(mapped[0::2], 0, max(ow - 1, 0))
        mapped[1::2] = np.clip(mapped[1::2], 0, max(oh - 1, 0))
        out.append((mapped, int(cid)))
    return out


def _focus_crop_box(
    gt: np.ndarray,
    boxes: Sequence[Tuple[np.ndarray, int]],
    margin: float,
) -> Tuple[int, int, int, int]:
    h, w = gt.shape
    xs: List[float] = []
    ys: List[float] = []
    gy, gx = np.where(gt > 0)
    if len(gx):
        xs.extend([float(gx.min()), float(gx.max())])
        ys.extend([float(gy.min()), float(gy.max())])
    for box, _ in boxes:
        xs.extend([float(box[0]), float(box[2])])
        ys.extend([float(box[1]), float(box[3])])
    if not xs or not ys:
        return 0, 0, w, h
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    dx = max(x2 - x1 + 1, 1.0)
    dy = max(y2 - y1 + 1, 1.0)
    x1 = int(math.floor(x1 - margin * dx))
    x2 = int(math.ceil(x2 + margin * dx)) + 1
    y1 = int(math.floor(y1 - margin * dy))
    y2 = int(math.ceil(y2 + margin * dy)) + 1
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def crop_display_arrays(
    image_rgb: np.ndarray,
    gt: np.ndarray,
    baseline: np.ndarray,
    sac: np.ndarray,
    boxes: Sequence[Tuple[np.ndarray, int]],
    margin: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Tuple[np.ndarray, int]]]:
    x1, y1, x2, y2 = _focus_crop_box(gt, boxes, margin)
    image2 = image_rgb[y1:y2, x1:x2]
    gt2 = gt[y1:y2, x1:x2]
    baseline2 = baseline[y1:y2, x1:x2]
    sac2 = sac[y1:y2, x1:x2]
    boxes2 = []
    for box, cid in boxes:
        mapped = box.astype(np.float32).copy()
        mapped[0::2] -= x1
        mapped[1::2] -= y1
        mapped[0::2] = np.clip(mapped[0::2], 0, max(image2.shape[1] - 1, 0))
        mapped[1::2] = np.clip(mapped[1::2], 0, max(image2.shape[0] - 1, 0))
        boxes2.append((mapped, cid))
    return image2, gt2, baseline2, sac2, boxes2


def validate_same_shape(gt: np.ndarray, baseline: np.ndarray, sac: np.ndarray, strict: bool) -> Tuple[np.ndarray, np.ndarray]:
    if baseline.shape == gt.shape and sac.shape == gt.shape:
        return baseline, sac
    if strict:
        raise ValueError(f"Shape mismatch: gt={gt.shape}, baseline={baseline.shape}, sac={sac.shape}")
    baseline2 = cv2.resize(baseline, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    sac2 = cv2.resize(sac, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST)
    return baseline2.astype(np.uint8), sac2.astype(np.uint8)


def get_class_ids(*maps: np.ndarray) -> List[int]:
    ids: Set[int] = set()
    for arr in maps:
        vals = np.unique(arr)
        ids.update(int(v) for v in vals if int(v) not in (0, 255))
    return sorted(ids)


def components_stats(pred: np.ndarray, class_ids: Sequence[int]) -> Tuple[int, float, float]:
    total_components = 0
    total_area = 0
    largest_area_sum = 0
    tiny_area = 0
    image_area = int(pred.size)
    tiny_threshold = max(8, int(round(image_area * 0.0005)))

    for cid in class_ids:
        mask = (pred == cid).astype(np.uint8)
        area = int(mask.sum())
        if area == 0:
            continue
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        comp_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)]
        if not comp_areas:
            continue
        total_components += len(comp_areas)
        total_area += sum(comp_areas)
        largest_area_sum += max(comp_areas)
        tiny_area += sum(a for a in comp_areas if a <= tiny_threshold)

    largest_ratio = largest_area_sum / max(total_area, 1)
    tiny_ratio = tiny_area / max(total_area, 1)
    return total_components, float(largest_ratio), float(tiny_ratio)


def boundary_far_fp_ratio(pred: np.ndarray, gt: np.ndarray, radius: int = 3) -> float:
    pred_fg = (pred > 0) & (pred != 255)
    gt_fg = gt > 0
    fp = pred_fg & (~gt_fg)
    if not fp.any():
        return 0.0
    k = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
    near_gt = cv2.dilate(gt_fg.astype(np.uint8), k, iterations=1) > 0
    far_fp = fp & (~near_gt)
    return float(far_fp.sum() / max(pred_fg.sum(), 1))


def quality_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    unknown = pred == 255
    pred_background = pred == 0
    pred_fg = (pred > 0) & (~unknown)
    gt_fg = gt > 0

    class_ids = get_class_ids(pred, gt)
    dice_list: List[float] = []
    iou_list: List[float] = []
    per_class: Dict[str, Dict[str, float]] = {}

    for cid in class_ids:
        p = pred == cid
        g = gt == cid
        inter = int((p & g).sum())
        p_sum = int(p.sum())
        g_sum = int(g.sum())
        if p_sum + g_sum == 0:
            continue
        dice = 2.0 * inter / max(p_sum + g_sum, 1)
        union = p_sum + g_sum - inter
        iou = inter / max(union, 1)
        dice_list.append(dice)
        iou_list.append(iou)
        per_class[str(cid)] = {
            "dice": float(dice),
            "iou": float(iou),
            "pred_area": p_sum,
            "gt_area": g_sum,
        }

    tp = int((pred_fg & gt_fg & (pred == gt)).sum())
    confusion = int((pred_fg & gt_fg & (pred != gt)).sum())
    fp = int((pred_fg & (~gt_fg)).sum())
    # Definite under-activation is background on GT foreground. Unknown is reported separately.
    fn = int((pred_background & gt_fg).sum())
    unknown_gt = int((unknown & gt_fg).sum())

    total = int(gt.size)
    gt_fg_n = int(gt_fg.sum())
    pred_fg_n = int(pred_fg.sum())
    gt_bg_n = total - gt_fg_n

    comps, largest_ratio, tiny_ratio = components_stats(pred, class_ids)
    n_present_gt = len([cid for cid in class_ids if np.any(gt == cid)])
    fragmentation = max(comps - n_present_gt, 0) / max(n_present_gt, 1)

    return {
        "dice_macro": float(np.mean(dice_list)) if dice_list else np.nan,
        "iou_macro": float(np.mean(iou_list)) if iou_list else np.nan,
        "unknown_ratio": float(unknown.mean()),
        "unknown_on_gt": float(unknown_gt / max(gt_fg_n, 1)),
        "fg_ratio": float(pred_fg_n / total),
        "gt_fg_ratio": float(gt_fg_n / total),
        "false_activation_ratio": float(fp / total),
        "fp_over_pred_fg": float(fp / max(pred_fg_n, 1)),
        "fp_over_gt_bg": float(fp / max(gt_bg_n, 1)),
        "under_activation_ratio": float(fn / max(gt_fg_n, 1)),
        "class_confusion_ratio": float(confusion / max(gt_fg_n, 1)),
        "correct_fg_ratio": float(tp / max(gt_fg_n, 1)),
        "num_connected_components": int(comps),
        "largest_component_ratio": float(largest_ratio),
        "tiny_fragment_ratio": float(tiny_ratio),
        "fragmentation_score": float(fragmentation),
        "far_fp_ratio": boundary_far_fp_ratio(pred, gt, radius=3),
        "per_class": per_class,
        "class_ids": class_ids,
    }


def flatten_metrics(prefix: str, metrics: Dict[str, Any], out: Dict[str, Any]) -> None:
    for key, value in metrics.items():
        if key in {"per_class", "class_ids"}:
            continue
        out[f"{prefix}_{key}"] = value


def parse_instances(
    prompt_meta: Optional[dict],
    prompt_space: str,
) -> List[Tuple[np.ndarray, int]]:
    if not prompt_meta:
        return []

    singular_keys = {
        "teacher": ("bbox_teacher", "bbox"),
        "student": ("bbox_student", "bbox"),
        "native": ("bbox_native", "bbox"),
    }[prompt_space]
    plural_keys = {
        "teacher": ("bboxes_teacher", "bboxes"),
        "student": ("bboxes_student", "bboxes"),
        "native": ("bboxes_native", "bboxes"),
    }[prompt_space]

    def normalize_box(value: Any) -> Optional[np.ndarray]:
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size != 4 or not np.isfinite(array).all():
            return None
        return array

    out: List[Tuple[np.ndarray, int]] = []
    instances = prompt_meta.get("instances")
    if isinstance(instances, list):
        for instance in instances:
            if not isinstance(instance, dict):
                continue
            box = None
            for key in singular_keys:
                box = normalize_box(instance.get(key))
                if box is not None:
                    break
            if box is None:
                continue
            out.append((box, int(instance.get("label_id", 1))))
        return out

    boxes = None
    for key in plural_keys:
        if isinstance(prompt_meta.get(key), list):
            boxes = prompt_meta[key]
            break
    if boxes is not None:
        labels = prompt_meta.get("label_ids", [1] * len(boxes))
        for index, value in enumerate(boxes):
            box = normalize_box(value)
            if box is None:
                continue
            label_id = int(labels[index] if index < len(labels) else 1)
            out.append((box, label_id))
        return out

    for key in singular_keys:
        box = normalize_box(prompt_meta.get(key))
        if box is not None:
            return [(box, int(prompt_meta.get("label_id", 1)))]
    return []


def teacher_box_to_student(
    box: np.ndarray,
    geom: dict,
    student_hw: Tuple[int, int],
) -> np.ndarray:
    teacher, student = _geometry_transforms(geom, student_hw)
    tw = float(teacher["new_w"])
    th = float(teacher["new_h"])
    tx = float(teacher.get("offset_x", 0))
    ty = float(teacher.get("offset_y", 0))
    sw = float(student["new_w"])
    sh = float(student["new_h"])
    sx = float(student.get("offset_x", 0))
    sy = float(student.get("offset_y", 0))
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    return np.array([
        (x1 - tx) * sw / max(tw, 1.0) + sx,
        (y1 - ty) * sh / max(th, 1.0) + sy,
        (x2 - tx) * sw / max(tw, 1.0) + sx,
        (y2 - ty) * sh / max(th, 1.0) + sy,
    ], dtype=np.float32)


def native_box_to_student(
    box: np.ndarray,
    geom: dict,
    student_hw: Tuple[int, int],
) -> np.ndarray:
    _, student = _geometry_transforms(geom, student_hw)
    oh = float(student["orig_h"])
    ow = float(student["orig_w"])
    sw = float(student["new_w"])
    sh = float(student["new_h"])
    sx = float(student.get("offset_x", 0))
    sy = float(student.get("offset_y", 0))
    x1, y1, x2, y2 = [float(v) for v in box.tolist()]
    return np.array([
        x1 * sw / max(ow, 1.0) + sx,
        y1 * sh / max(oh, 1.0) + sy,
        x2 * sw / max(ow, 1.0) + sx,
        y2 * sh / max(oh, 1.0) + sy,
    ], dtype=np.float32)


def map_boxes_to_student(
    instances: Sequence[Tuple[np.ndarray, int]],
    geom: Optional[dict],
    student_hw: Tuple[int, int],
    prompt_space: str,
) -> List[Tuple[np.ndarray, int]]:
    out = []
    sh, sw = student_hw
    for box, cid in instances:
        if prompt_space == "student":
            mapped = box.copy()
        elif prompt_space == "native":
            if geom is None:
                raise ValueError("geometry_meta is required for native->student box mapping")
            mapped = native_box_to_student(box, geom, student_hw)
        else:
            if geom is None:
                raise ValueError("geometry_meta is required for teacher->student box mapping")
            mapped = teacher_box_to_student(box, geom, student_hw)
        mapped[0::2] = np.clip(mapped[0::2], 0, sw - 1)
        mapped[1::2] = np.clip(mapped[1::2], 0, sh - 1)
        out.append((mapped, int(cid)))
    return out


def build_records(
    dataset: str,
    fold_root: Path,
    baseline_fold_root: Path,
    method: str,
    baseline_name: str,
    sac_name: str,
    only_box: bool,
    strict: bool,
    allow_partial: bool,
    max_samples_scan: int,
    max_warning_examples: int,
) -> Tuple[List[SampleRecord], Dict[int, str], List[str], Dict[str, int]]:
    warnings: List[str] = []
    warning_overflow = 0

    def add_warning(message: str) -> None:
        nonlocal warning_overflow
        if len(warnings) < max(max_warning_examples, 0):
            warnings.append(message)
        else:
            warning_overflow += 1

    manifest_path = fold_root / "meta" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = normalize_manifest(load_json(manifest_path))

    prompt_file = find_prompt_file(fold_root, method)
    prompts = normalize_named_records(load_json(prompt_file)) if prompt_file else {}
    if prompt_file is None:
        add_warning("No prompt JSON found; box samples cannot render supervision boxes.")

    split_file = find_full_box_file(fold_root, method)
    split_map = normalize_named_records(load_json(split_file)) if split_file else {}
    if split_file is None:
        add_warning("No full/box split JSON found; samples default to box mode.")

    geometry_path = fold_root / "meta" / "geometry_meta.json"
    geometry = normalize_named_records(load_json(geometry_path)) if geometry_path.exists() else {}
    if not geometry_path.exists():
        add_warning("No geometry_meta.json found; native display and teacher-box mapping will fail.")

    baseline_dir = baseline_fold_root / "pseudo_student" / baseline_name
    sac_dir = fold_root / "pseudo_student" / sac_name
    if not baseline_dir.exists():
        raise FileNotFoundError(f"Baseline pseudo directory missing: {baseline_dir}")
    if not sac_dir.exists():
        raise FileNotFoundError(f"SAC pseudo directory missing: {sac_dir}")

    stats: Dict[str, int] = {
        "manifest_items": len(manifest),
        "train_items_seen": 0,
        "accepted_records": 0,
        "skipped_non_train": 0,
        "skipped_only_box": 0,
        "skipped_missing_core": 0,
        "skipped_missing_baseline": 0,
        "skipped_missing_sac": 0,
        "skipped_bad_gt": 0,
        "missing_prompt_box_samples": 0,
        "missing_geometry": 0,
    }

    records: List[SampleRecord] = []
    observed_classes: Set[int] = set()

    for item in manifest:
        if str(item.get("split", "")) != "train":
            stats["skipped_non_train"] += 1
            continue
        stats["train_items_seen"] += 1

        slice_name = str(item.get("slice_name") or item.get("filename") or "")
        if not slice_name:
            message = "manifest train item has no slice_name"
            if strict:
                raise KeyError(message)
            add_warning(message)
            stats["skipped_missing_core"] += 1
            continue

        split_record = split_map.get(slice_name, {})
        label_mode = str(split_record.get("label_mode", "box")).lower()
        if label_mode not in {"full", "box"}:
            message = f"{slice_name}: invalid label_mode={label_mode!r}"
            if strict:
                raise ValueError(message)
            add_warning(message)
            label_mode = "box"
        if only_box and label_mode != "box":
            stats["skipped_only_box"] += 1
            continue

        image_value = item.get("student_img") or item.get("student_image") or item.get("image")
        gt_value = item.get("student_gt") or item.get("gt") or item.get("student_label")
        if image_value is None or gt_value is None:
            message = f"{slice_name}: manifest lacks student_img/student_gt"
            if strict:
                raise KeyError(message)
            add_warning(message)
            stats["skipped_missing_core"] += 1
            continue

        image_path = resolve_path(fold_root, image_value)
        gt_path = resolve_path(fold_root, gt_value)
        baseline_path = baseline_dir / slice_name
        sac_path = sac_dir / slice_name

        core_missing = [path for path in (image_path, gt_path) if not path.exists()]
        if core_missing:
            message = f"{slice_name}: missing core files: {', '.join(map(str, core_missing))}"
            if strict:
                raise FileNotFoundError(message)
            add_warning(message)
            stats["skipped_missing_core"] += 1
            continue

        if not baseline_path.exists():
            stats["skipped_missing_baseline"] += 1
            message = f"{slice_name}: missing baseline pseudo: {baseline_path}"
            if not allow_partial:
                raise FileNotFoundError(message)
            add_warning(message)
            continue
        if not sac_path.exists():
            stats["skipped_missing_sac"] += 1
            message = f"{slice_name}: missing SAC pseudo: {sac_path}"
            if not allow_partial:
                raise FileNotFoundError(message)
            add_warning(message)
            continue

        native_value = None
        for native_key in (
            "native_img", "native_image", "image_native", "orig_img",
            "original_image", "source_image", "raw_image",
        ):
            if item.get(native_key) not in (None, ""):
                native_value = item.get(native_key)
                break
        native_image_path = resolve_path(fold_root, native_value) if native_value is not None else None
        if native_image_path is not None and not native_image_path.exists():
            add_warning(
                f"{slice_name}: declared native image missing; reconstructing from student canvas: "
                f"{native_image_path}"
            )
            native_image_path = None

        geometry_key = str(item.get("geometry_key") or slice_name)
        geometry_meta = geometry.get(slice_name, geometry.get(geometry_key))
        if geometry_meta is None:
            stats["missing_geometry"] += 1

        prompt_meta = prompts.get(slice_name)
        if label_mode == "box" and not prompt_meta:
            stats["missing_prompt_box_samples"] += 1
            message = f"{slice_name}: box sample has no prompt metadata"
            if strict:
                raise KeyError(message)
            add_warning(message)

        try:
            gt = load_array_2d(gt_path, "gt")
            observed_classes.update(
                int(value) for value in np.unique(gt) if int(value) not in (0, 255)
            )
        except Exception as exc:
            if strict:
                raise
            add_warning(f"{slice_name}: failed reading GT: {type(exc).__name__}: {exc}")
            stats["skipped_bad_gt"] += 1
            continue

        raw_case_id = item.get("case_id")
        case_text = str(raw_case_id).strip() if raw_case_id not in (None, "") else ""
        if case_text.lower() in {"", "unknown", "unknown_case", "none", "null"}:
            case_id = Path(slice_name).stem
        else:
            case_id = case_text

        raw_slice_idx = item.get("slice_idx")
        try:
            slice_idx = int(raw_slice_idx) if raw_slice_idx not in (None, "") else -1
        except (TypeError, ValueError):
            slice_idx = -1
            add_warning(f"{slice_name}: invalid slice_idx={raw_slice_idx!r}; fallback to -1")

        records.append(
            SampleRecord(
                slice_name=slice_name,
                split="train",
                case_id=case_id,
                slice_idx=slice_idx,
                image_path=image_path,
                native_image_path=native_image_path,
                gt_path=gt_path,
                baseline_path=baseline_path,
                sac_path=sac_path,
                label_mode=label_mode,
                prompt_meta=prompt_meta,
                geometry_meta=geometry_meta,
            )
        )
        stats["accepted_records"] += 1
        if max_samples_scan > 0 and len(records) >= max_samples_scan:
            break

    if warning_overflow:
        warnings.append(f"... {warning_overflow} additional warnings omitted")

    class_names = parse_label_names(
        dataset,
        fold_root / "meta" / "label_meta.json",
        observed_classes,
    )
    return records, class_names, warnings, stats


def validate_label_values(
    array: np.ndarray,
    role: str,
    class_ids: Set[int],
    slice_name: str,
) -> None:
    allowed = {0, 255, *class_ids}
    values = set(int(value) for value in np.unique(array).tolist())
    invalid = sorted(values - allowed)
    if invalid:
        raise ValueError(
            f"{slice_name}: {role} contains invalid labels {invalid}; "
            f"allowed={sorted(allowed)}"
        )


def inspect_sample(
    record: SampleRecord,
    class_names: Dict[int, str],
    strict: bool,
    prompt_space: str,
    display_mode: str,
    auto_grayscale: bool,
    force_grayscale: bool,
    validate_labels: bool,
) -> RenderedSample:
    student_image_raw = load_array_2d(record.image_path, "image")
    student_image_rgb = normalize_image_to_rgb(
        student_image_raw,
        force_grayscale=force_grayscale,
        auto_grayscale=auto_grayscale,
    )
    gt_student = load_array_2d(record.gt_path, "gt")
    baseline_student = load_array_2d(record.baseline_path, "baseline")
    sac_student = load_array_2d(record.sac_path, "sac")
    baseline_student, sac_student = validate_same_shape(
        gt_student, baseline_student, sac_student, strict
    )

    if validate_labels:
        class_ids = set(int(value) for value in class_names)
        validate_label_values(gt_student, "GT", class_ids, record.slice_name)
        validate_label_values(baseline_student, "baseline pseudo", class_ids, record.slice_name)
        validate_label_values(sac_student, "SAC pseudo", class_ids, record.slice_name)

    if student_image_rgb.shape[:2] != gt_student.shape:
        if strict:
            raise ValueError(
                f"Image/GT shape mismatch: image={student_image_rgb.shape[:2]} "
                f"gt={gt_student.shape} for {record.slice_name}"
            )
        student_image_rgb = cv2.resize(
            student_image_rgb,
            (gt_student.shape[1], gt_student.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    # Metrics are always computed in the original student-label space.
    bm = quality_metrics(baseline_student, gt_student)
    sm = quality_metrics(sac_student, gt_student)

    instances = parse_instances(record.prompt_meta, prompt_space)
    image_source = "student_canvas"

    if display_mode == "student":
        image_rgb = student_image_rgb
        gt = gt_student
        baseline = baseline_student
        sac = sac_student
        boxes = (
            map_boxes_to_student(
                instances, record.geometry_meta, gt_student.shape, prompt_space
            )
            if instances else []
        )
    else:
        if record.geometry_meta is None:
            if strict:
                raise ValueError(
                    f"{record.slice_name}: native display requires geometry_meta"
                )
            image_rgb = student_image_rgb
            gt = gt_student
            baseline = baseline_student
            sac = sac_student
            boxes = (
                map_boxes_to_student(
                    instances, record.geometry_meta, gt_student.shape, prompt_space
                )
                if instances else []
            )
            image_source = "student_fallback_no_geometry"
        else:
            if record.native_image_path is not None:
                native_raw = load_array_2d(record.native_image_path, "image")
                image_rgb = normalize_image_to_rgb(
                    native_raw,
                    force_grayscale=force_grayscale,
                    auto_grayscale=auto_grayscale,
                )
                _, student_geom = _geometry_transforms(
                    record.geometry_meta, gt_student.shape
                )
                oh = int(round(float(student_geom["orig_h"])))
                ow = int(round(float(student_geom["orig_w"])))
                if image_rgb.shape[:2] != (oh, ow):
                    image_rgb = cv2.resize(
                        image_rgb, (ow, oh), interpolation=cv2.INTER_LINEAR
                    )
                image_source = "native_manifest"
            else:
                image_rgb = student_image_to_native(
                    student_image_rgb, record.geometry_meta
                )
                image_source = "native_reconstructed_from_student"

            gt = student_mask_to_native(gt_student, record.geometry_meta)
            baseline = student_mask_to_native(
                baseline_student, record.geometry_meta
            )
            sac = student_mask_to_native(sac_student, record.geometry_meta)
            boxes = (
                map_boxes_to_native(
                    instances, record.geometry_meta, gt_student.shape, prompt_space
                )
                if instances else []
            )

            if image_rgb.shape[:2] != gt.shape:
                if strict:
                    raise ValueError(
                        f"{record.slice_name}: native image/mask mismatch "
                        f"image={image_rgb.shape[:2]} mask={gt.shape}"
                    )
                image_rgb = cv2.resize(
                    image_rgb, (gt.shape[1], gt.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )

                image_source += "_focus"

    metrics: Dict[str, Any] = {
        "slice_name": record.slice_name,
        "case_id": record.case_id,
        "slice_idx": record.slice_idx,
        "label_mode": record.label_mode,
        "gt_fg_ratio": bm["gt_fg_ratio"],
        "classes_present": ",".join(map(str, get_class_ids(gt_student))),
        "display_mode": display_mode,
        "image_source": image_source,
    }
    flatten_metrics("baseline", bm, metrics)
    flatten_metrics("sac", sm, metrics)
    metrics.update({
        "delta_dice": sm["dice_macro"] - bm["dice_macro"],
        "delta_iou": sm["iou_macro"] - bm["iou_macro"],
        "delta_unknown_on_gt": sm["unknown_on_gt"] - bm["unknown_on_gt"],
        "delta_false_activation": sm["false_activation_ratio"] - bm["false_activation_ratio"],
        "delta_under_activation": sm["under_activation_ratio"] - bm["under_activation_ratio"],
        "delta_class_confusion": sm["class_confusion_ratio"] - bm["class_confusion_ratio"],
        "delta_fragmentation": sm["fragmentation_score"] - bm["fragmentation_score"],
        "delta_far_fp": sm["far_fp_ratio"] - bm["far_fp_ratio"],
        "baseline_per_class_json": json.dumps(bm["per_class"], ensure_ascii=False),
        "sac_per_class_json": json.dumps(sm["per_class"], ensure_ascii=False),
    })
    return RenderedSample(
        record=record,
        image_rgb=image_rgb,
        gt=gt,
        baseline=baseline,
        sac=sac,
        boxes_student=boxes,
        class_names=class_names,
        metrics=metrics,
        display_mode=display_mode,
        image_source=image_source,
    )


def color_for_class(cid: int) -> Tuple[int, int, int]:
    if cid in CLASS_COLORS:
        return CLASS_COLORS[cid]
    # Deterministic fallback generated in HSV.
    hue = ((cid * 0.61803398875) % 1.0) * 179
    hsv = np.uint8([[[int(hue), 180, 235]]])
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
    return tuple(int(x) for x in rgb)


def overlay_segmentation(image_rgb: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    out = image_rgb.astype(np.float32).copy()
    overlay = image_rgb.astype(np.float32).copy()
    for cid in get_class_ids(mask):
        overlay[mask == cid] = np.asarray(color_for_class(cid), dtype=np.float32)
    overlay[mask == 255] = np.asarray((175, 175, 175), dtype=np.float32)
    active = mask != 0
    out[active] = (1.0 - alpha) * out[active] + alpha * overlay[active]
    out = np.clip(out, 0, 255).astype(np.uint8)

    # Thin contours help separate neighboring structures.
    for cid in get_class_ids(mask):
        binary = (mask == cid).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_color = tuple(int(v) for v in color_for_class(cid))
        cv2.drawContours(out, contours, -1, contour_color, 1, cv2.LINE_AA)
    return out


def render_supervision(sample: RenderedSample, alpha: float) -> np.ndarray:
    if sample.record.label_mode == "full":
        out = overlay_segmentation(sample.image_rgb, sample.gt, alpha)
        cv2.putText(out, "FULL MASK", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, "FULL MASK", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
        return out

    out = sample.image_rgb.copy()
    if not sample.boxes_student:
        cv2.putText(out, "BOX MISSING", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(out, "BOX MISSING", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 40, 40), 1, cv2.LINE_AA)
        return out

    h, w = out.shape[:2]
    thickness = max(2, int(round(min(h, w) / 180)))
    for box, cid in sample.boxes_student:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = color_for_class(cid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        label = sample.class_names.get(cid, f"C{cid}")
        label = label[:18]
        y_text = max(16, y1 - 4)
        cv2.putText(out, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
    return out


def error_overlay(image_rgb: np.ndarray, pred: np.ndarray, gt: np.ndarray, alpha: float) -> np.ndarray:
    unknown = pred == 255
    pred_fg = (pred > 0) & (~unknown)
    gt_fg = gt > 0
    tp = pred_fg & gt_fg & (pred == gt)
    confusion = pred_fg & gt_fg & (pred != gt)
    fp = pred_fg & (~gt_fg)
    fn = (~pred_fg) & gt_fg & (~unknown)

    label = np.zeros(gt.shape, dtype=np.uint8)
    label[tp] = 1
    label[fp] = 2
    label[fn] = 3
    label[confusion] = 4
    label[unknown] = 5

    colors = {
        1: ERROR_COLORS["tp"],
        2: ERROR_COLORS["fp"],
        3: ERROR_COLORS["fn"],
        4: ERROR_COLORS["confusion"],
        5: ERROR_COLORS["unknown"],
    }
    out = image_rgb.astype(np.float32).copy()
    overlay = image_rgb.astype(np.float32).copy()
    for key, color in colors.items():
        overlay[label == key] = np.asarray(color, dtype=np.float32)
    active = label > 0
    out[active] = (1.0 - alpha) * out[active] + alpha * overlay[active]
    return np.clip(out, 0, 255).astype(np.uint8)


def sample_panels(sample: RenderedSample, overlay_alpha: float, error_alpha: float) -> List[np.ndarray]:
    return [
        sample.image_rgb,
        render_supervision(sample, overlay_alpha),
        overlay_segmentation(sample.image_rgb, sample.baseline, overlay_alpha),
        overlay_segmentation(sample.image_rgb, sample.sac, overlay_alpha),
        overlay_segmentation(sample.image_rgb, sample.gt, overlay_alpha),
        error_overlay(sample.image_rgb, sample.baseline, sample.gt, error_alpha),
        error_overlay(sample.image_rgb, sample.sac, sample.gt, error_alpha),
    ]


def metric_caption(m: Dict[str, Any]) -> str:
    return (
        f"B Dice={m['baseline_dice_macro']:.3f} | SAC={m['sac_dice_macro']:.3f} | "
        f"Δ={m['delta_dice']:+.3f} | "
        f"Ugt {m['baseline_unknown_on_gt']:.2f}→{m['sac_unknown_on_gt']:.2f} | "
        f"FP {m['baseline_false_activation_ratio']:.3f}→{m['sac_false_activation_ratio']:.3f}"
    )


def class_legend_handles(samples: Sequence[RenderedSample]) -> List[Patch]:
    ids: Set[int] = set()
    for s in samples:
        ids.update(get_class_ids(s.gt, s.baseline, s.sac))
    handles = []
    names = samples[0].class_names if samples else {}
    for cid in sorted(ids):
        color = np.array(color_for_class(cid)) / 255.0
        handles.append(Patch(facecolor=color, edgecolor="none", label=f"{cid}: {names.get(cid, f'Class {cid}') }"))
    return handles


def error_legend_handles() -> List[Patch]:
    labels = [
        ("tp", "Correct foreground"),
        ("fp", "False positive"),
        ("fn", "False negative"),
        ("confusion", "Wrong class"),
        ("unknown", "Unknown (255)"),
    ]
    return [Patch(facecolor=np.array(ERROR_COLORS[k]) / 255.0, edgecolor="none", label=text) for k, text in labels]


def show_panel(ax: Any, img: np.ndarray, title: Optional[str] = None) -> None:
    """Render one panel without implicit cropping, stretching, or axis drift."""
    h, w = img.shape[:2]
    if img.ndim == 2:
        ax.imshow(
            img,
            cmap="gray",
            vmin=0,
            vmax=255,
            origin="upper",
            interpolation="nearest",
        )
    else:
        ax.imshow(img, origin="upper", interpolation="nearest")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_anchor("C")
    ax.axis("off")
    if title is not None:
        ax.set_title(title, fontsize=10, pad=5)


def render_individual_figure(sample: RenderedSample, save_path: Path, overlay_alpha: float, error_alpha: float, dpi: int) -> None:
    panels = sample_panels(sample, overlay_alpha, error_alpha)
    titles = ["Input", "Supervision", "Baseline pseudo", "SAC pseudo", "Ground truth", "Baseline error", "SAC error"]
    fig, axes = plt.subplots(1, 7, figsize=(18.0, 3.45), constrained_layout=False)
    for ax, img, title in zip(axes, panels, titles):
        show_panel(ax, img, title)
    fig.suptitle(
        f"{sample.record.slice_name} | {sample.record.case_id} | {sample.record.label_mode.upper()}\n{metric_caption(sample.metrics)}",
        fontsize=11,
        y=0.99,
    )
    handles = class_legend_handles([sample]) + error_legend_handles()
    ncol = min(7, max(3, len(handles)))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=ncol, frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.77, bottom=0.18, wspace=0.025)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_contact_sheet(
    samples: Sequence[RenderedSample],
    group_name: str,
    save_path: Path,
    overlay_alpha: float,
    error_alpha: float,
    dpi: int,
) -> None:
    if not samples:
        return
    titles = ["Input", "Supervision", "Baseline pseudo", "SAC pseudo", "Ground truth", "Baseline error", "SAC error"]
    nrows = len(samples)
    fig, axes = plt.subplots(nrows, 7, figsize=(17.6, 2.55 * nrows + 1.4), squeeze=False)
    for r, sample in enumerate(samples):
        panels = sample_panels(sample, overlay_alpha, error_alpha)
        for c, (img, title) in enumerate(zip(panels, titles)):
            ax = axes[r, c]
            show_panel(ax, img, title if r == 0 else None)
        axes[r, 0].text(
            -0.04, 0.5,
            f"{Path(sample.record.slice_name).stem}\n{sample.record.label_mode}\nΔDice={sample.metrics['delta_dice']:+.3f}",
            transform=axes[r, 0].transAxes,
            ha="right", va="center", fontsize=7.5,
        )
    dataset = save_path.parent.parent.name
    fig.suptitle(f"{dataset} — {group_name.replace('_', ' ').title()}", fontsize=14, y=0.995)
    handles = class_legend_handles(samples) + error_legend_handles()
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=min(7, len(handles)), frameon=False, fontsize=8)
    fig.subplots_adjust(left=0.10, right=0.995, top=0.94, bottom=0.07, hspace=0.10, wspace=0.025)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _case_limit_ok(row: pd.Series, case_counts: Counter, max_per_case: int, is_3d: bool) -> bool:
    if not is_3d:
        return True
    return case_counts[str(row["case_id"])] < max_per_case


def pick_rows(
    df: pd.DataFrame,
    ordered_indices: Iterable[int],
    n: int,
    used: Set[str],
    case_counts: Counter,
    max_per_case: int,
    is_3d: bool,
) -> List[int]:
    selected = []
    for idx in ordered_indices:
        row = df.loc[idx]
        name = str(row["slice_name"])
        if name in used:
            continue
        if not _case_limit_ok(row, case_counts, max_per_case, is_3d):
            continue
        selected.append(idx)
        used.add(name)
        case_counts[str(row["case_id"])] += 1
        if len(selected) >= n:
            break
    return selected


def full_reference_order(df: pd.DataFrame) -> List[int]:
    if df.empty:
        return []
    sdf = df.sort_values("gt_fg_ratio")
    if len(sdf) == 1:
        return [int(sdf.index[0])]
    # Quantile-spread order: covers small, medium and large targets.
    positions = np.linspace(0, len(sdf) - 1, min(len(sdf), 20)).round().astype(int)
    return [int(sdf.index[p]) for p in positions]


def rare_class_order(df: pd.DataFrame) -> List[int]:
    frequencies: Counter = Counter()
    row_classes: Dict[int, Set[int]] = {}
    for idx, text in df["classes_present"].items():
        classes = {int(x) for x in str(text).split(",") if str(x).strip()}
        row_classes[int(idx)] = classes
        frequencies.update(classes)
    scores = {}
    for idx, classes in row_classes.items():
        scores[idx] = sum(1.0 / max(frequencies[c], 1) for c in classes)
    return sorted(scores, key=scores.get, reverse=True)


def select_diagnostic_groups(
    df: pd.DataFrame,
    per_group: int,
    max_per_case: int,
    include_full: bool,
    is_3d: bool,
) -> Dict[str, List[int]]:
    if df.empty:
        return {}
    used: Set[str] = set()
    case_counts: Counter = Counter()
    groups: Dict[str, List[int]] = {}

    box = df[df["label_mode"] == "box"].copy()
    full = df[df["label_mode"] == "full"].copy()
    if box.empty:
        box = df.copy()

    if include_full and not full.empty:
        groups["full_reference"] = pick_rows(
            df,
            full_reference_order(full),
            per_group,
            used,
            case_counts,
            max_per_case,
            is_3d,
        )

    orders: List[Tuple[str, Iterable[int]]] = [
        (
            "strong_improvement",
            box.sort_values(
                ["delta_dice", "sac_dice_macro"],
                ascending=[False, False],
            ).index,
        )
    ]

    median_delta = float(box["delta_dice"].median())
    typical_order = (box["delta_dice"] - median_delta).abs().sort_values().index
    orders.append(("typical", typical_order))
    orders.append((
        "regression",
        box.sort_values(["delta_dice", "sac_dice_macro"], ascending=[True, True]).index,
    ))
    orders.append((
        "difficult",
        box.sort_values(
            ["sac_dice_macro", "sac_under_activation_ratio"],
            ascending=[True, False],
        ).index,
    ))
    orders.append((
        "fp_risk",
        box.sort_values(
            ["delta_false_activation", "sac_far_fp_ratio", "sac_fp_over_pred_fg"],
            ascending=[False, False, False],
        ).index,
    ))

    uncertainty_score = (
        box["sac_unknown_on_gt"].fillna(0)
        + 0.10 * box["sac_fragmentation_score"].fillna(0)
        + 0.50 * box["sac_tiny_fragment_ratio"].fillna(0)
    )
    orders.append(("unknown_fragmented", uncertainty_score.sort_values(ascending=False).index))

    all_classes = {
        class_id
        for text_value in df["classes_present"]
        for class_id in str(text_value).split(",")
        if class_id
    }
    if df["classes_present"].str.contains(",", regex=False).any() or len(all_classes) > 1:
        orders.append(("rare_class_coverage", rare_class_order(box)))

    for name, order in orders:
        groups[name] = pick_rows(
            df,
            order,
            per_group,
            used,
            case_counts,
            max_per_case,
            is_3d,
        )
    return groups


def summarize_dataset(
    df: pd.DataFrame,
    groups: Dict[str, List[int]],
    summary_scope: str,
) -> Dict[str, Any]:
    if summary_scope == "box":
        scope_df = df[df["label_mode"] == "box"].copy()
    else:
        scope_df = df.copy()
    valid = scope_df[np.isfinite(scope_df["delta_dice"])].copy()

    summary = {
        "num_samples_visualized": int(len(df)),
        "num_box_visualized": int((df["label_mode"] == "box").sum()),
        "num_full_visualized": int((df["label_mode"] == "full").sum()),
        "summary_scope": summary_scope,
        "num_samples_in_summary": int(len(valid)),
        "num_dice_improved": int((valid["delta_dice"] > 0).sum()),
        "num_dice_regressed": int((valid["delta_dice"] < 0).sum()),
        "num_dice_tied": int((valid["delta_dice"] == 0).sum()),
        "improved_fraction": float((valid["delta_dice"] > 0).mean()) if len(valid) else np.nan,
        "mean_baseline_dice": float(valid["baseline_dice_macro"].mean()) if len(valid) else np.nan,
        "mean_sac_dice": float(valid["sac_dice_macro"].mean()) if len(valid) else np.nan,
        "mean_delta_dice": float(valid["delta_dice"].mean()) if len(valid) else np.nan,
        "mean_delta_iou": float(valid["delta_iou"].mean()) if len(valid) else np.nan,
        "mean_delta_unknown_on_gt": float(valid["delta_unknown_on_gt"].mean()) if len(valid) else np.nan,
        "mean_delta_false_activation": float(valid["delta_false_activation"].mean()) if len(valid) else np.nan,
        "mean_delta_under_activation": float(valid["delta_under_activation"].mean()) if len(valid) else np.nan,
        "mean_delta_class_confusion": float(valid["delta_class_confusion"].mean()) if len(valid) else np.nan,
        "mean_delta_fragmentation": float(valid["delta_fragmentation"].mean()) if len(valid) else np.nan,
        "selected_groups": {key: len(value) for key, value in groups.items()},
    }
    return summary


def render_dataset(
    dataset: str,
    fold_root: Path,
    baseline_fold_root: Path,
    output_root: Path,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    dataset_out = output_root / dataset
    dataset_out.mkdir(parents=True, exist_ok=True)

    records, class_names, warnings, build_stats = build_records(
        dataset=dataset,
        fold_root=fold_root,
        baseline_fold_root=baseline_fold_root,
        method=args.method,
        baseline_name=args.baseline_pseudo_name,
        sac_name=args.sac_pseudo_name,
        only_box=args.only_box,
        strict=args.strict,
        allow_partial=args.allow_partial,
        max_samples_scan=args.max_samples_scan,
        max_warning_examples=args.max_warning_examples,
    )
    if not records:
        raise RuntimeError(f"No valid train records for {dataset}")

    samples: Dict[str, RenderedSample] = {}
    rows: List[Dict[str, Any]] = []
    grayscale_datasets = {
        x.strip().lower()
        for x in str(args.grayscale_datasets).split(",")
        if x.strip()
    }
    runtime_warnings = list(warnings)
    print(f"[{dataset}] scanning {len(records)} train samples ...", flush=True)
    for i, record in enumerate(records, 1):
        try:
            sample = inspect_sample(
                record=record,
                class_names=class_names,
                strict=args.strict,
                prompt_space=args.prompt_space,
                display_mode=args.display_mode,
                auto_grayscale=args.auto_grayscale,
                force_grayscale=dataset.lower() in grayscale_datasets,
                validate_labels=args.validate_labels,
            )
            samples[record.slice_name] = sample
            rows.append(sample.metrics)
        except Exception as exc:
            if args.strict:
                raise
            runtime_warnings.append(f"{record.slice_name}: {type(exc).__name__}: {exc}")
        if i % 500 == 0 or i == len(records):
            print(f"[{dataset}] inspected {i}/{len(records)}", flush=True)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No samples successfully inspected for {dataset}")
    metrics_path = dataset_out / "metrics_all_train.csv"
    df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    is_3d = dataset.lower() in KNOWN_3D_DATASETS
    groups = select_diagnostic_groups(
        df,
        args.per_group,
        args.max_per_case,
        args.include_full,
        is_3d=is_3d,
    )
    selected_rows = []
    for group_name, indices in groups.items():
        for rank, idx in enumerate(indices, 1):
            rec = df.loc[idx].to_dict()
            rec["diagnostic_group"] = group_name
            rec["group_rank"] = rank
            selected_rows.append(rec)
    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(dataset_out / "selected_cases.csv", index=False, encoding="utf-8-sig")

    for group_name, indices in groups.items():
        selected_samples = [samples[str(df.loc[idx, "slice_name"])] for idx in indices]
        if args.render_individual:
            for rank, sample in enumerate(selected_samples, 1):
                stem = Path(sample.record.slice_name).stem
                save_path = dataset_out / "individual" / group_name / f"{rank:02d}_{stem}.png"
                render_individual_figure(sample, save_path, args.overlay_alpha, args.error_alpha, args.dpi)
        if args.render_contact_sheets:
            render_contact_sheet(
                selected_samples,
                group_name,
                dataset_out / "contact_sheets" / f"{group_name}.png",
                args.overlay_alpha,
                args.error_alpha,
                args.sheet_dpi,
            )

    summary = summarize_dataset(df, groups, args.summary_scope)
    summary.update({
        "dataset": dataset,
        "fold_root": str(fold_root),
        "baseline_fold_root": str(baseline_fold_root),
        "baseline_pseudo_name": args.baseline_pseudo_name,
        "sac_pseudo_name": args.sac_pseudo_name,
        "prompt_space": args.prompt_space,
        "display_mode": args.display_mode,
        "auto_grayscale": args.auto_grayscale,
        "grayscale_datasets": args.grayscale_datasets,
        "output_dir": str(dataset_out),
        "warnings_count": len(runtime_warnings),
        "allow_partial": bool(args.allow_partial),
        "is_3d": bool(is_3d),
        "build_stats": build_stats,
    })
    save_json(summary, dataset_out / "summary.json")
    (dataset_out / "warnings.log").write_text("\n".join(runtime_warnings), encoding="utf-8")
    print(
        f"[{dataset}] done: mean ΔDice={summary['mean_delta_dice']:+.4f}, "
        f"improved={summary['num_dice_improved']}/{summary['num_samples_in_summary']}, "
        f"visualized={summary['num_samples_visualized']}, out={dataset_out}",
        flush=True,
    )
    return summary


def write_global_summary(summaries: List[Dict[str, Any]], output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    save_json(summaries, output_root / "summary_all_datasets.json")
    flat_rows = []
    for s in summaries:
        row = {k: v for k, v in s.items() if not isinstance(v, (dict, list))}
        flat_rows.append(row)
    pd.DataFrame(flat_rows).to_csv(output_root / "summary_all_datasets.csv", index=False, encoding="utf-8-sig")


def main() -> int:
    args = parse_args()
    baseline_root = args.baseline_processed_root or args.processed_root
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    args.output_root.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in config.items()}
    save_json(config, args.output_root / "run_config.json")

    summaries: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for dataset in datasets:
        fold_root = args.processed_root / dataset / args.fold
        baseline_fold_root = baseline_root / dataset / args.fold
        try:
            summary = render_dataset(dataset, fold_root, baseline_fold_root, args.output_root, args)
            summaries.append(summary)
        except Exception as exc:
            failures.append({"dataset": dataset, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[ERROR] {dataset}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            if args.strict and len(datasets) == 1:
                raise

    write_global_summary(summaries, args.output_root)
    save_json(failures, args.output_root / "failures.json")
    print(f"All done. Successful={len(summaries)}, failed={len(failures)}")
    print(f"Output root: {args.output_root}")
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
