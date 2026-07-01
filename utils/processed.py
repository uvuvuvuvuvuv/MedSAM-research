
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
processed.py

Strict Stage-1 preprocessing with explicit native/teacher/student geometry contracts.

Key changes vs. the original processed.py:
1) student-space is a real fixed-size space, not native passthrough;
2) geometry_meta.json writes native_to_teacher / teacher_to_native / native_to_student / student_to_native;
3) manifest.json writes spacing and keeps backward-compatible spacing_xyz;
4) split_meta.json writes split_protocol / student_target_h / student_target_w;
5) native GT is saved explicitly so prompts can be defined in native space first.

Notes on student size presets:
- We prefer per-dataset settings that are directly stated in representative papers when available.
- When no single canonical per-dataset crop is clearly established, we mark the size as
  literature_backed_but_not_unique, modality_informed_default, or
  project_evidence_informed_override in split_meta.json instead of pretending it is a
  paper-mandated fact.
- All sizes remain override-able through --student_size_json so you can lock a different
  fixed crop before the formal baseline begins.
"""

from __future__ import annotations

import argparse
import json
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import nibabel as nib
import numpy as np
from PIL import Image
from stage_timer_utils import StageTimer

try:
    from pipeline_progress import ProgressWriter
except Exception:
    try:
        from utils.pipeline_progress import ProgressWriter
    except Exception:
        class ProgressWriter:  # type: ignore
            def __init__(self, *args, **kwargs): pass
            def start(self, *args, **kwargs): pass
            def update(self, *args, **kwargs): pass
            def finish(self, *args, **kwargs): pass


DEFAULT_TEACHER_SIZE = 1024

DATASET_CONFIG = {
    "tn3k":         {"kind": "2d",      "modality": "ultrasound",     "gray_cutoff": False},
    "tg3k":         {"kind": "2d",      "modality": "ultrasound",     "gray_cutoff": False},
    "kvasirseg":    {"kind": "2d",      "modality": "endoscopy",      "gray_cutoff": False},
    "cvc_clinicdb": {"kind": "2d",      "modality": "endoscopy",      "gray_cutoff": False},
    "ddti":         {"kind": "2d",      "modality": "ultrasound",     "gray_cutoff": False},
    "monuseg":      {"kind": "2d",      "modality": "histopathology", "gray_cutoff": False},
    "otu_2d":       {"kind": "2d",      "modality": "ultrasound",     "gray_cutoff": False},
    "ph2":          {"kind": "2d",      "modality": "dermoscopy",     "gray_cutoff": False},

    "btcv":         {"kind": "3d",      "modality": "ct",  "ct_clip": [-125, 275], "multi_class": True},
    "acdc":         {"kind": "3d",      "modality": "mri", "mri_pct": [0.5, 99.5], "multi_class": True},
    "prostate158":  {"kind": "3d",      "modality": "mri", "mri_pct": [0.5, 99.5], "multi_class": True},

    "synapse":      {"kind": "synapse", "modality": "ct",  "ct_clip": [-125, 275], "multi_class": True},
}


# Dataset-level mask binarization rules required by the baseline document.
# JPEG / grayscale label sources must not use the old `> 0` rule because it
# absorbs compression halos into foreground and enlarges native-space tight boxes.
GRAY_MASK_THRESHOLD_DATASETS = {"tn3k", "tg3k", "kvasirseg"}
GRAY_MASK_THRESHOLD = 128



# Conservative native-space filtering for known non-lesion tiny mask artifacts.
# This is intentionally applied only at Stage-1 preprocessing, before teacher/student
# resizing and before prompt generation, so removed artifacts cannot create fake bboxes.
#
# Safety rule in remove_tiny_false_gt_components(): a tiny component is removed only
# when at least one non-tiny foreground component remains. Therefore a sample with one
# genuinely small lesion is preserved instead of being wiped out.
DEFAULT_TINY_GT_FILTER_PRESETS: Dict[str, Dict[str, Any]] = {
    "tg3k": {
        "enabled": True,
        "min_area": 32,
        "max_bbox_side": 8,
        "connectivity": 8,
        "policy": "remove_tiny_components_only_when_large_component_exists",
        "note": "Filters known tiny non-lesion white dots in TG3K organized labels.",
    },
    "kvasirseg": {
        "enabled": True,
        "min_area": 32,
        "max_bbox_side": 8,
        "connectivity": 8,
        "policy": "remove_tiny_components_only_when_large_component_exists",
        "note": "Filters tiny corner / JPEG-residual components in Kvasir-SEG organized labels.",
    },
}

# Size-selection metadata is kept explicit so later writing can distinguish
# direct task-family anchors from project defaults.
DEFAULT_STUDENT_SIZE_PRESETS: Dict[str, Dict[str, Any]] = {
    # Literature-backed / commonly adopted per-dataset settings where we found direct evidence.
    "kvasirseg": {
        "hw": [352, 352],
        "selection_policy": "literature_backed",
        "selection_basis": "polyp_segmentation_common_352",
        "selection_note": "PraNet/SSFormer-style polyp pipelines commonly resize Kvasir-SEG to 352x352.",
    },
    "cvc_clinicdb": {
        "hw": [352, 352],
        "selection_policy": "literature_backed",
        "selection_basis": "polyp_segmentation_common_352",
        "selection_note": "PraNet/SSFormer-style polyp pipelines commonly resize CVC-ClinicDB to 352x352.",
    },
    "tn3k": {
        "hw": [256, 256],
        "selection_policy": "literature_backed",
        "selection_basis": "thyroid_ultrasound_common_256",
        "selection_note": "Several TN3K/DDTI/TG3K thyroid-ultrasound papers use 256x256 preprocessing.",
    },
    "tg3k": {
        "hw": [256, 256],
        "selection_policy": "literature_backed",
        "selection_basis": "thyroid_ultrasound_common_256",
        "selection_note": "Several TN3K/DDTI/TG3K thyroid-ultrasound papers use 256x256 preprocessing.",
    },
    "ddti": {
        "hw": [256, 256],
        "selection_policy": "literature_backed",
        "selection_basis": "thyroid_ultrasound_common_256",
        "selection_note": "Several TN3K/DDTI/TG3K thyroid-ultrasound papers use 256x256 preprocessing.",
    },
    "otu_2d": {
        "hw": [256, 256],
        "selection_policy": "literature_backed_partial",
        "selection_basis": "ovarian_ultrasound_common_256",
        "selection_note": "Ovarian-tumor ultrasound segmentation/inpainting works frequently report 256x256 settings; use 256x256 as the stable baseline default.",
    },
    "monuseg": {
        "hw": [512, 512],
        "selection_policy": "literature_backed",
        "selection_basis": "monuseg_common_512",
        "selection_note": "Multiple nuclei / microscopy segmentation works resize MoNuSeg to 512x512.",
    },
    "ph2": {
        "hw": [256, 256],
        "selection_policy": "literature_backed_but_not_unique",
        "selection_basis": "dermoscopy_common_256",
        "selection_note": "PH2/skin-lesion segmentation papers often use 256x256 for efficient training, though 512x512 is also reported as an accuracy-first option when compute allows.",
    },

    # Evidence-informed project choices for abdominal CT / MRI.
    "btcv": {
        "hw": [512, 512],
        "selection_policy": "project_evidence_informed_override",
        "selection_basis": "abdominal_ct_highres_512",
        "selection_note": "Set to 512x512 because BTCV native slices are typically 512x512 and higher-resolution abdominal-CT settings are commonly favorable; this is a project choice rather than a unique canonical BTCV rule.",
    },
    "synapse": {
        "hw": [512, 512],
        "selection_policy": "project_evidence_informed_override",
        "selection_basis": "abdominal_ct_highres_512",
        "selection_note": "Set to 512x512 because Synapse abdominal CT often benefits from larger in-plane resolution in 2D training; this is a project choice rather than a unique canonical Synapse rule.",
    },
    "acdc": {
        "hw": [320, 320],
        "selection_policy": "modality_informed_default",
        "selection_basis": "cardiac_mri_typical_shape_320",
        "selection_note": "ACDC images are commonly around 320x320 in-plane; use 320x320 as the fixed student space.",
    },
    "prostate158": {
        "hw": [320, 320],
        "selection_policy": "modality_informed_default",
        "selection_basis": "prostate_mri_common_320",
        "selection_note": "Use 320x320 as a stable prostate-MRI student space default; override via JSON if a different fixed crop is preferred for your project.",
    },
}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def open_2d_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path))


def normalize_seg_label(arr: np.ndarray, dataset_name: Optional[str] = None) -> np.ndarray:
    """Normalize a 2D segmentation label according to the fixed baseline rule.

    - tn3k / tg3k / kvasirseg: JPEG or grayscale mask source, threshold at >=128.
    - cvc_clinicdb / ddti / otu_2d / ph2 and other binary masks: preserve binary semantics.
    """
    x = np.asarray(arr)
    if x.ndim == 3:
        x = x[..., 0]
    ds_key = infer_dataset_key(dataset_name or "")

    if ds_key in GRAY_MASK_THRESHOLD_DATASETS:
        return (x.astype(np.float32) >= float(GRAY_MASK_THRESHOLD)).astype(np.uint8)

    x = x.astype(np.uint8)
    uniq = set(int(v) for v in np.unique(x).tolist())
    if uniq.issubset({0, 1}):
        return x.astype(np.uint8)
    if uniq.issubset({0, 255}):
        return (x > 0).astype(np.uint8)

    # Non-canonical but still binary-like source. Keep the previous behavior only
    # for datasets not declared as JPEG/grayscale sources in the baseline contract.
    return (x > 0).astype(np.uint8)


def load_2d_label(path: Path, dataset_name: Optional[str] = None) -> np.ndarray:
    return normalize_seg_label(np.array(Image.open(path)), dataset_name=dataset_name)


def resolve_tiny_gt_filter(dataset_name: str, override_map: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return dataset-specific tiny-component filtering settings.

    Default behavior is conservative: only datasets with known systematic tiny false-GT
    artifacts are enabled. Use --tiny_gt_filter_json to override or enable other datasets.
    """
    ds_key = infer_dataset_key(dataset_name)
    spec = dict(DEFAULT_TINY_GT_FILTER_PRESETS.get(ds_key, {
        "enabled": False,
        "min_area": 0,
        "max_bbox_side": 0,
        "connectivity": 8,
        "policy": "disabled_by_default",
        "note": "No tiny-GT artifact filter is enabled for this dataset by default.",
    }))

    if override_map and ds_key in override_map:
        user_spec = override_map[ds_key]
        if isinstance(user_spec, bool):
            spec["enabled"] = bool(user_spec)
        elif isinstance(user_spec, (int, float)):
            spec["enabled"] = True
            spec["min_area"] = int(user_spec)
        elif isinstance(user_spec, dict):
            spec.update(user_spec)
        else:
            raise ValueError(f"Invalid tiny_gt_filter override for {dataset_name}: {user_spec}")

    spec["enabled"] = bool(spec.get("enabled", False))
    spec["min_area"] = int(spec.get("min_area", 0))
    spec["max_bbox_side"] = int(spec.get("max_bbox_side", 0))
    spec["connectivity"] = int(spec.get("connectivity", 8))
    return spec


def remove_tiny_false_gt_components(
    gt: np.ndarray,
    filter_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Remove isolated tiny binary-GT components before resizing.

    This is for false annotation dots / JPEG residuals, not for general denoising.
    It deliberately skips multi-class masks and never removes all foreground.
    """
    x = np.asarray(gt)
    if x.ndim == 3:
        x = x[..., 0]
    x = x.astype(np.uint8)

    info: Dict[str, Any] = {
        "enabled": bool(filter_cfg.get("enabled", False)),
        "applied": False,
        "reason": "",
        "min_area": int(filter_cfg.get("min_area", 0)),
        "max_bbox_side": int(filter_cfg.get("max_bbox_side", 0)),
        "num_components_before": 0,
        "num_components_after": 0,
        "num_removed_components": 0,
        "removed_area": 0,
        "removed_components": [],
    }

    if not info["enabled"]:
        info["reason"] = "disabled"
        return x, info

    uniq = set(int(v) for v in np.unique(x).tolist())
    if not uniq.issubset({0, 1}):
        info["reason"] = f"skip_non_binary_labels={sorted(uniq)[:20]}"
        return x, info

    mask = (x > 0).astype(np.uint8)
    if mask.max() == 0:
        info["reason"] = "empty_gt"
        return x, info

    connectivity = 8 if int(filter_cfg.get("connectivity", 8)) == 8 else 4
    num, cc, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=connectivity)
    components = []
    for cid in range(1, num):
        left = int(stats[cid, cv2.CC_STAT_LEFT])
        top = int(stats[cid, cv2.CC_STAT_TOP])
        width = int(stats[cid, cv2.CC_STAT_WIDTH])
        height = int(stats[cid, cv2.CC_STAT_HEIGHT])
        area = int(stats[cid, cv2.CC_STAT_AREA])
        components.append({
            "component_id": int(cid),
            "area": area,
            "bbox_xywh": [left, top, width, height],
        })

    info["num_components_before"] = len(components)
    if len(components) <= 1:
        info["reason"] = "single_component_preserved"
        info["num_components_after"] = len(components)
        return x, info

    min_area = int(filter_cfg.get("min_area", 0))
    max_bbox_side = int(filter_cfg.get("max_bbox_side", 0))

    def is_tiny(comp: Dict[str, Any]) -> bool:
        area = int(comp["area"])
        _, _, bw, bh = comp["bbox_xywh"]
        by_area = (min_area > 0 and area <= min_area)
        by_box = (max_bbox_side > 0 and max(int(bw), int(bh)) <= max_bbox_side)
        return bool(by_area or by_box)

    tiny_ids = [c["component_id"] for c in components if is_tiny(c)]
    keep_ids = [c["component_id"] for c in components if not is_tiny(c)]

    if len(tiny_ids) == 0:
        info["reason"] = "no_tiny_components"
        info["num_components_after"] = len(components)
        return x, info

    if len(keep_ids) == 0:
        info["reason"] = "all_components_tiny_preserved_to_avoid_empty_gt"
        info["num_components_after"] = len(components)
        return x, info

    out = x.copy()
    removed_components = []
    removed_area = 0
    for comp in components:
        cid = comp["component_id"]
        if cid not in tiny_ids:
            continue
        comp_mask = (cc == cid)
        removed_area += int(comp_mask.sum())
        out[comp_mask] = 0
        removed_components.append(comp)

    num_after, _, _, _ = cv2.connectedComponentsWithStats((out > 0).astype(np.uint8), connectivity=connectivity)
    info.update({
        "applied": True,
        "reason": "removed_tiny_components_with_large_component_present",
        "num_components_after": int(max(0, num_after - 1)),
        "num_removed_components": int(len(removed_components)),
        "removed_area": int(removed_area),
        "removed_components": removed_components,
    })
    return out.astype(np.uint8), info


def load_binary_aux_mask(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return (arr > 0).astype(np.uint8)


def load_nifti_pair(image_path: Path, label_path: Optional[Path]) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float, float]]:
    img_nii = nib.load(str(image_path))
    img = img_nii.get_fdata()
    spacing = tuple(float(x) for x in img_nii.header.get_zooms()[:3])
    gt = None
    if label_path is not None and label_path.exists():
        gt = nib.load(str(label_path)).get_fdata().astype(np.uint8)
    return img, gt, spacing


def try_npz_extract(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    obj = np.load(npz_path, allow_pickle=True)
    keys = set(obj.keys())
    img = None
    gt = None
    for k in ["imgs", "img", "image", "images"]:
        if k in keys:
            img = obj[k]
            break
    for k in ["gts", "gt", "label", "labels", "mask"]:
        if k in keys:
            gt = obj[k]
            break
    if img is None or gt is None:
        raise ValueError(f"Cannot infer image/label keys from npz: {npz_path}, keys={sorted(keys)}")
    return np.asarray(img), np.asarray(gt)


def try_h5_extract(h5_path: Path) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[float, float, float]]:
    image = None
    label = None
    spacing = (1.0, 1.0, 1.0)
    with h5py.File(str(h5_path), "r") as f:
        keys = list(f.keys())
        for k in ["image", "img", "imgs", "data"]:
            if k in f:
                image = np.asarray(f[k])
                break
        for k in ["label", "labels", "gt", "gts", "mask"]:
            if k in f:
                label = np.asarray(f[k])
                break
        for k in ["spacing", "spacings", "pixdim"]:
            if k in f:
                arr = np.asarray(f[k]).reshape(-1)
                if len(arr) >= 3:
                    spacing = (float(arr[0]), float(arr[1]), float(arr[2]))
                break
        if image is None:
            for k in keys:
                arr = np.asarray(f[k])
                if arr.ndim == 3:
                    image = arr
                    break
        if image is None:
            raise ValueError(f"Cannot infer image dataset from h5: {h5_path}, keys={keys}")
    return np.asarray(image), (None if label is None else np.asarray(label).astype(np.uint8)), spacing


def preprocess_ct_volume(img: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    img = np.clip(img, clip_min, clip_max)
    denom = max(clip_max - clip_min, 1e-8)
    img = (img - clip_min) / denom
    return img.astype(np.float32)


def preprocess_mri_volume(img: np.ndarray, lower_pct: float, upper_pct: float) -> np.ndarray:
    out = img.astype(np.float32)
    positive = out[out > 0]
    if positive.size == 0:
        positive = out.reshape(-1)
    lo = np.percentile(positive, lower_pct)
    hi = np.percentile(positive, upper_pct)
    out = np.clip(out, lo, hi)
    denom = max(hi - lo, 1e-8)
    out = (out - lo) / denom
    out[img == 0] = 0
    return out.astype(np.float32)


def preprocess_2d_image(arr: np.ndarray, do_intensity_cutoff: bool = False) -> np.ndarray:
    x = np.asarray(arr)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=-1)
    elif x.ndim == 3 and x.shape[-1] > 3:
        x = x[..., :3]
    x = x.astype(np.float32)
    if x.max() > 255.0:
        lo, hi = float(x.min()), float(x.max())
        if hi > lo:
            x = (x - lo) / (hi - lo) * 255.0
        x = np.clip(x, 0, 255)
    if do_intensity_cutoff:
        positive = x[x > 0]
        if positive.size > 0:
            lo = np.percentile(positive, 0.5)
            hi = np.percentile(positive, 99.5)
            x = np.clip(x, lo, hi)
            denom = max(hi - lo, 1e-8)
            x = (x - lo) / denom * 255.0
    x = np.clip(x / 255.0, 0, 1).astype(np.float32)
    return x


def ensure_float3c_slice(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = np.repeat(x[..., None], 3, axis=-1)
    elif x.ndim == 3 and x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    elif x.ndim == 3 and x.shape[-1] > 3:
        x = x[..., :3]
    x = x.astype(np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0, 1).astype(np.float32)


def resize_array_to_canvas(
    arr: np.ndarray,
    target_hw: Tuple[int, int],
    interpolation: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    h, w = int(arr.shape[0]), int(arr.shape[1])
    scale = min(target_h / max(h, 1), target_w / max(w, 1))
    new_h = max(int(round(h * scale)), 1)
    new_w = max(int(round(w * scale)), 1)

    resized = cv2.resize(arr, (new_w, new_h), interpolation=interpolation)
    if arr.ndim == 2:
        canvas = np.zeros((target_h, target_w), dtype=resized.dtype)
    else:
        canvas = np.zeros((target_h, target_w, arr.shape[2]), dtype=resized.dtype)

    offset_y = 0
    offset_x = 0
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    geom = {
        "orig_h": h,
        "orig_w": w,
        "target_h": target_h,
        "target_w": target_w,
        "new_h": new_h,
        "new_w": new_w,
        "scale": float(scale),
        "scale_x": float(scale),
        "scale_y": float(scale),
        "offset_x": int(offset_x),
        "offset_y": int(offset_y),
        "pad_right": int(target_w - new_w - offset_x),
        "pad_bottom": int(target_h - new_h - offset_y),
        "canvas_origin": "top_left",
        "interpolation": int(interpolation),
    }
    return canvas, geom


def resize_pair_to_canvas(
    img: np.ndarray,
    gt: np.ndarray,
    target_hw: Tuple[int, int],
    img_interp: int = cv2.INTER_CUBIC,
    gt_interp: int = cv2.INTER_NEAREST,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    img_canvas, img_geom = resize_array_to_canvas(img, target_hw, img_interp)
    gt_canvas, gt_geom = resize_array_to_canvas(gt.astype(np.uint8), target_hw, gt_interp)

    # The geometric transform must be shared. Because both are resized from the
    # same native shape to the same target with identical scale rule, img_geom and
    # gt_geom should match on the spatial fields.
    for key in ["orig_h", "orig_w", "target_h", "target_w", "new_h", "new_w", "scale", "scale_x", "scale_y", "offset_x", "offset_y", "pad_right", "pad_bottom", "canvas_origin"]:
        if img_geom[key] != gt_geom[key]:
            raise ValueError(f"Geometry mismatch for paired resize on key={key}: img={img_geom[key]} gt={gt_geom[key]}")
    geom = dict(img_geom)
    geom["img_interpolation"] = int(img_interp)
    geom["mask_interpolation"] = int(gt_interp)
    return img_canvas, gt_canvas.astype(np.uint8), geom


def build_inverse_geometry(forward_geom: Dict[str, Any]) -> Dict[str, Any]:
    orig_h = int(forward_geom["orig_h"])
    orig_w = int(forward_geom["orig_w"])
    new_h = int(forward_geom["new_h"])
    new_w = int(forward_geom["new_w"])
    target_h = int(forward_geom["target_h"])
    target_w = int(forward_geom["target_w"])
    offset_x = int(forward_geom["offset_x"])
    offset_y = int(forward_geom["offset_y"])

    return {
        "from_h": target_h,
        "from_w": target_w,
        "crop_h": new_h,
        "crop_w": new_w,
        "offset_x": offset_x,
        "offset_y": offset_y,
        "to_h": orig_h,
        "to_w": orig_w,
        "scale_x": float(orig_w / max(new_w, 1)),
        "scale_y": float(orig_h / max(new_h, 1)),
        "canvas_origin": str(forward_geom["canvas_origin"]),
        "interpolation_image": int(cv2.INTER_CUBIC),
        "interpolation_mask": int(cv2.INTER_NEAREST),
    }


def infer_dataset_key(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")

def make_unique_sample_name(split: str, case_id: str, slice_idx: Optional[int] = None, ext: str = ".npy") -> str:
    split = str(split).strip().lower()
    case_id = str(case_id).strip()

    # 防御性清洗，避免路径或空格问题
    case_id = case_id.replace("/", "_").replace("\\", "_").replace(" ", "_")

    if slice_idx is None:
        return f"{split}__{case_id}{ext}"

    return f"{split}__{case_id}__z{int(slice_idx):03d}{ext}"

def load_split_meta(organized_ds_root: Path) -> Dict[str, Any]:
    meta_path = organized_ds_root / "meta" / "split_meta.json"
    return load_json(meta_path) if meta_path.exists() else {}


def load_index(split_root: Path) -> List[Dict[str, Any]]:
    idx = split_root / "index.json"
    return load_json(idx) if idx.exists() else []


def load_case_meta(case_dir: Path) -> Dict[str, Any]:
    return load_json(case_dir / "case_meta.json")


def make_leakage_audit(train_case_records: List[Dict[str, Any]], test_case_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    train_patients = {str(x["patient_id"]) for x in train_case_records}
    test_patients = {str(x["patient_id"]) for x in test_case_records}
    overlap = sorted(train_patients & test_patients)
    train_cases = {str(x["case_id"]) for x in train_case_records}
    test_cases = {str(x["case_id"]) for x in test_case_records}
    case_overlap = sorted(train_cases & test_cases)
    return {
        "train_patients": len(train_patients),
        "test_patients": len(test_patients),
        "patient_overlap_count": len(overlap),
        "patient_overlap_examples": overlap[:20],
        "train_cases": len(train_cases),
        "test_cases": len(test_cases),
        "case_overlap_count": len(case_overlap),
        "case_overlap_examples": case_overlap[:20],
        "is_clean": len(overlap) == 0 and len(case_overlap) == 0,
    }


def resolve_student_size(dataset_name: str, override_map: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    ds_key = infer_dataset_key(dataset_name)
    if override_map and ds_key in override_map:
        spec = override_map[ds_key]
        if isinstance(spec, dict):
            hw = spec.get("hw") or [spec.get("h"), spec.get("w")]
            policy = spec.get("selection_policy", "user_override")
            basis = spec.get("selection_basis", "user_override")
            note = spec.get("selection_note", "User-specified student size override.")
        else:
            hw = spec
            policy = "user_override"
            basis = "user_override"
            note = "User-specified student size override."
        if not isinstance(hw, (list, tuple)) or len(hw) != 2:
            raise ValueError(f"Invalid override student size for {dataset_name}: {spec}")
        return {
            "hw": [int(hw[0]), int(hw[1])],
            "selection_policy": str(policy),
            "selection_basis": str(basis),
            "selection_note": str(note),
        }

    if ds_key not in DEFAULT_STUDENT_SIZE_PRESETS:
        raise KeyError(
            f"No student size preset for dataset={dataset_name}. "
            f"Please provide --student_size_json with an explicit mapping."
        )
    spec = DEFAULT_STUDENT_SIZE_PRESETS[ds_key]
    return {
        "hw": [int(spec["hw"][0]), int(spec["hw"][1])],
        "selection_policy": str(spec["selection_policy"]),
        "selection_basis": str(spec["selection_basis"]),
        "selection_note": str(spec["selection_note"]),
    }


class ProcessContext:
    def __init__(
        self,
        dataset_name: str,
        processed_fold_root: Path,
        teacher_size: int,
        student_hw: Tuple[int, int],
        student_size_meta: Dict[str, Any],
        tiny_gt_filter_meta: Optional[Dict[str, Any]] = None,
    ):
        self.dataset_name = dataset_name
        self.fold_root = processed_fold_root
        self.meta_dir = processed_fold_root / "meta"
        self.native_gt_dir = processed_fold_root / "native_npy" / "gts"
        self.teacher_img_dir = processed_fold_root / "teacher_npy" / "imgs"
        self.teacher_gt_dir = processed_fold_root / "teacher_npy" / "gts"
        self.student_img_dir = processed_fold_root / "student_npy" / "imgs"
        self.student_gt_dir = processed_fold_root / "student_npy" / "gts"
        self.student_aux_dir = processed_fold_root / "student_npy" / "aux"
        self.student_fov_dir = self.student_aux_dir / "fov_mask"
        self.student_roi_dir = self.student_aux_dir / "roi"
        self.prompts_dir = processed_fold_root / "prompts"
        self.teacher_size = int(teacher_size)
        self.student_target_h = int(student_hw[0])
        self.student_target_w = int(student_hw[1])
        self.student_size_meta = dict(student_size_meta)
        self.tiny_gt_filter_meta = dict(tiny_gt_filter_meta or {})

        for p in [
            self.meta_dir,
            self.native_gt_dir,
            self.teacher_img_dir,
            self.teacher_gt_dir,
            self.student_img_dir,
            self.student_gt_dir,
            self.prompts_dir,
            processed_fold_root / "pseudo_teacher",
            processed_fold_root / "pseudo_student",
        ]:
            ensure_dir(p)

        self.manifest: List[Dict[str, Any]] = []
        self.geometry_meta: Dict[str, Dict[str, Any]] = {}
        self.case_spacing: Dict[str, List[float]] = {}
        self.label_slice_count = defaultdict(int)
        self.label_case_set = defaultdict(set)
        self.aux_meta: Dict[str, Dict[str, str]] = {}
        self.stats = {
            "num_samples_total": 0,
            "num_train_samples": 0,
            "num_test_samples": 0,
            "num_empty_slices": 0,
            "num_nonempty_slices": 0,
            "num_samples_with_tiny_gt_removed": 0,
            "num_tiny_gt_components_removed": 0,
            "num_tiny_gt_pixels_removed": 0,
        }
        self.tiny_gt_filter_records: List[Dict[str, Any]] = []
        self.train_case_records: List[Dict[str, str]] = []
        self.test_case_records: List[Dict[str, str]] = []

    def register_case_record(self, split: str, case_id: str, patient_id: str):
        rec = {"case_id": case_id, "patient_id": patient_id}
        if split == "train":
            self.train_case_records.append(rec)
        else:
            self.test_case_records.append(rec)

    def register_tiny_gt_filter(
        self,
        sample_name: str,
        case_id: str,
        split: str,
        filter_info: Dict[str, Any],
    ):
        if not filter_info.get("applied", False):
            return
        rec = {
            "sample_name": sample_name,
            "case_id": case_id,
            "split": split,
            **filter_info,
        }
        self.tiny_gt_filter_records.append(rec)
        self.stats["num_samples_with_tiny_gt_removed"] += 1
        self.stats["num_tiny_gt_components_removed"] += int(filter_info.get("num_removed_components", 0))
        self.stats["num_tiny_gt_pixels_removed"] += int(filter_info.get("removed_area", 0))

    def update_label_meta(self, gt_student: np.ndarray, case_id: str):
        uniq = [int(x) for x in np.unique(gt_student) if int(x) > 0]
        for lb in uniq:
            self.label_slice_count[lb] += 1
            self.label_case_set[lb].add(case_id)

    def _resize_aux_mask_to_student(self, mask_native: np.ndarray) -> np.ndarray:
        canvas, _ = resize_array_to_canvas(mask_native.astype(np.uint8), (self.student_target_h, self.student_target_w), cv2.INTER_NEAREST)
        return canvas.astype(np.uint8)

    def add_sample(
        self,
        sample_name: str,
        case_id: str,
        patient_id: str,
        split: str,
        slice_idx: Optional[int],
        img_native_3c: np.ndarray,
        gt_native: np.ndarray,
        spacing_xyz: Tuple[float, float, float],
        aux_native: Optional[Dict[str, np.ndarray]] = None,
    ):
        if aux_native is None:
            aux_native = {}

        img_native_3c = ensure_float3c_slice(img_native_3c)
        gt_native = gt_native.astype(np.uint8)

        img_student, gt_student, geom_student = resize_pair_to_canvas(
            img_native_3c,
            gt_native,
            (self.student_target_h, self.student_target_w),
            img_interp=cv2.INTER_CUBIC,
            gt_interp=cv2.INTER_NEAREST,
        )
        img_student = np.clip(img_student, 0, 1).astype(np.float32)
        gt_student = gt_student.astype(np.uint8)

        img_teacher, gt_teacher, geom_teacher = resize_pair_to_canvas(
            img_native_3c,
            gt_native,
            (self.teacher_size, self.teacher_size),
            img_interp=cv2.INTER_CUBIC,
            gt_interp=cv2.INTER_NEAREST,
        )
        img_teacher = np.clip(img_teacher, 0, 1).astype(np.float32)
        gt_teacher = gt_teacher.astype(np.uint8)

        np.save(self.native_gt_dir / sample_name, gt_native.astype(np.uint8))
        np.save(self.student_img_dir / sample_name, img_student)
        np.save(self.student_gt_dir / sample_name, gt_student)
        np.save(self.teacher_img_dir / sample_name, img_teacher)
        np.save(self.teacher_gt_dir / sample_name, gt_teacher)

        aux_entry: Dict[str, str] = {}
        if "fov_mask" in aux_native and aux_native["fov_mask"] is not None:
            ensure_dir(self.student_fov_dir)
            fov = self._resize_aux_mask_to_student(aux_native["fov_mask"])
            np.save(self.student_fov_dir / sample_name, fov)
            aux_entry["fov_mask"] = f"student_npy/aux/fov_mask/{sample_name}"
        if "roi" in aux_native and aux_native["roi"] is not None:
            ensure_dir(self.student_roi_dir)
            roi = self._resize_aux_mask_to_student(aux_native["roi"])
            np.save(self.student_roi_dir / sample_name, roi)
            aux_entry["roi"] = f"student_npy/aux/roi/{sample_name}"
        self.aux_meta[sample_name] = aux_entry

        teacher_to_native = build_inverse_geometry(geom_teacher)
        student_to_native = build_inverse_geometry(geom_student)

        self.geometry_meta[sample_name] = {
            "case_id": case_id,
            "patient_id": patient_id,
            "slice_idx": None if slice_idx is None else int(slice_idx),
            "native_hw": [int(gt_native.shape[0]), int(gt_native.shape[1])],
            "teacher_hw": [int(self.teacher_size), int(self.teacher_size)],
            "student_hw": [int(self.student_target_h), int(self.student_target_w)],
            "spacing": [float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])],
            "spacing_xyz": [float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])],
            "slice_axis": None if slice_idx is None else "last",
            "slice_rotation": None if slice_idx is None else {"type": "np.rot90", "k": 1},
            "native_to_teacher": geom_teacher,
            "teacher_to_native": teacher_to_native,
            "native_to_student": geom_student,
            "student_to_native": student_to_native,
            # backward-compatible shortcuts for old generate_pseudo_labels.py
            "orig_h": int(geom_teacher["orig_h"]),
            "orig_w": int(geom_teacher["orig_w"]),
            "new_h": int(geom_teacher["new_h"]),
            "new_w": int(geom_teacher["new_w"]),
            "teacher": geom_teacher,
            "student": geom_student,
        }

        self.case_spacing[case_id] = [float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])]

        entry = {
            "dataset_name": self.dataset_name,
            "fold": "fold_0",
            "split": split,
            "case_id": case_id,
            "patient_id": patient_id,
            "slice_name": sample_name,
            "slice_idx": None if slice_idx is None else int(slice_idx),
            "orig_h": int(gt_native.shape[0]),
            "orig_w": int(gt_native.shape[1]),
            "spacing": [float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])],
            "spacing_xyz": [float(spacing_xyz[0]), float(spacing_xyz[1]), float(spacing_xyz[2])],
            "geometry_key": sample_name,
            "native_gt": f"native_npy/gts/{sample_name}",
            "teacher_img": f"teacher_npy/imgs/{sample_name}",
            "teacher_gt": f"teacher_npy/gts/{sample_name}",
            "student_img": f"student_npy/imgs/{sample_name}",
            "student_gt": f"student_npy/gts/{sample_name}",
            "student_aux": aux_entry,
            "has_fg": bool(np.any(gt_native > 0)),
        }
        self.manifest.append(entry)
        self.update_label_meta(gt_student, case_id)

        self.stats["num_samples_total"] += 1
        if split == "train":
            self.stats["num_train_samples"] += 1
        else:
            self.stats["num_test_samples"] += 1
        if np.any(gt_native > 0):
            self.stats["num_nonempty_slices"] += 1
        else:
            self.stats["num_empty_slices"] += 1

    def finalize(self, organized_split_meta: Dict[str, Any]):
        leakage = make_leakage_audit(self.train_case_records, self.test_case_records)
        label_meta = {
            "unique_labels": sorted([int(k) for k in self.label_slice_count.keys()]),
            "num_slices_per_label": {str(int(k)): int(v) for k, v in self.label_slice_count.items()},
            "num_cases_per_label": {str(int(k)): int(len(v)) for k, v in self.label_case_set.items()},
        }
        source_split_protocol = organized_split_meta.get("split_protocol", "unspecified") if isinstance(organized_split_meta, dict) else "unspecified"
        split_meta = {
            "dataset_name": self.dataset_name,
            "fold": "fold_0",
            "split_protocol": source_split_protocol,
            "teacher_target_size": self.teacher_size,
            "student_target_h": self.student_target_h,
            "student_target_w": self.student_target_w,
            "student_size_selection_policy": self.student_size_meta.get("selection_policy", "unknown"),
            "student_size_selection_basis": self.student_size_meta.get("selection_basis", "unknown"),
            "student_size_selection_note": self.student_size_meta.get("selection_note", ""),
            "label_binarization_rule": (
                ">=128_for_jpeg_gray_mask" if infer_dataset_key(self.dataset_name) in GRAY_MASK_THRESHOLD_DATASETS
                else "preserve_binary_or_threshold_noncanonical_to_foreground"
            ),
            "label_binarization_threshold": (
                int(GRAY_MASK_THRESHOLD) if infer_dataset_key(self.dataset_name) in GRAY_MASK_THRESHOLD_DATASETS else None
            ),
            "tiny_gt_filter": self.tiny_gt_filter_meta,
            "multi_class_preserved": bool(len(label_meta["unique_labels"]) > 1),
            "has_fov_mask": any("fov_mask" in v for v in self.aux_meta.values()),
            "has_roi": any("roi" in v for v in self.aux_meta.values()),
            "source_split_meta": organized_split_meta,
            "train_cases": sorted({x["case_id"] for x in self.train_case_records}),
            "test_cases": sorted({x["case_id"] for x in self.test_case_records}),
        }
        save_json(self.manifest, self.meta_dir / "manifest.json")
        save_json(self.geometry_meta, self.meta_dir / "geometry_meta.json")
        save_json(leakage, self.meta_dir / "leakage_audit.json")
        save_json(self.case_spacing, self.meta_dir / "case_spacing.json")
        save_json(label_meta, self.meta_dir / "label_meta.json")
        save_json(self.aux_meta, self.meta_dir / "aux_meta.json")
        save_json(self.tiny_gt_filter_records, self.meta_dir / "tiny_gt_filter_records.json")
        save_json(self.stats, self.meta_dir / "preprocess_stats.json")
        save_json(split_meta, self.meta_dir / "split_meta.json")


def process_2d_case(ctx: ProcessContext, case_dir: Path, case_meta: Dict[str, Any], split: str, ds_cfg: Dict[str, Any]):
    case_id = case_meta["case_id"]
    patient_id = case_meta["patient_id"]
    image_file = case_meta.get("image_file")
    label_file = case_meta.get("label_file")
    if image_file is None or label_file is None:
        raise ValueError(f"2D case missing image_file/label_file: {case_id}")
    image_path = case_dir / image_file
    label_path = case_dir / label_file
    img_raw = open_2d_image(image_path)
    gt_raw = load_2d_label(label_path, dataset_name=ctx.dataset_name)
    img_native = preprocess_2d_image(img_raw, do_intensity_cutoff=bool(ds_cfg.get("gray_cutoff", False)))

    gt_native = gt_raw.astype(np.uint8)
    if split == "train":
        gt_native, tiny_filter_info = remove_tiny_false_gt_components(
            gt_native,
            ds_cfg.get("tiny_gt_filter", {}),
        )
    else:
        # Highest-standard evaluation protocol:
        # testing annotations are only geometrically transformed for evaluation
        # alignment and are never semantically cleaned.
        tiny_filter_cfg = ds_cfg.get("tiny_gt_filter", {})
        tiny_filter_info = {
            "enabled": bool(tiny_filter_cfg.get("enabled", False)),
            "applied": False,
            "reason": "disabled_for_test_split_eval_gt_preserved",
            "min_area": int(tiny_filter_cfg.get("min_area", 0)),
            "max_bbox_side": int(tiny_filter_cfg.get("max_bbox_side", 0)),
            "num_components_before": 0,
            "num_components_after": 0,
            "num_removed_components": 0,
            "removed_area": 0,
            "removed_components": [],
        }
    aux_files = case_meta.get("aux_files", {})
    aux_native: Dict[str, np.ndarray] = {}
    if "fov_mask" in aux_files:
        fov_path = case_dir / aux_files["fov_mask"]
        if fov_path.exists():
            aux_native["fov_mask"] = load_binary_aux_mask(fov_path)
    if "roi" in aux_files:
        roi_path = case_dir / aux_files["roi"]
        if roi_path.exists():
            aux_native["roi"] = load_binary_aux_mask(roi_path)
    sample_name = make_unique_sample_name(split, case_id)
    ctx.register_tiny_gt_filter(sample_name, case_id, split, tiny_filter_info)
    ctx.add_sample(sample_name, case_id, patient_id, split, None, img_native, gt_native, (1.0, 1.0, 1.0), aux_native)


def process_3d_case(ctx: ProcessContext, case_dir: Path, case_meta: Dict[str, Any], split: str, ds_cfg: Dict[str, Any]):
    case_id = case_meta["case_id"]
    patient_id = case_meta["patient_id"]
    image_file = case_meta.get("image_file")
    label_file = case_meta.get("label_file")
    if image_file is None or label_file is None:
        raise ValueError(f"3D case missing image_file/label_file: {case_id}")
    image_path = case_dir / image_file
    label_path = case_dir / label_file
    img_vol, gt_vol, spacing_xyz = load_nifti_pair(image_path, label_path)
    if gt_vol is None:
        raise ValueError(f"3D case has no label volume: {case_id}")
    modality = ds_cfg["modality"].lower()
    if modality == "ct":
        clip_min, clip_max = ds_cfg.get("ct_clip", [-125, 275])
        img_vol_pre = preprocess_ct_volume(img_vol, clip_min, clip_max)
    else:
        lower_pct, upper_pct = ds_cfg.get("mri_pct", [0.5, 99.5])
        img_vol_pre = preprocess_mri_volume(img_vol, lower_pct, upper_pct)
    num_slices = img_vol_pre.shape[-1]
    for i in range(num_slices):
        img_slice = np.rot90(img_vol_pre[:, :, i])
        gt_slice = np.rot90(gt_vol[:, :, i]).astype(np.uint8)
        sample_name = make_unique_sample_name(split, case_id, i)
        ctx.add_sample(sample_name, case_id, patient_id, split, i, img_slice, gt_slice, spacing_xyz, None)


def process_synapse_case(ctx: ProcessContext, case_dir: Path, case_meta: Dict[str, Any], split: str, ds_cfg: Dict[str, Any]):
    case_id = case_meta["case_id"]
    patient_id = case_meta["patient_id"]
    source_format = case_meta.get("source_format", "")

    if source_format == "npz_slices":
        slice_dir_rel = case_meta.get("aux_files", {}).get("slices_dir", "aux/slices")
        slice_dir = case_dir / slice_dir_rel
        if not slice_dir.exists():
            raise FileNotFoundError(f"Synapse slice dir not found: {slice_dir}")

        npz_files = sorted(slice_dir.glob("*.npz"))
        for j, npz_path in enumerate(npz_files):
            img, gt = try_npz_extract(npz_path)
            img = np.asarray(img).astype(np.float32)
            gt = np.asarray(gt).astype(np.uint8)

            if img.ndim == 2:
                img_native = ensure_float3c_slice(img)
            elif img.ndim == 3:
                if img.shape[0] == 3 and img.shape[-1] != 3:
                    img = np.transpose(img, (1, 2, 0))
                img_native = ensure_float3c_slice(img)
            else:
                raise ValueError(f"Unexpected synapse npz image shape: {img.shape}, {npz_path}")

            if gt.ndim == 3:
                gt = np.squeeze(gt)

            sample_name = f"{case_id}_z{str(j).zfill(3)}.npy"
            ctx.add_sample(
                sample_name,
                case_id,
                patient_id,
                split,
                j,
                img_native,
                gt.astype(np.uint8),
                (1.0, 1.0, 1.0),
                None,
            )

    elif source_format == "h5_volume":
        image_file = case_meta.get("image_file")
        if image_file is None:
            raise ValueError(f"Synapse h5 test case missing image_file: {case_id}")

        h5_path = case_dir / image_file
        img_vol, gt_vol, spacing_xyz = try_h5_extract(h5_path)
        img_vol = np.asarray(img_vol).astype(np.float32)

        # test h5 可能已经是 0~1 归一化后的数据；只有像原始 CT/HU 时才做 CT 预处理
        if float(img_vol.min()) >= 0.0 and float(img_vol.max()) <= 1.5:
            img_vol_pre = img_vol.astype(np.float32)
        else:
            clip_min, clip_max = ds_cfg.get("ct_clip", [-125, 275])
            img_vol_pre = preprocess_ct_volume(img_vol, clip_min, clip_max)

        if gt_vol is None:
            raise ValueError(f"Synapse test h5 has no label: {h5_path}")
        if img_vol_pre.ndim != 3:
            raise ValueError(f"Unexpected synapse h5 volume shape: {img_vol_pre.shape}")

        for i in range(img_vol_pre.shape[0]):
            img_slice = img_vol_pre[i, :, :]
            gt_slice = gt_vol[i, :, :].astype(np.uint8)

            sample_name = f"{case_id}_z{str(i).zfill(3)}.npy"
            ctx.add_sample(
                sample_name,
                case_id,
                patient_id,
                split,
                i,
                img_slice,
                gt_slice,
                spacing_xyz,
                None,
            )

    else:
        raise ValueError(f"Unsupported synapse source_format for case {case_id}: {source_format}")


def process_dataset(
    organized_root: Path,
    processed_root: Path,
    dataset_name: str,
    teacher_size: int,
    student_size_overrides: Optional[Dict[str, Any]] = None,
    tiny_gt_filter_overrides: Optional[Dict[str, Any]] = None,
    progress_root: Optional[str] = None,
    progress_every: int = 20,
):
    ds_key = infer_dataset_key(dataset_name)
    if ds_key not in DATASET_CONFIG:
        raise KeyError(f"Unsupported dataset in processed_strict.py: {dataset_name}")
    ds_cfg = dict(DATASET_CONFIG[ds_key])
    ds_cfg["tiny_gt_filter"] = resolve_tiny_gt_filter(dataset_name, tiny_gt_filter_overrides)
    organized_ds_root = organized_root / dataset_name
    if not organized_ds_root.exists():
        candidates = {infer_dataset_key(p.name): p for p in organized_root.iterdir() if p.is_dir()}
        if ds_key in candidates:
            organized_ds_root = candidates[ds_key]
            dataset_name = organized_ds_root.name
        else:
            raise FileNotFoundError(f"organized dataset not found: {dataset_name}")

    student_size_meta = resolve_student_size(dataset_name, student_size_overrides)
    student_hw = tuple(student_size_meta["hw"])

    processed_fold_root = processed_root / dataset_name / "fold_0"
    stage_time_path = processed_fold_root / "meta" / "stage_time_preprocess.json"

    total_cases_for_progress = 0
    for _split in ["train", "test"]:
        try:
            total_cases_for_progress += len(load_index(organized_ds_root / _split))
        except Exception:
            pass
    progress_writer = ProgressWriter(
        progress_root,
        stage="preprocess",
        dataset=dataset_name,
        fold="fold_0",
        split="train_test",
        enabled=bool(progress_root),
    )
    progress_writer.start(total=total_cases_for_progress, message="preprocess started")
    progress_current = 0
    progress_every = max(1, int(progress_every or 1))

    with StageTimer(
        save_path=str(stage_time_path),
        stage_name="preprocess",
        dataset=dataset_name,
        fold="fold_0",
        split="train_test",
    ) as timer:
        ctx = ProcessContext(
            dataset_name,
            processed_fold_root,
            teacher_size,
            student_hw,
            student_size_meta,
            tiny_gt_filter_meta=ds_cfg.get("tiny_gt_filter", {}),
        )
        organized_split_meta = load_split_meta(organized_ds_root)

        for split in ["train", "test"]:
            split_root = organized_ds_root / split
            index = load_index(split_root)
            for row in index:
                case_dir = organized_ds_root / row["case_dir"]
                case_meta = load_case_meta(case_dir)
                ctx.register_case_record(split, case_meta["case_id"], case_meta["patient_id"])

                if ds_cfg["kind"] == "2d":
                    process_2d_case(ctx, case_dir, case_meta, split, ds_cfg)
                elif ds_cfg["kind"] == "3d":
                    process_3d_case(ctx, case_dir, case_meta, split, ds_cfg)
                elif ds_cfg["kind"] == "synapse":
                    process_synapse_case(ctx, case_dir, case_meta, split, ds_cfg)
                else:
                    raise ValueError(f"Unsupported dataset kind: {ds_cfg['kind']} for {dataset_name}")

                progress_current += 1
                if progress_current == 1 or progress_current % progress_every == 0 or progress_current == total_cases_for_progress:
                    progress_writer.update(
                        status="running",
                        current=progress_current,
                        total=total_cases_for_progress,
                        message=f"processed {split} case {progress_current}/{total_cases_for_progress}",
                        stats=dict(ctx.stats),
                    )

        ctx.finalize(organized_split_meta)
        timer.set_outputs(ctx.stats["num_samples_total"])

        print(
            f"[OK] preprocessed {dataset_name}: total_samples={ctx.stats['num_samples_total']} "
            f"train={ctx.stats['num_train_samples']} test={ctx.stats['num_test_samples']} "
            f"student_hw=({ctx.student_target_h},{ctx.student_target_w})"
        )
        progress_writer.finish(
            status="success",
            current=progress_current,
            total=total_cases_for_progress,
            message="preprocess finished",
            stats=dict(ctx.stats),
        )


def parse_datasets_arg(datasets_arg: str, organized_root: Path) -> List[str]:
    if datasets_arg.strip().lower() == "all":
        return sorted([p.name for p in organized_root.iterdir() if p.is_dir()])
    return [x.strip() for x in datasets_arg.split(",") if x.strip()]


def load_student_size_override_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None or len(path.strip()) == 0:
        return None
    override_path = Path(path)
    if not override_path.exists():
        raise FileNotFoundError(f"student size override json not found: {override_path}")
    obj = load_json(override_path)
    if not isinstance(obj, dict):
        raise ValueError("student size override json must be a dict")
    return {infer_dataset_key(k): v for k, v in obj.items()}


def load_tiny_gt_filter_override_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if path is None or len(path.strip()) == 0:
        return None
    override_path = Path(path)
    if not override_path.exists():
        raise FileNotFoundError(f"tiny GT filter override json not found: {override_path}")
    obj = load_json(override_path)
    if not isinstance(obj, dict):
        raise ValueError("tiny GT filter override json must be a dict")
    return {infer_dataset_key(k): v for k, v in obj.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--organized_root", type=str, required=True)
    parser.add_argument("--processed_root", type=str, required=True)
    parser.add_argument("--datasets", type=str, default="btcv,synapse,acdc,prostate158,kvasirseg,cvc_clinicdb,tn3k,tg3k,ddti,otu_2d,ph2")
    parser.add_argument("--teacher_size", type=int, default=DEFAULT_TEACHER_SIZE)
    parser.add_argument(
        "--student_size_json",
        type=str,
        default="",
        help="Optional JSON override. Example: {'btcv': [320,320], 'tn3k': {'hw':[512,512], 'selection_policy':'user_override'}}",
    )
    parser.add_argument(
        "--tiny_gt_filter_json",
        type=str,
        default="",
        help=(
            "Optional JSON override for native-space tiny false-GT filtering. "
            "Example: {'tg3k': {'enabled': true, 'min_area': 32, 'max_bbox_side': 8}, "
            "'kvasirseg': true, 'ph2': false}"
        ),
    )
    parser.add_argument(
        "--progress_root",
        type=str,
        default="",
        help="Optional progress output root. For md-aligned formal runs, prefer data/vis/progress instead of data/processed/_progress.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=20,
        help="Update progress every N processed cases/slices.",
    )
    args = parser.parse_args()

    organized_root = Path(args.organized_root)
    processed_root = Path(args.processed_root)
    ensure_dir(processed_root)
    datasets = parse_datasets_arg(args.datasets, organized_root)
    student_size_overrides = load_student_size_override_json(args.student_size_json)
    tiny_gt_filter_overrides = load_tiny_gt_filter_override_json(args.tiny_gt_filter_json)

    summary = []
    for ds in datasets:
        try:
            process_dataset(
                organized_root,
                processed_root,
                ds,
                args.teacher_size,
                student_size_overrides,
                tiny_gt_filter_overrides,
                progress_root=(args.progress_root.strip() if args.progress_root else ""),
                progress_every=int(args.progress_every),
            )
            summary.append({"dataset": ds, "status": "success"})
        except Exception as e:
            summary.append({"dataset": ds, "status": "failed", "error": repr(e)})
            print(f"[ERROR] {ds}: {repr(e)}")
            traceback.print_exc()
    save_json(summary, processed_root / "_processed_summary.json")
    print(f"[DONE] summary -> {processed_root / '_processed_summary.json'}")


if __name__ == "__main__":
    main()
