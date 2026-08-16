#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2 pseudo-label generation for the strict box-constrained baseline.

Legal tri-pseudo semantics:
- outside bbox union: 0
- inside bbox and confirmed by MedSAM as foreground: 1..K
- inside bbox but not confirmed by MedSAM as foreground: 255

Notes
-----
1) bbox is defined in native space and delivered in teacher space by the strict prompt builder.
2) This script uses teacher-space bbox for MedSAM inference, then restores the confirmed
   foreground to native/student space and reconstructs native tri-pseudo from native boxes.
3) only tri pseudo is part of the formal baseline.
4) no hard pseudo or weight_map is saved by this script.
"""

import os
import json
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm
from scipy import ndimage
from segment_anything import sam_model_registry, SamPredictor

from utils.stage_timer_utils import StageTimer

from methods.common.multiclass_resolver import (
    aggregate_same_class_evidence,
    resolve_multiclass_evidence,
    sigmoid_np,
)


MODEL_TYPE = "vit_b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DEFAULT_TARGET_DATASETS = ["btcv", "synapse", "acdc", "prostate158",
    "kvasirseg", "cvc_clinicdb", "tn3k", "tg3k",
    "ddti", "otu_2d", "monuseg", "ph2"]
MULTICLASS_DATASETS = {"btcv", "synapse", "acdc", "prostate158"}


@dataclass
class DatasetCfg:
    keep_largest_cc: bool = True
    enhancement: str = "none"
    score_area_bonus: float = 0.05
    spill_weight: float = 0.80


DATASET_CFG: Dict[str, DatasetCfg] = {
    "btcv": DatasetCfg(keep_largest_cc=False, enhancement="clahe", score_area_bonus=0.05, spill_weight=0.35),
    "synapse": DatasetCfg(keep_largest_cc=False, enhancement="clahe", score_area_bonus=0.05, spill_weight=0.35),
    "acdc": DatasetCfg(keep_largest_cc=False, enhancement="clahe_light", score_area_bonus=0.05, spill_weight=0.40),
    "prostate158": DatasetCfg(keep_largest_cc=False, enhancement="clahe_light", score_area_bonus=0.05, spill_weight=0.40),
    "cvc_clinicdb": DatasetCfg(keep_largest_cc=True, enhancement="none", score_area_bonus=0.10, spill_weight=0.80),
    "kvasirseg": DatasetCfg(keep_largest_cc=True, enhancement="clahe_light", score_area_bonus=0.10, spill_weight=0.80),
    "tn3k": DatasetCfg(keep_largest_cc=True, enhancement="clahe_light", score_area_bonus=0.10, spill_weight=0.80),
}


@dataclass
class PromptInstance:
    teacher_box: np.ndarray
    native_box: Optional[np.ndarray]
    label_id: int
    point_coords: Optional[np.ndarray] = None
    point_labels: Optional[np.ndarray] = None


@dataclass
class Proposal:
    label_id: int
    score: float
    best_mask: np.ndarray
    best_probability: np.ndarray
    best_model_score: float
    teacher_box: np.ndarray
    native_box: Optional[np.ndarray]
    point_coords: Optional[np.ndarray] = None
    point_labels: Optional[np.ndarray] = None


def parse_list(s: str) -> List[str]:
    s = (s or "").strip()
    return [x.strip() for x in s.split(",") if x.strip()]


def safe_suffix(tag: str) -> str:
    raw = (tag or "").strip()
    if not raw:
        return ""
    cleaned = "".join(ch if (ch.isalnum() or ch in {"_", "-"}) else "_" for ch in raw)
    return f"_{cleaned}"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log_error(log_file: str, msg: str):
    ensure_dir(os.path.dirname(log_file))
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(msg.rstrip() + "\n")


def infer_is_multiclass(dataset_name: str, fold_root: str) -> bool:
    ds_key = dataset_name.lower()
    split_meta_path = os.path.join(fold_root, "meta", "split_meta.json")
    if os.path.exists(split_meta_path):
        split_meta = load_json(split_meta_path)
        if "multi_class_preserved" in split_meta:
            return bool(split_meta["multi_class_preserved"])
    return ds_key in MULTICLASS_DATASETS


def get_dataset_cfg(dataset_name: str) -> DatasetCfg:
    return DATASET_CFG.get(dataset_name.lower(), DatasetCfg())


def load_manifest_index(fold_root: str) -> Dict[str, dict]:
    manifest_path = os.path.join(fold_root, "meta", "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")
    manifest = load_json(manifest_path)
    out = {}
    for item in manifest:
        slice_name = item["slice_name"]
        if slice_name in out:
            prev = out[slice_name]
            raise ValueError(
                f"Duplicate slice_name in manifest.json: {slice_name}. "
                f"Previous case/split=({prev['case_id']},{prev['split']}), "
                f"current case/split=({item['case_id']},{item['split']}). "
                f"Please regenerate processed data with globally unique sample names."
            )
        out[slice_name] = {
            "split": item["split"],
            "fold": item["fold"],
            "case_id": item["case_id"],
            "teacher_img": os.path.join(fold_root, item["teacher_img"]),
            "teacher_gt": os.path.join(fold_root, item["teacher_gt"]),
            "student_img": os.path.join(fold_root, item["student_img"]),
            "student_gt": os.path.join(fold_root, item["student_gt"]),
        }
    return out


def load_geometry_meta(fold_root: str) -> Dict[str, dict]:
    geom_path = os.path.join(fold_root, "meta", "geometry_meta.json")
    if not os.path.exists(geom_path):
        raise FileNotFoundError(f"geometry_meta.json not found: {geom_path}")
    return load_json(geom_path)


def load_teacher_target_size(fold_root: str) -> int:
    split_meta_path = os.path.join(fold_root, "meta", "split_meta.json")
    if not os.path.exists(split_meta_path):
        return 1024
    split_meta = load_json(split_meta_path)
    return int(split_meta.get("teacher_target_size", 1024))


def _to_uint8_rgb(image: np.ndarray, expected_hw: Tuple[int, int]) -> np.ndarray:
    if image.ndim == 4:
        image = np.squeeze(image)

    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=-1)
    elif image.ndim == 3:
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] > 3:
            image = image[:, :, :3]
    else:
        raise ValueError(f"Unexpected image ndim: {image.ndim}")

    if image.shape[:2] != expected_hw:
        raise ValueError(f"Teacher image shape mismatch: got {image.shape[:2]}, expect {expected_hw}")

    if image.max() <= 1.5:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    else:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return image


def _load_gt_2d(path: str) -> np.ndarray:
    gt = np.load(path)
    if gt.ndim == 3:
        gt = gt[:, :, 0]
    return gt.astype(np.uint8)


def enhance_image_by_dataset(image_rgb: np.ndarray, enhancement: str) -> np.ndarray:
    if enhancement == "none":
        return image_rgb

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    if enhancement == "clahe_light":
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    else:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)


def _clip_box_to_image(box: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32).tolist()
    x1 = max(0.0, min(float(w - 1), x1))
    x2 = max(0.0, min(float(w - 1), x2))
    y1 = max(0.0, min(float(h - 1), y1))
    y2 = max(0.0, min(float(h - 1), y2))

    if x2 <= x1:
        x2 = min(float(w - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(h - 1), y1 + 1.0)

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def _clip_points_to_image(point_coords: np.ndarray, w: int, h: int) -> np.ndarray:
    if point_coords.size == 0:
        return point_coords.reshape(0, 2).astype(np.float32)
    pts = point_coords.astype(np.float32).copy()
    pts[:, 0] = np.clip(pts[:, 0], 0.0, float(w - 1))
    pts[:, 1] = np.clip(pts[:, 1], 0.0, float(h - 1))
    return pts


def box_to_mask_xyxy(box: np.ndarray, h: int, w: int) -> np.ndarray:
    x1, y1, x2, y2 = box.astype(np.float32).tolist()
    x1 = int(np.floor(x1))
    y1 = int(np.floor(y1))
    x2 = int(np.ceil(x2))
    y2 = int(np.ceil(y2))

    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))

    m = np.zeros((h, w), dtype=bool)
    m[y1:y2 + 1, x1:x2 + 1] = True
    return m


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.uint8)
    if mask.max() == 0:
        return mask
    num, cc = cv2.connectedComponents(mask)
    if num <= 1:
        return mask

    best_area = 0
    best_id = None
    for i in range(1, num):
        area = int((cc == i).sum())
        if area > best_area:
            best_area = area
            best_id = i

    if best_id is None:
        return np.zeros_like(mask, dtype=np.uint8)
    return (cc == best_id).astype(np.uint8)


def refine_mask_by_prior(mask: np.ndarray, box_mask: np.ndarray, keep_largest_cc: bool, is_multiclass: bool) -> np.ndarray:
    mask = (mask > 0) & box_mask
    mask = mask.astype(np.uint8)

    if mask.max() == 0:
        return mask

    close_k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)
    mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    if not is_multiclass:
        open_k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k)
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)

    if keep_largest_cc:
        mask = keep_largest_component(mask)

    return ((mask > 0) & box_mask).astype(np.uint8)


def _get_geom_sizes(geom: dict) -> Tuple[int, int, int, int]:
    if "orig_h" in geom and "orig_w" in geom and "new_h" in geom and "new_w" in geom:
        return int(geom["orig_h"]), int(geom["orig_w"]), int(geom["new_h"]), int(geom["new_w"])

    if "native_hw" in geom and "teacher_to_native" in geom:
        orig_h, orig_w = int(geom["native_hw"][0]), int(geom["native_hw"][1])
        t2n = geom["teacher_to_native"]
        if isinstance(t2n, dict):
            new_h = int(t2n.get("crop_h", t2n.get("new_h", geom.get("teacher_hw", [1024, 1024])[0])))
            new_w = int(t2n.get("crop_w", t2n.get("new_w", geom.get("teacher_hw", [1024, 1024])[1])))
        else:
            raise ValueError("Unsupported teacher_to_native structure in geometry_meta.json")
        return orig_h, orig_w, new_h, new_w

    raise KeyError("geometry meta missing required size fields")


def restore_teacher_to_native(mask_teacher: np.ndarray, geom: dict) -> np.ndarray:
    orig_h, orig_w, new_h, new_w = _get_geom_sizes(geom)
    cropped = mask_teacher[:new_h, :new_w]
    native = cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return native.astype(mask_teacher.dtype)


def restore_native_mask_to_student(mask_native: np.ndarray, geom: dict) -> np.ndarray:
    """
    Map a native-space mask back to the fixed student canvas using geometry_meta["native_to_student"].
    Output shape must match student_img / student_gt.
    """
    if "native_to_student" not in geom:
        raise KeyError("geometry meta missing native_to_student")

    g = geom["native_to_student"]
    target_h = int(g["target_h"])
    target_w = int(g["target_w"])
    new_h = int(g["new_h"])
    new_w = int(g["new_w"])
    offset_x = int(g.get("offset_x", 0))
    offset_y = int(g.get("offset_y", 0))

    resized = cv2.resize(mask_native.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    return canvas


def teacher_box_to_native(box: np.ndarray, geom: dict) -> np.ndarray:
    orig_h, orig_w, new_h, new_w = _get_geom_sizes(geom)
    x1, y1, x2, y2 = box.astype(np.float32).tolist()

    x1 = x1 * float(orig_w) / max(float(new_w), 1.0)
    x2 = x2 * float(orig_w) / max(float(new_w), 1.0)
    y1 = y1 * float(orig_h) / max(float(new_h), 1.0)
    y2 = y2 * float(orig_h) / max(float(new_h), 1.0)

    return _clip_box_to_image(np.array([x1, y1, x2, y2], dtype=np.float32), orig_w, orig_h)


def restore_native_confirmed_fg_with_boxes(
    confirmed_fg_teacher: np.ndarray,
    prompt_instances: List[PromptInstance],
    geom: dict,
    is_multiclass: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    confirmed_fg_native = restore_teacher_to_native(confirmed_fg_teacher, geom).astype(np.uint8)
    orig_h, orig_w, _, _ = _get_geom_sizes(geom)

    allowed_by_label: Dict[int, np.ndarray] = {}
    union_all = np.zeros((orig_h, orig_w), dtype=bool)

    for ins in prompt_instances:
        lid = int(ins.label_id) if is_multiclass else 1
        if lid <= 0 or lid == 255:
            continue

        if ins.native_box is not None:
            native_box = _clip_box_to_image(ins.native_box, orig_w, orig_h)
        else:
            native_box = teacher_box_to_native(ins.teacher_box, geom)

        bm = box_to_mask_xyxy(native_box, orig_h, orig_w)
        union_all |= bm

        if lid not in allowed_by_label:
            allowed_by_label[lid] = bm.copy()
        else:
            allowed_by_label[lid] |= bm

    for lid, allow_mask in allowed_by_label.items():
        confirmed_fg_native[(confirmed_fg_native == lid) & (~allow_mask)] = 0

    return confirmed_fg_native, union_all


def rebuild_tri_from_confirmed_fg_and_boxes(
    confirmed_fg_map: np.ndarray,
    union_all: np.ndarray,
) -> np.ndarray:
    tri = np.zeros_like(confirmed_fg_map, dtype=np.uint8)
    tri[union_all] = 255
    tri[confirmed_fg_map > 0] = confirmed_fg_map[confirmed_fg_map > 0]
    return tri

def build_output_dirs(fold_root: str, split: str) -> Dict[str, str]:
    d = {
        "teacher_tri": os.path.join(fold_root, "pseudo_teacher", f"tri_{split}"),
        "teacher_vis": os.path.join(fold_root, "pseudo_teacher", f"vis_{split}"),
        "student_tri": os.path.join(fold_root, "pseudo_student", f"tri_{split}"),
        "student_vis": os.path.join(fold_root, "pseudo_student", f"vis_{split}"),
    }
    for p in d.values():
        ensure_dir(p)
    return d


def vis_overlay_multiclass(base: np.ndarray, label_map: np.ndarray) -> np.ndarray:
    img = base.copy()
    if img.max() <= 1.5:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    overlay = np.zeros_like(img)
    ignore = (label_map == 255)
    overlay[:, :, 1] = ignore.astype(np.uint8) * 255
    fg = (label_map > 0) & (label_map != 255)
    overlay[:, :, 2] = fg.astype(np.uint8) * 255
    return cv2.addWeighted(img, 0.75, overlay, 0.25, 0)


def draw_box_and_label(img: np.ndarray, box: np.ndarray, label_id: int, color=(255, 255, 0), text_prefix="L"):
    x1, y1, x2, y2 = [int(round(v)) for v in box.tolist()]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = f"{text_prefix}{label_id}"
    cv2.putText(img, text, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)


def draw_points_with_labels(
    img: np.ndarray,
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
):
    if point_coords is None or point_labels is None:
        return
    if len(point_coords) == 0:
        return
    for idx, (pt, lb) in enumerate(zip(point_coords, point_labels)):
        x, y = int(round(float(pt[0]))), int(round(float(pt[1])))
        lb_int = int(lb)
        if lb_int == 0:
            color = (0, 0, 255)   # red for negative point
            txt = f"N{idx}"
        else:
            color = (0, 255, 0)   # green for positive point
            txt = f"P{idx}"
        cv2.circle(img, (x, y), 4, color, -1)
        cv2.circle(img, (x, y), 6, (255, 255, 255), 1)
        cv2.putText(img, txt, (x + 6, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, txt, (x + 6, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def save_comparison_vis(
    image_rgb: np.ndarray,
    tri_map: np.ndarray,
    gt_map: Optional[np.ndarray],
    prompt_instances: List[PromptInstance],
    save_path: str,
):
    left = image_rgb.copy()
    if left.max() <= 1.5:
        left = (left * 255).astype(np.uint8)
    else:
        left = left.astype(np.uint8)

    right = vis_overlay_multiclass(image_rgb, tri_map)

    if gt_map is not None:
        gt_overlay = np.zeros_like(left)
        gt_overlay[:, :, 1] = (gt_map > 0).astype(np.uint8) * 255
        left = cv2.addWeighted(left, 0.75, gt_overlay, 0.25, 0)

    for ins in prompt_instances:
        draw_box_and_label(left, ins.teacher_box, ins.label_id, color=(255, 255, 255), text_prefix="GT")
        draw_box_and_label(right, ins.teacher_box, ins.label_id, color=(255, 255, 0), text_prefix="BX")
        draw_points_with_labels(left, ins.point_coords, ins.point_labels)
        draw_points_with_labels(right, ins.point_coords, ins.point_labels)

    canvas = np.hstack([left, right])
    ensure_dir(os.path.dirname(save_path))
    cv2.imwrite(save_path, canvas)


def _first_box_of_meta(ins_meta: dict) -> np.ndarray:
    if "bbox_teacher" in ins_meta:
        return np.array(ins_meta["bbox_teacher"], dtype=np.float32)
    if "bbox" in ins_meta:
        return np.array(ins_meta["bbox"], dtype=np.float32)
    raise KeyError("instance missing bbox_teacher/bbox")


def _parse_instances(meta: dict, filename: str, use_point_prompts: bool = False) -> List[PromptInstance]:
    if meta.get("empty_slice", False):
        return []

    if "instances" in meta and isinstance(meta["instances"], list) and len(meta["instances"]) > 0:
        out: List[PromptInstance] = []
        for ins in meta["instances"]:
            teacher_box = _first_box_of_meta(ins)
            native_box = None
            if "bbox_native" in ins and ins["bbox_native"] is not None:
                native_box = np.array(ins["bbox_native"], dtype=np.float32)
            label_id = int(ins.get("label_id", 1))

            point_coords = None
            point_labels = None
            if use_point_prompts and ("point_coords" in ins or "point_labels" in ins):
                raw_coords = ins.get("point_coords", [])
                raw_labels = ins.get("point_labels", [])
                if raw_coords is None:
                    raw_coords = []
                if raw_labels is None:
                    raw_labels = []
                coords_arr = np.asarray(raw_coords, dtype=np.float32).reshape(-1, 2) if len(raw_coords) > 0 else np.zeros((0, 2), dtype=np.float32)
                labels_arr = np.asarray(raw_labels, dtype=np.int32).reshape(-1) if len(raw_labels) > 0 else np.zeros((0,), dtype=np.int32)
                if coords_arr.shape[0] != labels_arr.shape[0]:
                    raise ValueError(
                        f"point prompt length mismatch in {filename}: "
                        f"len(point_coords)={coords_arr.shape[0]} vs len(point_labels)={labels_arr.shape[0]}"
                    )
                if coords_arr.shape[0] > 0:
                    point_coords = coords_arr
                    point_labels = labels_arr
            out.append(PromptInstance(
                teacher_box=teacher_box,
                native_box=native_box,
                label_id=label_id,
                point_coords=point_coords,
                point_labels=point_labels,
            ))
        return out

    if "bboxes" in meta:
        return [PromptInstance(teacher_box=np.array(b, dtype=np.float32), native_box=None, label_id=1, point_coords=None, point_labels=None) for b in meta["bboxes"]]

    if "bbox" in meta:
        return [PromptInstance(teacher_box=np.array(meta["bbox"], dtype=np.float32), native_box=None, label_id=1, point_coords=None, point_labels=None)]

    raise ValueError(f"Prompt format error: no instances/bboxes/bbox -> {filename}")


def select_and_build_proposal(
    predictor: SamPredictor,
    teacher_box: np.ndarray,
    native_box: Optional[np.ndarray],
    label_id: int,
    point_coords: Optional[np.ndarray],
    point_labels: Optional[np.ndarray],
    cfg: DatasetCfg,
    teacher_hw: Tuple[int, int],
    is_multiclass: bool,
) -> Optional[Proposal]:
    """
    Build one formal MedSAM proposal.

    Student V2 change:
      - request pixel logits with return_logits=True;
      - preserve the original foreground decision logits > 0,
        which is equivalent to sigmoid probability > 0.5;
      - preserve the original morphology, box constraint and
        fused proposal ranking;
      - retain the probability map belonging to EXACTLY the
        same multimask candidate selected as best_mask.

    Therefore probability does not redefine foreground.
    It is only additional evidence for cross-class arbitration.
    """
    h, w = teacher_hw

    box = _clip_box_to_image(
        teacher_box,
        w,
        h,
    )
    box_mask = box_to_mask_xyxy(
        box,
        h,
        w,
    )
    box_area = max(
        1.0,
        float(box_mask.sum()),
    )

    points = None
    labels = None

    if (
        point_coords is not None
        and point_labels is not None
        and len(point_coords) > 0
    ):
        points = _clip_points_to_image(
            point_coords,
            w=w,
            h=h,
        )
        labels = point_labels.astype(
            np.int32
        )

    if (
        points is not None
        and labels is not None
        and len(points) > 0
    ):
        mask_logits, scores, _ = predictor.predict(
            point_coords=points.astype(np.float32),
            point_labels=labels.astype(np.int32),
            box=box.astype(np.float32),
            multimask_output=True,
            return_logits=True,
        )
    else:
        mask_logits, scores, _ = predictor.predict(
            box=box.astype(np.float32),
            multimask_output=True,
            return_logits=True,
        )

    mask_logits = np.asarray(
        mask_logits,
        dtype=np.float32,
    )
    scores = np.asarray(
        scores,
    ).reshape(-1)

    if mask_logits.ndim == 2:
        mask_logits = mask_logits[
            None,
            ...,
        ]

    if mask_logits.shape[0] != scores.shape[0]:
        raise RuntimeError(
            "MedSAM multimask output mismatch: "
            f"logits={mask_logits.shape}, "
            f"scores={scores.shape}"
        )

    candidates = []

    spill_weight = (
        cfg.spill_weight
        if is_multiclass
        else max(cfg.spill_weight, 0.80)
    )
    area_bonus = (
        cfg.score_area_bonus
        if is_multiclass
        else max(cfg.score_area_bonus, 0.10)
    )

    for i in range(mask_logits.shape[0]):
        logits_i = np.asarray(
            mask_logits[i],
            dtype=np.float32,
        )

        if logits_i.shape != (h, w):
            raise RuntimeError(
                "MedSAM returned unexpected mask-logit size: "
                f"{logits_i.shape}, expected {(h, w)}"
            )

        # Original SAM mask_threshold is zero.
        # logits > 0 <=> sigmoid(logits) > 0.5.
        raw = (
            logits_i > 0.0
        ).astype(np.uint8)

        if raw.max() == 0:
            continue

        raw_area = float(
            raw.sum()
        )
        if raw_area <= 0:
            continue

        outside = float(
            (raw > 0).sum()
            - (
                (raw > 0)
                & box_mask
            ).sum()
        )

        spill_ratio = (
            outside
            / max(raw_area, 1.0)
        )

        refined = refine_mask_by_prior(
            mask=raw,
            box_mask=box_mask,
            keep_largest_cc=cfg.keep_largest_cc,
            is_multiclass=is_multiclass,
        )

        area = float(
            refined.sum()
        )
        if area <= 0:
            continue

        inside_ratio = min(
            1.0,
            area / box_area,
        )

        fused_score = (
            float(scores[i])
            + area_bonus * inside_ratio
            - spill_weight * spill_ratio
        )

        probability = sigmoid_np(
            logits_i
        )

        candidates.append(
            {
                "score": float(fused_score),
                "model_score": float(scores[i]),
                "mask": refined.astype(bool),
                "probability": probability.astype(
                    np.float32
                ),
                "candidate_index": int(i),
            }
        )

    if not candidates:
        return None

    # IMPORTANT:
    # ranking is exactly the existing V1 fused-score policy.
    candidates.sort(
        key=lambda x: x["score"],
        reverse=True,
    )

    best = candidates[0]

    best_probability = np.asarray(
        best["probability"],
        dtype=np.float32,
    )

    best_mask = np.asarray(
        best["mask"],
        dtype=bool,
    )

    if (
        best_probability.shape
        != best_mask.shape
    ):
        raise RuntimeError(
            "Selected proposal probability/mask "
            "shape mismatch: "
            f"{best_probability.shape} "
            f"vs {best_mask.shape}"
        )

    return Proposal(
        label_id=int(label_id),
        score=float(best["score"]),
        best_mask=best_mask,
        best_probability=best_probability,
        best_model_score=float(best["model_score"]),
        teacher_box=box.astype(np.float32),
        native_box=(
            native_box.astype(np.float32)
            if native_box is not None
            else None
        ),
        point_coords=(
            points.astype(np.float32)
            if points is not None
            else None
        ),
        point_labels=(
            labels.astype(np.int32)
            if labels is not None
            else None
        ),
    )


def generate_teacher_maps(
    predictor: SamPredictor,
    teacher_rgb: np.ndarray,
    prompt_instances: List[PromptInstance],
    is_multiclass: bool,
    teacher_hw: Tuple[int, int],
    cfg: DatasetCfg,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    List[PromptInstance],
]:
    """
    Generate teacher-space tri labels using the V2 resolver.

    V2 semantics:
      outside ALL valid prompt boxes:
          0

      inside valid prompt-box union but no selected MedSAM
      proposal confirms foreground:
          255

      exactly one class confirms foreground:
          class ID

      multiple DIFFERENT classes confirm foreground:
          select the class with highest pixel probability

      numerical top1/top2 tie:
          255

    Same-class instances are aggregated by:
      support       = OR
      probability   = MAX
    """
    h, w = teacher_hw

    img_for_sam = enhance_image_by_dataset(
        teacher_rgb,
        cfg.enhancement,
    )
    predictor.set_image(
        img_for_sam
    )

    proposals: List[Proposal] = []

    # Canonical union must depend on prompts/boxes,
    # NOT on whether MedSAM succeeds.
    union_all = np.zeros(
        (h, w),
        dtype=bool,
    )

    for ins in prompt_instances:
        lid = (
            int(ins.label_id)
            if is_multiclass
            else 1
        )

        if lid <= 0 or lid == 255:
            continue

        canonical_box = _clip_box_to_image(
            ins.teacher_box,
            w,
            h,
        )

        # Critical V2 fix:
        # even if MedSAM returns no valid proposal,
        # this canonical prompt box is still uncertain
        # supervision rather than background.
        union_all |= box_to_mask_xyxy(
            canonical_box,
            h,
            w,
        )

        prop = select_and_build_proposal(
            predictor=predictor,
            teacher_box=canonical_box,
            native_box=ins.native_box,
            label_id=lid,
            point_coords=ins.point_coords,
            point_labels=ins.point_labels,
            cfg=cfg,
            teacher_hw=teacher_hw,
            is_multiclass=is_multiclass,
        )

        if prop is None:
            continue

        proposals.append(
            prop
        )

    # Keep deterministic ordering for visualization/auditing.
    proposals.sort(
        key=lambda p: p.score,
        reverse=True,
    )

    class_probabilities: Dict[
        int,
        np.ndarray,
    ] = {}

    class_support_masks: Dict[
        int,
        np.ndarray,
    ] = {}

    for prop in proposals:
        aggregate_same_class_evidence(
            class_probabilities=class_probabilities,
            class_support_masks=class_support_masks,
            label_id=int(prop.label_id),
            probability=prop.best_probability,
            support_mask=prop.best_mask,
        )

    confirmed_fg_map, tri_map, _resolver_stats = (
        resolve_multiclass_evidence(
            class_probabilities=class_probabilities,
            class_support_masks=class_support_masks,
            union_all=union_all,
        )
    )

    # Hard contract:
    # confirmed foreground and tri pseudo must agree
    # on every confirmed pixel.
    confirmed_pixels = (
        confirmed_fg_map > 0
    )

    if not np.array_equal(
        tri_map[confirmed_pixels],
        confirmed_fg_map[confirmed_pixels],
    ):
        raise RuntimeError(
            "V2 resolver produced inconsistent "
            "confirmed_fg_map / tri_map."
        )

    # Outside canonical prompt union must remain background.
    if np.any(
        tri_map[~union_all] != 0
    ):
        raise RuntimeError(
            "V2 tri pseudo contains nonzero labels "
            "outside canonical box union."
        )

    vis_instances = [
        PromptInstance(
            teacher_box=p.teacher_box.copy(),
            native_box=(
                p.native_box.copy()
                if p.native_box is not None
                else None
            ),
            label_id=int(p.label_id),
            point_coords=(
                p.point_coords.copy()
                if p.point_coords is not None
                else None
            ),
            point_labels=(
                p.point_labels.copy()
                if p.point_labels is not None
                else None
            ),
        )
        for p in proposals
    ]

    return (
        confirmed_fg_map,
        tri_map,
        vis_instances,
    )


def find_prompt_json(fold_root: str, split: str, prompt_name: Optional[str] = None) -> str:
    if prompt_name is not None and str(prompt_name).strip():
        pn = str(prompt_name).strip()
        if os.path.isabs(pn):
            candidates = [pn]
        else:
            candidates = [
                os.path.join(fold_root, pn),
                os.path.join(fold_root, "prompts", pn),
            ]
    else:
        candidates = [
            os.path.join(fold_root, f"prompts_{split}.json"),
            os.path.join(fold_root, "prompts", f"prompts_{split}.json"),
        ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"prompt json not found for split={split}. prompt_name={prompt_name}. tried: {candidates}")


def process_one_fold(
    dataset_name: str,
    fold_root: str,
    split: str,
    predictor: SamPredictor,
    max_samples: Optional[int],
    overwrite: bool,
    vis_limit: int,
    prompt_name: Optional[str],
    use_point_prompts: bool,
    stage_tag: str,
):
    try:
        prompt_json_path = find_prompt_json(fold_root, split, prompt_name=prompt_name)
    except FileNotFoundError as e:
        print(f"  - skip: {e}")
        return

    ds_cfg = get_dataset_cfg(dataset_name)
    fold_name = os.path.basename(fold_root)
    suffix = safe_suffix(stage_tag)
    stage_name = (f"pseudo_train{suffix}" if split == "train" else f"pseudo_{split}{suffix}")
    stage_time_path = os.path.join(fold_root, "meta", f"stage_time_{stage_name}.json")
    with StageTimer(
        save_path=stage_time_path,
        stage_name=stage_name,
        dataset=dataset_name,
        fold=fold_name,
        split=split,
    ) as timer:
        log_file = os.path.join(fold_root, "meta", f"pseudo_box255_error_{split}{suffix}.log")
        outputs = build_output_dirs(fold_root, split)
        manifest_index = load_manifest_index(fold_root)
        geometry_meta = load_geometry_meta(fold_root)
        teacher_target = load_teacher_target_size(fold_root)
        prompts = load_json(prompt_json_path)
        is_multiclass = infer_is_multiclass(dataset_name, fold_root)
        mode = "multiclass" if is_multiclass else "binary"

        items = list(prompts.items())
        if max_samples is not None and max_samples > 0:
            items = items[:max_samples]

        vis_names = set([x[0] for x in items[:min(vis_limit, len(items))]])

        stats = {
            "dataset": dataset_name,
            "fold": fold_name,
            "split": split,
            "mode": mode,
            "teacher_target_size": teacher_target,
            "num_prompt_items": len(items),
            "num_processed": 0,
            "num_skipped_existing": 0,
            "num_errors": 0,
            "pseudo_version": "Pseudo-v4-box-interior-255",
            "tri_definition": {
                "outside_bbox_union": 0,
                "inside_bbox_union_and_medsam_confirmed": "1..K",
                "inside_bbox_union_but_not_medsam_confirmed": 255,
            },
            "config": ds_cfg.__dict__,
            "prompt_json_path": prompt_json_path,
            "prompt_name": "" if prompt_name is None else str(prompt_name),
            "use_point_prompts": bool(use_point_prompts),
            "stage_tag": str(stage_tag),
        }

        print(f"\nPseudo-v4 | dataset={dataset_name} | fold={fold_name} | split={split} | mode={mode}")

        for filename, meta in tqdm(items, desc=f"{dataset_name}:{fold_name}:{split}", dynamic_ncols=True, ascii=True):
            try:
                if filename not in manifest_index:
                    raise KeyError(f"{filename} not found in manifest.json")
                if filename not in geometry_meta:
                    raise KeyError(f"{filename} not found in geometry_meta.json")

                manifest_item = manifest_index[filename]
                if manifest_item["split"] != split:
                    raise RuntimeError(f"[LEAKAGE] {filename} is split={manifest_item['split']}, but running split={split}")
                if meta.get("split", split) != split:
                    raise RuntimeError(f"Prompt split mismatch: prompt says {meta.get('split')}, cli split={split}, file={filename}")
                if meta.get("fold", fold_name) != fold_name:
                    raise RuntimeError(f"Prompt fold mismatch: prompt says {meta.get('fold')}, current fold={fold_name}, file={filename}")

                teacher_tri_path = os.path.join(outputs["teacher_tri"], filename)
                student_tri_path = os.path.join(outputs["student_tri"], filename)
                teacher_vis_path = os.path.join(outputs["teacher_vis"], filename.replace(".npy", ".jpg"))
                student_vis_path = os.path.join(outputs["student_vis"], filename.replace(".npy", ".jpg"))

                must_exist = [teacher_tri_path, student_tri_path]

                if (not overwrite) and all(os.path.exists(p) for p in must_exist):
                    stats["num_skipped_existing"] += 1
                    continue

                teacher_img_path = manifest_item["teacher_img"]
                student_img_path = manifest_item["student_img"]
                teacher_gt_path = manifest_item["teacher_gt"]

                if not os.path.exists(teacher_img_path):
                    raise FileNotFoundError(f"teacher image not found: {teacher_img_path}")
                if not os.path.exists(student_img_path):
                    raise FileNotFoundError(f"student image not found: {student_img_path}")

                teacher_img = np.load(teacher_img_path)
                teacher_rgb = _to_uint8_rgb(teacher_img, (teacher_target, teacher_target))
                student_img = np.load(student_img_path)
                student_rgb = _to_uint8_rgb(student_img, student_img.shape[:2])

                prompt_instances = _parse_instances(meta, filename, use_point_prompts=use_point_prompts)

                if len(prompt_instances) == 0:
                    confirmed_fg_teacher = np.zeros((teacher_target, teacher_target), dtype=np.uint8)
                    teacher_tri = np.zeros((teacher_target, teacher_target), dtype=np.uint8)
                    vis_instances = []
                else:
                    confirmed_fg_teacher, teacher_tri, vis_instances = generate_teacher_maps(
                        predictor=predictor,
                        teacher_rgb=teacher_rgb,
                        prompt_instances=prompt_instances,
                        is_multiclass=is_multiclass,
                        teacher_hw=(teacher_target, teacher_target),
                        cfg=ds_cfg,
                    )

                np.save(teacher_tri_path, teacher_tri)

                geom = geometry_meta[filename]
                confirmed_fg_student, native_union_boxes = restore_native_confirmed_fg_with_boxes(
                    confirmed_fg_teacher=confirmed_fg_teacher,
                    prompt_instances=prompt_instances,
                    geom=geom,
                    is_multiclass=is_multiclass,
                )
                student_tri = rebuild_tri_from_confirmed_fg_and_boxes(
                    confirmed_fg_map=confirmed_fg_student,
                    union_all=native_union_boxes,
                )

                if student_tri.shape != student_img.shape[:2]:
                    student_tri = restore_native_mask_to_student(student_tri, geom)
                np.save(student_tri_path, student_tri)

                if filename in vis_names:
                    try:
                        gt_map = _load_gt_2d(teacher_gt_path) if os.path.exists(teacher_gt_path) else None
                        save_comparison_vis(
                            image_rgb=teacher_rgb,
                            tri_map=teacher_tri,
                            gt_map=gt_map,
                            prompt_instances=vis_instances,
                            save_path=teacher_vis_path,
                        )

                        native_prompt_vis = []
                        orig_h, orig_w, _, _ = _get_geom_sizes(geom)
                        for ins in prompt_instances:
                            if ins.native_box is not None:
                                nb = _clip_box_to_image(ins.native_box, orig_w, orig_h)
                            else:
                                nb = teacher_box_to_native(ins.teacher_box, geom)
                            native_prompt_vis.append(PromptInstance(teacher_box=nb, native_box=nb, label_id=ins.label_id))

                        student_gt_path = manifest_item["student_gt"]
                        student_gt = _load_gt_2d(student_gt_path) if os.path.exists(student_gt_path) else None
                        save_comparison_vis(
                            image_rgb=student_rgb,
                            tri_map=student_tri,
                            gt_map=student_gt,
                            prompt_instances=native_prompt_vis,
                            save_path=student_vis_path,
                        )
                    except Exception as e_vis:
                        print(f"[VIS-ONLY WARNING] {filename}: {repr(e_vis)}")

                stats["num_processed"] += 1
                timer.set_outputs(stats["num_processed"])

            except Exception as e:
                stats["num_errors"] += 1
                log_error(log_file, f"[{dataset_name}][{fold_name}][{split}] Error {filename}: {repr(e)}")

        save_json(stats, os.path.join(fold_root, f"pseudo_box255_stats_{split}{suffix}.json"))
        timer.set_outputs(stats["num_processed"])
        print(
            f"done {dataset_name} {fold_name} {split}: "
            f"processed={stats['num_processed']} skipped={stats['num_skipped_existing']} errors={stats['num_errors']}"
        )


def process_dataset(
    dataset_name: str,
    predictor: SamPredictor,
    base_dir: str,
    fold_arg: str,
    split_arg: str,
    max_samples: Optional[int],
    overwrite: bool,
    vis_limit: int,
    prompt_name: Optional[str],
    use_point_prompts: bool,
    stage_tag: str,
):
    ds_root = os.path.join(base_dir, "processed", dataset_name)
    if not os.path.isdir(ds_root):
        raise FileNotFoundError(f"dataset processed root not found: {ds_root}")

    folds = sorted(glob.glob(os.path.join(ds_root, "fold_*")))
    if not folds:
        raise FileNotFoundError(f"no folds found under {ds_root}")

    if fold_arg != "all":
        folds = [x for x in folds if os.path.basename(x) == fold_arg]
        if not folds:
            raise FileNotFoundError(f"{dataset_name}: fold {fold_arg} not found")

    split_list = ["train", "test"] if split_arg == "all" else [split_arg]

    for fold_root in folds:
        for split in split_list:
            process_one_fold(
                dataset_name=dataset_name,
                fold_root=fold_root,
                split=split,
                predictor=predictor,
                max_samples=max_samples,
                overwrite=overwrite,
                vis_limit=vis_limit,
                prompt_name=prompt_name,
                use_point_prompts=use_point_prompts,
                stage_tag=stage_tag,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_TARGET_DATASETS), help="comma-separated datasets")
    parser.add_argument("--fold", type=str, default="all", help='fold_0/fold_1/... or "all"')
    parser.add_argument("--split", type=str, default="train", choices=["train", "test", "all"])
    parser.add_argument("--max_samples", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--vis_limit", type=int, default=0, help="save visualization for first N prompt slices per fold/split")
    parser.add_argument("--prompt_name", type=str, default="", help="prompt file name under fold root/prompts (e.g. prompts_train_idea1_hardneg.json)")
    parser.add_argument("--use_point_prompts", action="store_true", help="enable point_coords/point_labels prompts together with bbox")
    parser.add_argument("--stage_tag", type=str, default="", help="optional suffix tag for stage_time/stats outputs, e.g. idea1")
    args = parser.parse_args()

    if args.use_point_prompts:
        raise ValueError(
            "generate_pseudo_labels.py is M0/box-only only. "
            "Use generate_pseudo_labels_m1.py for box+point M1 pseudo labels."
        )

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    datasets = parse_list(args.datasets)
    if not datasets:
        datasets = DEFAULT_TARGET_DATASETS

    print("Generating pseudo labels with the following configuration:")
    print(f"   DEVICE          = {DEVICE}")
    print(f"   datasets        = {datasets}")
    print(f"   fold            = {args.fold}")
    print(f"   split           = {args.split}")
    print(f"   overwrite       = {args.overwrite}")
    print(f"   prompt_name     = {args.prompt_name if args.prompt_name else '<default by split>'}")
    print(f"   use_point_prompts = {args.use_point_prompts}")
    print(f"   stage_tag       = {args.stage_tag if args.stage_tag else '<none>'}")

    sam_model = sam_model_registry[MODEL_TYPE](checkpoint=args.checkpoint)
    sam_model.to(device=DEVICE)
    predictor = SamPredictor(sam_model)

    with torch.no_grad():
        for ds in datasets:
            process_dataset(
                dataset_name=ds,
                predictor=predictor,
                base_dir=args.base_dir,
                fold_arg=args.fold,
                split_arg=args.split,
                max_samples=(None if args.max_samples is None or args.max_samples < 0 else args.max_samples),
                overwrite=args.overwrite,
                vis_limit=args.vis_limit,
                prompt_name=(args.prompt_name.strip() if args.prompt_name and args.prompt_name.strip() else None),
                use_point_prompts=bool(args.use_point_prompts),
                stage_tag=str(args.stage_tag),
            )


if __name__ == "__main__":
    main()
