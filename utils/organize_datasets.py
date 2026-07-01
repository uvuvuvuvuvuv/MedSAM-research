#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Organize heterogeneous medical segmentation datasets into a unified layout.

organized/
└── <dataset_name>/
    ├── meta/
    │   ├── adapter_report.txt
    │   ├── raw_manifest.json
    │   ├── sanity_check.csv
    │   ├── split_meta.json
    │   ├── train_cases.json
    │   └── test_cases.json
    ├── train/
    │   ├── index.json
    │   └── cases/<case_id>/...
    └── test/
        ├── index.json
        └── cases/<case_id>/...

This script only builds the organized layer. It does NOT generate teacher/student
npy files, prompts, or pseudo labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import tifffile
except Exception:
    tifffile = None


IMG_EXTS_2D = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
JPEG_MASK_DATASETS = {"tn3k", "tg3k", "kvasirseg"}
PURE_BINARY_MASK_DATASETS = {"cvc_clinicdb", "ddti", "otu_2d", "ph2", "drive", "hrf", "chasedb1"}
SMALL_SATELLITE_CLEANUP_RULES = {
    "cvc_clinicdb": {"max_area": 128, "max_ratio": 0.01},
    "ddti": {"max_area": 128, "max_ratio": 0.01},
}
DATASET_MASK_RULES = {
    "tn3k": {"binarize_mode": "threshold", "threshold": 128, "cleanup_mode": None},
    "tg3k": {"binarize_mode": "threshold", "threshold": 128, "cleanup_mode": "tg3k_corner_artifacts"},
    "kvasirseg": {"binarize_mode": "threshold", "threshold": 128, "cleanup_mode": "kvasir_corner_artifacts"},
    "cvc_clinicdb": {"binarize_mode": "nonzero", "threshold": 1, "cleanup_mode": "tiny_satellite_components"},
    "ddti": {"binarize_mode": "nonzero", "threshold": 1, "cleanup_mode": "tiny_satellite_components"},
    "otu_2d": {"binarize_mode": "nonzero", "threshold": 1, "cleanup_mode": None},
    "ph2": {"binarize_mode": "nonzero", "threshold": 1, "cleanup_mode": None},
    "drive": {"binarize_mode": "nonzero", "threshold": 1, "cleanup_mode": None},
    "hrf": {"binarize_mode": "nonzero", "threshold": 1, "cleanup_mode": None},
    "chasedb1": {"binarize_mode": "nonzero", "threshold": 1, "cleanup_mode": None},
}


def norm_name(s: str) -> str:
    return s.strip().lower().replace("-", "_").replace(" ", "_")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_text(text: str, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def safe_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".npy.h5"):
        return name[:-7]
    return path.stem


def suffix_for_copy(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return ".nii.gz"
    if path.name.endswith(".npy.h5"):
        return ".h5"
    return path.suffix.lower()


def file_stem_map(files: List[Path]) -> Dict[str, Path]:
    return {safe_stem(p): p for p in files}


def list_immediate_files(folder: Path) -> List[Path]:
    return [p for p in folder.iterdir() if p.is_file()]


def image_open_any(path: Path) -> Image.Image:
    if path.suffix.lower() in {".tif", ".tiff"} and tifffile is not None:
        arr = tifffile.imread(str(path))
        return Image.fromarray(arr)
    return Image.open(path)


def to_uint8_img(img: Image.Image) -> Image.Image:
    arr = np.array(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.float32)
        lo, hi = arr.min(), arr.max()
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def mask_rule_for_dataset(dataset_name: Optional[str], mask_role: str = "label") -> Dict[str, Any]:
    if mask_role != "label":
        return {"binarize_mode": "nonzero", "threshold": 1, "cleanup_mode": None}
    ds_key = norm_name(dataset_name or "")
    return dict(DATASET_MASK_RULES.get(ds_key, {"binarize_mode": "nonzero", "threshold": 1, "cleanup_mode": None}))


def summarize_connected_components(mask_u8: np.ndarray) -> Dict[str, Any]:
    fg = (mask_u8 > 0).astype(np.uint8)
    if fg.sum() == 0:
        return {
            "num_components": 0,
            "largest_component_area": 0,
            "smallest_component_area": 0,
            "top_left_foreground": False,
            "border_foreground_pixels": 0,
            "component_boxes_head": [],
        }
    num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    areas = []
    boxes = []
    for comp_id in range(1, num_labels):
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        areas.append(area)
        boxes.append({"area": area, "bbox": [x, y, x + w - 1, y + h - 1]})
    border_fg = fg[0, :].sum() + fg[-1, :].sum() + fg[1:-1, 0].sum() + fg[1:-1, -1].sum()
    boxes = sorted(boxes, key=lambda item: (-item["area"], item["bbox"][1], item["bbox"][0]))
    return {
        "num_components": len(areas),
        "largest_component_area": int(max(areas)) if areas else 0,
        "smallest_component_area": int(min(areas)) if areas else 0,
        "top_left_foreground": bool(fg[0, 0]),
        "border_foreground_pixels": int(border_fg),
        "component_boxes_head": boxes[:8],
    }


def filter_dataset_artifact_components(mask_u8: np.ndarray, dataset_name: Optional[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    ds_key = norm_name(dataset_name or "")
    fg = (mask_u8 > 0).astype(np.uint8)
    if fg.sum() == 0:
        return fg.astype(np.uint8) * 255, {"cleanup_mode": None, "removed_components": []}
    num_labels, label_map, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    cleaned = fg.copy()
    removed: List[Dict[str, Any]] = []
    largest_area = 0
    if num_labels > 1:
        largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
    sat_rule = SMALL_SATELLITE_CLEANUP_RULES.get(ds_key)

    def remove_component(comp_id: int, reason: str) -> None:
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        cleaned[label_map == comp_id] = 0
        removed.append(
            {
                "component_id": int(comp_id),
                "area": area,
                "bbox": [x, y, x + w - 1, y + h - 1],
                "reason": reason,
            }
        )

    for comp_id in range(1, num_labels):
        x = int(stats[comp_id, cv2.CC_STAT_LEFT])
        y = int(stats[comp_id, cv2.CC_STAT_TOP])
        w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
        h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
        area = int(stats[comp_id, cv2.CC_STAT_AREA])
        x2 = x + w - 1
        y2 = y + h - 1
        if ds_key == "tg3k":
            if x == 0 and y == 0 and x2 <= 15 and y2 <= 15 and area <= 128:
                remove_component(comp_id, "tg3k_top_left_systematic_artifact")
                continue
            if y == 0 and h <= 2 and area <= 4:
                remove_component(comp_id, "tg3k_top_edge_singleton")
                continue
        if ds_key == "kvasirseg":
            if x == 0 and y == 0 and x2 <= 7 and y2 <= 7 and area <= 16:
                remove_component(comp_id, "kvasir_top_left_jpeg_artifact")
                continue
            if (x == 0 or y == 0) and area <= 8 and max(w, h) <= 8:
                remove_component(comp_id, "kvasir_border_jpeg_artifact")
                continue
        if sat_rule is not None and largest_area > 0 and area < largest_area:
            ratio_cap = max(1, int(np.ceil(float(largest_area) * float(sat_rule["max_ratio"]))))
            if area <= int(sat_rule["max_area"]) and area <= ratio_cap:
                remove_component(comp_id, f"{ds_key}_tiny_satellite_component")
                continue
    cleanup_mode = DATASET_MASK_RULES.get(ds_key, {}).get("cleanup_mode")
    return cleaned.astype(np.uint8) * 255, {"cleanup_mode": cleanup_mode, "removed_components": removed}


def binarize_mask_array(arr: np.ndarray, dataset_name: Optional[str], mask_role: str = "label") -> Tuple[np.ndarray, Dict[str, Any]]:
    raw = np.array(arr)
    if raw.ndim == 3:
        raw = raw.max(axis=2)
    rule = mask_rule_for_dataset(dataset_name, mask_role=mask_role)
    if raw.dtype == np.bool_:
        fg = raw.astype(np.uint8)
    else:
        raw_u8 = raw.astype(np.uint8)
        if rule["binarize_mode"] == "threshold":
            fg = (raw_u8 >= int(rule["threshold"])).astype(np.uint8)
        else:
            fg = (raw_u8 > 0).astype(np.uint8)
    bin_mask = fg.astype(np.uint8) * 255
    cleanup_info = {"cleanup_mode": None, "removed_components": []}
    if mask_role == "label":
        bin_mask, cleanup_info = filter_dataset_artifact_components(bin_mask, dataset_name)
    unique_vals = np.unique(raw)
    audit = {
        "dataset_name": norm_name(dataset_name or ""),
        "mask_role": mask_role,
        "binarize_mode": rule["binarize_mode"],
        "threshold": int(rule["threshold"]),
        "cleanup_mode": cleanup_info.get("cleanup_mode"),
        "source_dtype": str(raw.dtype),
        "source_shape": list(raw.shape),
        "source_unique_count": int(len(unique_vals)),
        "source_unique_head": [int(v) for v in unique_vals[:12]],
        "source_unique_tail": [int(v) for v in unique_vals[-12:]],
        "fg_pixels_gt0": int((raw > 0).sum()),
        "fg_pixels_ge128": int((raw >= 128).sum()) if raw.dtype != np.bool_ else int(raw.sum()),
        "fg_pixels_eq255": int((raw == 255).sum()) if raw.dtype != np.bool_ else int(raw.sum()),
        "removed_components": cleanup_info.get("removed_components", []),
    }
    audit.update(summarize_connected_components(bin_mask))
    audit["foreground_pixels_after_cleanup"] = int((bin_mask > 0).sum())
    return bin_mask.astype(np.uint8), audit


def save_image_as_png(
    src: Path,
    dst: Path,
    is_mask: bool = False,
    *,
    dataset_name: Optional[str] = None,
    mask_role: str = "label",
) -> Optional[Dict[str, Any]]:
    ensure_dir(dst.parent)
    img = image_open_any(src)
    if is_mask:
        arr = np.array(img)
        mask, audit = binarize_mask_array(arr, dataset_name=dataset_name, mask_role=mask_role)
        Image.fromarray(mask).save(dst)
        return audit
    else:
        img = to_uint8_img(img)
        if img.mode not in ("L", "RGB"):
            img = img.convert("RGB")
        img.save(dst)
    return None


def copy_or_link(src: Path, dst: Path, mode: str = "copy"):
    ensure_dir(dst.parent)
    if dst.exists():
        dst.unlink()
    if mode == "link":
        try:
            os.link(src, dst)
            return
        except Exception:
            pass
    shutil.copy2(src, dst)


def read_table(path: Path):
    if pd is None:
        raise RuntimeError("pandas is required to read csv/xls/xlsx tables")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported table format: {path}")


def write_csv(rows: List[Dict], path: Path):
    ensure_dir(path.parent)
    keys = sorted({k for r in rows for k in r.keys()}) if rows else []
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def train_test_split_fixed(items: List[str], n_train: int, seed: int) -> Tuple[List[str], List[str]]:
    items = sorted(items)
    rnd = random.Random(seed)
    tmp = items[:]
    rnd.shuffle(tmp)
    return sorted(tmp[:n_train]), sorted(tmp[n_train:])


def stratified_split(case_to_label: Dict[str, str], train_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    rnd = random.Random(seed)
    buckets = defaultdict(list)
    for c, y in case_to_label.items():
        buckets[str(y)].append(c)
    train, test = [], []
    for _, cases in sorted(buckets.items()):
        cases = sorted(cases)
        rnd.shuffle(cases)
        n_train = int(round(len(cases) * train_ratio))
        if len(cases) > 1:
            n_train = min(max(n_train, 1), len(cases) - 1)
        train.extend(cases[:n_train])
        test.extend(cases[n_train:])
    return sorted(train), sorted(test)


def find_first(root: Path, names: List[str], is_dir: bool = True) -> Optional[Path]:
    for name in names:
        p = root / name
        if is_dir and p.is_dir():
            return p
        if (not is_dir) and p.is_file():
            return p
    return None


def choose_one(folder: Path) -> Optional[Path]:
    files = [p for p in folder.iterdir() if p.is_file()]
    if not files:
        return None
    files.sort(key=lambda p: p.name)
    return files[0]


def parse_simple_list_file(path: Path) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(Path(s).stem)
    return out


def deep_find_train_val(obj) -> Tuple[List[str], List[str]]:
    train, val = [], []

    def rec(x):
        nonlocal train, val
        if isinstance(x, dict):
            for k, v in x.items():
                lk = str(k).lower()
                if lk in {"train", "training"} and isinstance(v, list):
                    train.extend([Path(str(t)).stem for t in v])
                elif lk in {"val", "valid", "validation", "test"} and isinstance(v, list):
                    val.extend([Path(str(t)).stem for t in v])
                else:
                    rec(v)
        elif isinstance(x, list):
            for it in x:
                rec(it)

    rec(obj)
    return sorted(set(train)), sorted(set(val))


def infer_pathology_from_suffix(case_id: str) -> Optional[str]:
    if case_id.endswith("_h"):
        return "h"
    if case_id.endswith("_dr"):
        return "dr"
    if case_id.endswith("_g"):
        return "g"
    return None


def sanitize_case_id(s: str) -> str:
    s = str(s).strip()
    s = s.replace("\\", "_").replace("/", "_")
    s = re.sub(r"\s+", "_", s)
    return s


def xml_to_binary_mask(xml_path: Path, ref_image_path: Path) -> np.ndarray:
    img = image_open_any(ref_image_path)
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for region in root.iter():
        vertices = []
        for vertex in region.findall(".//Vertex"):
            x = vertex.attrib.get("X", vertex.attrib.get("x"))
            y = vertex.attrib.get("Y", vertex.attrib.get("y"))
            if x is None or y is None:
                continue
            try:
                vertices.append((float(x), float(y)))
            except Exception:
                continue
        if len(vertices) >= 3:
            draw.polygon(vertices, outline=1, fill=1)
    return (np.array(mask) > 0).astype(np.uint8) * 255


def save_binary_mask_from_array(arr: np.ndarray, dst: Path):
    ensure_dir(dst.parent)
    arr = (arr > 0).astype(np.uint8) * 255
    Image.fromarray(arr).save(dst)


def maybe_read_ph2_metadata(root: Path) -> Dict[str, str]:
    if pd is None:
        return {}
    candidates = []
    for p in root.iterdir():
        if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"} and "ph2" in p.name.lower():
            candidates.append(p)
    labels = {}
    for p in candidates:
        try:
            df = read_table(p)
        except Exception:
            continue
        cols = {str(c).lower(): c for c in df.columns}
        id_col = cols.get("image_name") or cols.get("image") or cols.get("id") or df.columns[0]
        lab_col = cols.get("diagnosis") or cols.get("class") or cols.get("label")
        if lab_col is None:
            for c in df.columns:
                vals = set(str(v).strip().lower() for v in df[c].dropna().tolist()[:50])
                joined = " ".join(vals)
                if any(k in joined for k in ["melanoma", "nevus", "atypical"]):
                    lab_col = c
                    break
        if lab_col is None:
            continue
        for _, row in df.iterrows():
            cid = sanitize_case_id(str(row[id_col]))
            labels[cid] = str(row[lab_col]).strip()
        if labels:
            return labels
    return {}


class CaseItem(dict):
    pass


# -----------------------------
# Adapters
# -----------------------------
def adapter_tn3k(raw_ds_root: Path, seed: int):
    train_img = raw_ds_root / "trainval-image"
    train_gt = raw_ds_root / "trainval-mask"
    test_img = raw_ds_root / "test-image"
    test_gt = raw_ds_root / "test-mask"
    items = []
    for split, img_dir, gt_dir in [("train", train_img, train_gt), ("test", test_img, test_gt)]:
        img_map = file_stem_map(list_immediate_files(img_dir))
        gt_map = file_stem_map(list_immediate_files(gt_dir))
        for stem, ip in sorted(img_map.items()):
            gp = gt_map.get(stem)
            if gp is None:
                continue
            items.append(CaseItem(dataset_name="tn3k", case_id=stem, patient_id=stem, split=split,
                                  modality="Ultrasound", task_dim="2D", class_info="binary",
                                  image_src=ip, label_src=gp, source_format="image", aux_src={}, extra_meta={}))
    report = "TN3K: fixed official trainval/test split; 2D image-mask pairing by identical stem."
    meta = {"dataset_name": "tn3k", "split_protocol": "official_fixed_split",
            "train_cases": [x["case_id"] for x in items if x["split"] == "train"],
            "test_cases": [x["case_id"] for x in items if x["split"] == "test"]}
    return items, report, meta
def adapter_cvc_clinicdb(raw_ds_root: Path, seed: int) -> Tuple[List[CaseItem], str, Dict]:
    items = []

    split_specs = [
        ("train", raw_ds_root / "train" / "image",  raw_ds_root / "train" / "masks"),
        ("test",  raw_ds_root / "test"  / "images", raw_ds_root / "test"  / "masks"),
    ]

    for split, img_dir, gt_dir in split_specs:
        if not img_dir.exists() or not gt_dir.exists():
            raise FileNotFoundError(f"CVC-ClinicDB split dirs missing: {img_dir} / {gt_dir}")

        img_map = file_stem_map([p for p in img_dir.iterdir() if p.is_file()])
        gt_map  = file_stem_map([p for p in gt_dir.iterdir() if p.is_file()])
        common = sorted(set(img_map) & set(gt_map))

        for stem in common:
            items.append(CaseItem(
                dataset_name="cvc_clinicdb",
                case_id=stem,
                patient_id=stem,
                split=split,
                modality="Endoscopy",
                task_dim="2D",
                class_info="binary",
                image_src=img_map[stem],
                label_src=gt_map[stem],
                source_format="image",
                aux_src={},
                extra_meta={}
            ))

    report = "CVC-ClinicDB: use existing nested train/test split; pair train/image with train/masks and test/images with test/masks by identical stem."
    split_meta = {
        "dataset_name": "cvc_clinicdb",
        "split_protocol": "existing_nested_train_test_split",
        "train_cases": [x["case_id"] for x in items if x["split"] == "train"],
        "test_cases": [x["case_id"] for x in items if x["split"] == "test"],
    }
    return items, report, split_meta

def adapter_tg3k(raw_ds_root: Path, seed: int):
    img_dir = raw_ds_root / "thyroid-image"
    gt_dir = raw_ds_root / "thyroid-mask"
    json_path = raw_ds_root / "tg3k-trainval.json"
    if not json_path.exists():
        raise FileNotFoundError("tg3k-trainval.json not found")
    train_cases, val_cases = deep_find_train_val(read_json(json_path))
    img_map = file_stem_map(list_immediate_files(img_dir))
    gt_map = file_stem_map(list_immediate_files(gt_dir))
    items = []
    for stem, ip in sorted(img_map.items()):
        gp = gt_map.get(stem)
        if gp is None:
            continue
        split = "train" if stem in train_cases else ("test" if stem in val_cases else None)
        if split is None:
            continue
        items.append(CaseItem(dataset_name="tg3k", case_id=stem, patient_id=stem, split=split,
                              modality="Ultrasound", task_dim="2D", class_info="binary",
                              image_src=ip, label_src=gp, source_format="image", aux_src={}, extra_meta={}))
    report = "TG3K: official train/val list parsed from tg3k-trainval.json; val mapped to test."
    meta = {"dataset_name": "tg3k", "split_protocol": "official_trainval_json_val_to_test",
            "split_file": str(json_path),
            "train_cases": [x["case_id"] for x in items if x["split"] == "train"],
            "test_cases": [x["case_id"] for x in items if x["split"] == "test"]}
    return items, report, meta


def adapter_generic_2d_split(raw_ds_root: Path, dataset_name: str, train_n: int, seed: int):
    image_dirs = ["images", "image", "imgs", "img", "Original", "Images"]
    mask_dirs = ["masks", "mask", "annotations", "gts", "GT", "Ground Truth"]
    img_dir = find_first(raw_ds_root, image_dirs, is_dir=True)
    gt_dir = find_first(raw_ds_root, mask_dirs, is_dir=True)
    if img_dir is None or gt_dir is None:
        raise FileNotFoundError(f"Could not infer image/mask directories for {dataset_name}")
    img_map = file_stem_map(list_immediate_files(img_dir))
    gt_map = file_stem_map(list_immediate_files(gt_dir))
    common = sorted(set(img_map) & set(gt_map))
    train_cases, test_cases = train_test_split_fixed(common, train_n, seed)
    items = []
    for cid in common:
        split = "train" if cid in train_cases else "test"
        items.append(CaseItem(dataset_name=dataset_name, case_id=cid, patient_id=cid, split=split,
                              modality="Endoscopy", task_dim="2D", class_info="binary",
                              image_src=img_map[cid], label_src=gt_map[cid], source_format="image", aux_src={}, extra_meta={}))
    report = f"{dataset_name}: inferred 2D image/mask folders; fixed split {len(train_cases)}/{len(test_cases)} with saved case lists."
    meta = {"dataset_name": dataset_name, "split_protocol": "fixed_count_split", "train_cases": train_cases, "test_cases": test_cases}
    return items, report, meta


def adapter_synapse(raw_ds_root: Path, seed: int):
    train_dir = raw_ds_root / "train_npz"
    test_dir = raw_ds_root / "test_vol_h5"
    items = []
    train_files = sorted([p for p in list_immediate_files(train_dir) if p.suffix.lower() == ".npz"])
    by_case = defaultdict(list)
    for p in train_files:
        m = re.match(r"(case\d+)_slice(\d+)", safe_stem(p))
        if not m:
            continue
        by_case[m.group(1)].append((int(m.group(2)), p))
    for cid, slices in sorted(by_case.items()):
        items.append(CaseItem(dataset_name="synapse", case_id=cid, patient_id=cid, split="train",
                              modality="CT", task_dim="3D", class_info="multiclass",
                              image_src=None, label_src=None, source_format="npz_slices",
                              aux_src={"slices": [str(p) for _, p in sorted(slices)]}, extra_meta={"num_slices": len(slices)}))
    test_files = sorted([p for p in list_immediate_files(test_dir) if p.name.lower().endswith(".h5")])
    for p in test_files:
        cid = safe_stem(p).replace(".npy", "")
        items.append(CaseItem(dataset_name="synapse", case_id=cid, patient_id=cid, split="test",
                              modality="CT", task_dim="3D", class_info="multiclass",
                              image_src=p, label_src=None, source_format="h5_volume", aux_src={}, extra_meta={}))
    report = "Synapse: do not redefine split; follow train_npz / test_vol_h5. Training uses slice-level npz grouped by case; testing keeps volume-level h5."
    meta = {"dataset_name": "synapse", "split_protocol": "official_train_npz_test_vol_h5",
            "train_cases": [x["case_id"] for x in items if x["split"] == "train"],
            "test_cases": [x["case_id"] for x in items if x["split"] == "test"]}
    return items, report, meta


def adapter_acdc(raw_ds_root: Path, seed: int):
    items = []
    for split, split_dir in [("train", raw_ds_root / "training"), ("test", raw_ds_root / "testing")]:
        for patient_dir in sorted([p for p in split_dir.iterdir() if p.is_dir() and p.name.lower().startswith("patient")]):
            patient_id = patient_dir.name
            files = list_immediate_files(patient_dir)
            gt_map = {safe_stem(p).replace("_gt", ""): p for p in files if p.name.endswith("_gt.nii.gz")}
            cine4d = patient_dir / f"{patient_id}_4d.nii.gz"
            for ip in sorted([p for p in files if p.name.endswith(".nii.gz") and "_gt" not in p.name and "_4d" not in p.name]):
                stem = safe_stem(ip)
                gp = gt_map.get(stem)
                if gp is None:
                    continue
                aux = {}
                if cine4d.exists():
                    aux["cine_4d"] = cine4d
                items.append(CaseItem(dataset_name="acdc", case_id=stem, patient_id=patient_id, split=split,
                                      modality="Cine MRI", task_dim="3D", class_info="multiclass",
                                      image_src=ip, label_src=gp, source_format="nii.gz", aux_src=aux, extra_meta={}))
    report = "ACDC: use official training/testing split. Each labeled frame (patientXXX_frameYY) is treated as a 3D case; patientXXX_4d is kept only as auxiliary reference."
    meta = {"dataset_name": "acdc", "split_protocol": "official_challenge_split",
            "train_cases": [x["case_id"] for x in items if x["split"] == "train"],
            "test_cases": [x["case_id"] for x in items if x["split"] == "test"]}
    return items, report, meta


def adapter_btcv(raw_ds_root: Path, seed: int):
    img_dir = raw_ds_root / "images"
    gt_dir = raw_ds_root / "labels"
    img_map = {safe_stem(p): p for p in list_immediate_files(img_dir)}
    gt_map = {}
    for p in list_immediate_files(gt_dir):
        gt_map[safe_stem(p).replace("label", "img")] = p
    common = sorted(set(img_map) & set(gt_map))
    train_cases, test_cases = train_test_split_fixed(common, 24, seed)
    items = []
    for cid in common:
        items.append(CaseItem(dataset_name="btcv", case_id=cid, patient_id=cid,
                              split="train" if cid in train_cases else "test",
                              modality="CT", task_dim="3D", class_info="multiclass",
                              image_src=img_map[cid], label_src=gt_map[cid], source_format="nii.gz", aux_src={}, extra_meta={}))
    report = "BTCV: public labeled subset organized at case level. Fixed internal 24/6 split on the 30 labeled volumes after sorting and seeded shuffling; exact case lists are saved for reproducibility."
    meta = {"dataset_name": "btcv", "split_protocol": "internal_fixed_case_split_24_6", "seed": seed,
            "train_cases": train_cases, "test_cases": test_cases}
    return items, report, meta


def adapter_chasedb1(raw_ds_root: Path, seed: int):
    files = list_immediate_files(raw_ds_root)
    base_map, gt1_map, gt2_map = {}, {}, {}
    for p in files:
        stem = safe_stem(p)
        if stem.endswith("_1stHO"):
            gt1_map[stem[:-6]] = p
        elif stem.endswith("_2ndHO"):
            gt2_map[stem[:-6]] = p
        else:
            base_map[stem] = p
    patient_bases = defaultdict(list)
    for stem in base_map:
        m = re.match(r"Image_(\d+)([LR])$", stem, re.IGNORECASE)
        if m:
            pid = m.group(1).zfill(2)
            patient_bases[pid].append(stem)
    train_pids = [f"{i:02d}" for i in range(1, 11)]
    test_pids = [f"{i:02d}" for i in range(11, 15)]
    items = []
    for pid, stems in sorted(patient_bases.items()):
        split = "train" if pid in train_pids else "test"
        for stem in sorted(stems):
            if stem not in gt1_map:
                continue
            aux = {}
            if stem in gt2_map:
                aux["reader2_label"] = gt2_map[stem]
            items.append(CaseItem(dataset_name="chasedb1", case_id=stem, patient_id=pid, split=split,
                                  modality="Color Fundus", task_dim="2D", class_info="binary",
                                  image_src=base_map[stem], label_src=gt1_map[stem], source_format="image",
                                  aux_src=aux, extra_meta={"laterality": stem[-1].upper()}))
    report = "CHASEDB1: use Image_XXL/Image_XXR as images; 1stHO as main vessel GT; 2ndHO stored as auxiliary label. Strict patient-level split: first 10 patients for train, last 4 patients for test."
    meta = {"dataset_name": "chasedb1", "split_protocol": "strict_patient_level_10_4",
            "train_patients": train_pids, "test_patients": test_pids,
            "train_cases": [x["case_id"] for x in items if x["split"] == "train"],
            "test_cases": [x["case_id"] for x in items if x["split"] == "test"]}
    return items, report, meta


def adapter_ddti(raw_ds_root: Path, seed: int):
    img_dir = raw_ds_root / "image"
    gt_dir = raw_ds_root / "mask"
    meta_file = None
    for cand in ["category.csv", "category.xlsx", "category.xls", "category"]:
        p = raw_ds_root / cand
        if p.exists():
            meta_file = p
            break
    if meta_file is None:
        raise FileNotFoundError("DDTI category metadata not found")
    df = read_table(meta_file)
    cols = {str(c).lower(): c for c in df.columns}
    id_col = cols.get("id", df.columns[0])
    cate_col = cols.get("cate", df.columns[1])
    label_map = {}
    for _, row in df.iterrows():
        rid = str(row[id_col]).strip()
        rid = safe_stem(Path(rid))
        label_map[rid] = str(row[cate_col]).strip()
    img_map = file_stem_map(list_immediate_files(img_dir))
    gt_map = file_stem_map(list_immediate_files(gt_dir))
    common = sorted(set(img_map) & set(gt_map) & set(label_map))
    train_cases, test_cases = stratified_split({c: label_map[c] for c in common}, 0.8, seed)
    items = []
    for cid in common:
        items.append(CaseItem(dataset_name="ddti", case_id=cid, patient_id=cid,
                              split="train" if cid in train_cases else "test",
                              modality="Ultrasound", task_dim="2D", class_info="binary",
                              image_src=img_map[cid], label_src=gt_map[cid], source_format="image", aux_src={},
                              pathology_label=label_map[cid], extra_meta={}))
    report = "DDTI: pair image/mask by numeric stem; parse category metadata and perform strict 8:2 stratified split by pathology label."
    meta = {"dataset_name": "ddti", "split_protocol": "stratified_8_2_by_category", "seed": seed,
            "train_cases": train_cases, "test_cases": test_cases}
    return items, report, meta


def adapter_drive(raw_ds_root: Path, seed: int):
    items = []
    for split, split_dir in [("train", raw_ds_root / "training"), ("test", raw_ds_root / "test")]:
        img_dir = split_dir / "images"
        gt_dir = split_dir / "1st_manual"
        fov_dir = split_dir / "mask"
        img_map = file_stem_map(list_immediate_files(img_dir))
        gt_map, fov_map = {}, {}
        for p in list_immediate_files(gt_dir):
            m = re.match(r"(\d+)_manual1", safe_stem(p))
            if m:
                num = m.group(1)
                cand = [k for k in img_map if k.startswith(num + "_")]
                if len(cand) == 1:
                    gt_map[cand[0]] = p
        for p in list_immediate_files(fov_dir):
            m = re.match(r"(\d+)", safe_stem(p))
            if m:
                cand = [k for k in img_map if k.startswith(m.group(1) + "_")]
                if len(cand) == 1:
                    fov_map[cand[0]] = p
        for cid, ip in sorted(img_map.items()):
            gp = gt_map.get(cid)
            if gp is None:
                continue
            aux = {}
            if cid in fov_map:
                aux["fov_mask"] = fov_map[cid]
            items.append(CaseItem(dataset_name="drive", case_id=cid, patient_id=cid, split=split,
                                  modality="Color Fundus", task_dim="2D", class_info="binary",
                                  image_src=ip, label_src=gp, source_format="image", aux_src=aux, extra_meta={}))
    report = "DRIVE: follow official training/test split. 1st_manual is the main vessel GT; mask is stored as FOV mask for evaluation-time valid-region restriction."
    meta = {"dataset_name": "drive", "split_protocol": "official_fixed_split",
            "train_cases": [x["case_id"] for x in items if x["split"] == "train"],
            "test_cases": [x["case_id"] for x in items if x["split"] == "test"]}
    return items, report, meta


def adapter_hrf(raw_ds_root: Path, seed: int):
    img_dir = raw_ds_root / "images"
    gt_dir = raw_ds_root / "manual1"
    fov_dir = raw_ds_root / "mask"
    img_map = file_stem_map(list_immediate_files(img_dir))
    gt_map = file_stem_map(list_immediate_files(gt_dir))
    fov_map = {safe_stem(p).replace("_mask", ""): p for p in list_immediate_files(fov_dir)}
    common = sorted(set(img_map) & set(gt_map))
    pathology = {c: infer_pathology_from_suffix(c) for c in common}
    train_cases, test_cases = [], []
    rnd = random.Random(seed)
    for grp in ["h", "dr", "g"]:
        cases = sorted([c for c in common if pathology[c] == grp])
        rnd.shuffle(cases)
        train_cases.extend(cases[:10])
        test_cases.extend(cases[10:])
    train_cases, test_cases = sorted(train_cases), sorted(test_cases)
    items = []
    for cid in common:
        aux = {}
        if cid in fov_map:
            aux["fov_mask"] = fov_map[cid]
        items.append(CaseItem(dataset_name="hrf", case_id=cid, patient_id=cid,
                              split="train" if cid in train_cases else "test",
                              modality="Color Fundus", task_dim="2D", class_info="binary",
                              image_src=img_map[cid], label_src=gt_map[cid], source_format="image", aux_src=aux,
                              pathology_label=pathology[cid], extra_meta={"pathology_group": pathology[cid]}))
    report = "HRF: image/manual1/mask are paired by stem; manual1 is main vessel GT and mask is FOV mask. Strict stratified split by suffix groups (_h, _dr, _g): 10 train and 5 test per class."
    meta = {"dataset_name": "hrf", "split_protocol": "stratified_10_5_per_group", "seed": seed,
            "train_cases": train_cases, "test_cases": test_cases}
    return items, report, meta


def adapter_monuseg(raw_ds_root: Path, seed: int):
    items = []

    def parse_split(split_name: str, split_dir: Path):
        local = []
        if not split_dir.exists():
            return local
        tissue_dir = split_dir / "Tissue Images"
        ann_dir = split_dir / "Annotations"
        if tissue_dir.exists() and ann_dir.exists():
            img_files = [p for p in list_immediate_files(tissue_dir) if p.suffix.lower() in IMG_EXTS_2D]
            xml_map = {safe_stem(p): p for p in list_immediate_files(ann_dir) if p.suffix.lower() == ".xml"}
            for ip in sorted(img_files):
                cid = safe_stem(ip)
                xp = xml_map.get(cid)
                if xp is None:
                    continue
                local.append(CaseItem(dataset_name="monuseg", case_id=cid, patient_id=cid, split=split_name,
                                      modality="Histopathology", task_dim="2D", class_info="binary",
                                      image_src=ip, label_src=xp, source_format="image+xml", aux_src={}, extra_meta={}))
        else:
            files = list_immediate_files(split_dir)
            img_map = {safe_stem(p): p for p in files if p.suffix.lower() in IMG_EXTS_2D}
            xml_map = {safe_stem(p): p for p in files if p.suffix.lower() == ".xml"}
            for cid, ip in sorted(img_map.items()):
                xp = xml_map.get(cid)
                if xp is None:
                    continue
                local.append(CaseItem(dataset_name="monuseg", case_id=cid, patient_id=cid, split=split_name,
                                      modality="Histopathology", task_dim="2D", class_info="binary",
                                      image_src=ip, label_src=xp, source_format="image+xml", aux_src={}, extra_meta={}))
        return local

    items.extend(parse_split("train", raw_ds_root / "train"))
    items.extend(parse_split("test", raw_ds_root / "test"))
    report = "MoNuSeg: follow current train/test folder split. XML/polygon annotations are rasterized into binary nuclei masks for semantic segmentation-style training."
    meta = {"dataset_name": "monuseg", "split_protocol": "given_train_test_with_xml_to_binary",
            "train_cases": [x["case_id"] for x in items if x["split"] == "train"],
            "test_cases": [x["case_id"] for x in items if x["split"] == "test"]}
    return items, report, meta


def adapter_otu2d(raw_ds_root: Path, seed: int):
    img_dir = raw_ds_root / "images"
    gt_dir = raw_ds_root / "annotations"
    train_file = find_first(raw_ds_root, ["train", "train.txt"], is_dir=False)
    val_file = find_first(raw_ds_root, ["val", "val.txt"], is_dir=False)
    if train_file is None or val_file is None:
        raise FileNotFoundError("OTU_2d requires train and val list files")
    train_ids = set(parse_simple_list_file(train_file))
    val_ids = set(parse_simple_list_file(val_file))
    img_map = file_stem_map(list_immediate_files(img_dir))
    gt_map = {}
    for p in list_immediate_files(gt_dir):
        stem = safe_stem(p)
        if stem.endswith("_binary") and not stem.endswith("_binary_binary"):
            gt_map[stem[:-7]] = p
    common = sorted(set(img_map) & set(gt_map) & (train_ids | val_ids))
    items = []
    for cid in common:
        items.append(CaseItem(dataset_name="otu_2d", case_id=cid, patient_id=cid,
                              split="train" if cid in train_ids else "test",
                              modality="Ultrasound", task_dim="2D", class_info="binary",
                              image_src=img_map[cid], label_src=gt_map[cid], source_format="image", aux_src={}, extra_meta={}))
    report = "OTU_2d/MMOTU: only annotations/<id>_binary is used as the legal GT; raw labels and *_binary_binary are ignored. Official train/val lists are mapped to train/test."
    meta = {"dataset_name": "otu_2d", "split_protocol": "official_train_val_lists_val_to_test",
            "train_cases": sorted(train_ids & set(common)), "test_cases": sorted(val_ids & set(common))}
    return items, report, meta


def adapter_ph2(raw_ds_root: Path, seed: int):
    img_root = find_first(raw_ds_root, ["PH2 Dataset images", "PH2_Dataset_images"], is_dir=True)
    if img_root is None:
        raise FileNotFoundError("PH2 Dataset images folder not found")
    pathology_map = maybe_read_ph2_metadata(raw_ds_root)
    items = []
    for case_dir in sorted([p for p in img_root.iterdir() if p.is_dir()]):
        cid = case_dir.name
        derm_dir = find_first(case_dir, [f"{cid}_Dermoscopic_Image"], is_dir=True)
        lesion_dir = find_first(case_dir, [f"{cid}_lesion"], is_dir=True)
        roi_dir = find_first(case_dir, [f"{cid}_roi"], is_dir=True)
        if derm_dir is None or lesion_dir is None:
            continue
        ip = choose_one(derm_dir)
        gp = choose_one(lesion_dir)
        aux = {}
        if roi_dir is not None:
            rp = choose_one(roi_dir)
            if rp:
                aux["roi"] = rp
        items.append(CaseItem(dataset_name="ph2", case_id=cid, patient_id=cid, split="unspecified",
                              modality="Dermoscopy", task_dim="2D", class_info="binary",
                              image_src=ip, label_src=gp, source_format="image", aux_src=aux,
                              pathology_label=pathology_map.get(cid), extra_meta={}))
    case_to_lab = {x["case_id"]: x.get("pathology_label") for x in items if x.get("pathology_label")}
    if len(case_to_lab) >= 150:
        train_cases, test_cases = stratified_split(case_to_lab, 0.8, seed)
    else:
        all_cases = sorted([x["case_id"] for x in items])
        train_cases, test_cases = train_test_split_fixed(all_cases, 160, seed)
    for x in items:
        x["split"] = "train" if x["case_id"] in train_cases else "test"
    report = "PH2: use Dermoscopic_Image as image, lesion as main GT, roi as auxiliary path. Metadata-derived pathology labels are used for strict 8:2 stratified splitting when available."
    meta = {"dataset_name": "ph2", "split_protocol": "stratified_8_2_by_metadata_if_available", "seed": seed,
            "train_cases": train_cases, "test_cases": test_cases}
    return items, report, meta


def adapter_prostate158(raw_ds_root: Path, seed: int) -> Tuple[List[CaseItem], str, Dict]:
    items = []

    split_specs = [
        ("train", raw_ds_root / "train" / "train"),
        ("test",  raw_ds_root / "test" / "prostate158_test" / "test"),
    ]

    for split, split_dir in split_specs:
        if not split_dir.exists():
            raise FileNotFoundError(f"Prostate158 split dir not found: {split_dir}")

        for case_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            cid = case_dir.name

            t2 = case_dir / "t2.nii.gz"
            main_gt = case_dir / "t2_anatomy_reader1.nii.gz"

            if not t2.exists() or not main_gt.exists():
                continue

            aux = {}
            for name in [
                "adc.nii.gz",
                "dwi.nii.gz",
                "t2_anatomy_reader2.nii.gz",
                "t2_tumor_reader1.nii.gz",
                "adc_tumor_reader1.nii.gz",
                "adc_tumor_reader2.nii.gz",
                "empty.nii.gz",
            ]:
                p = case_dir / name
                if p.exists():
                    aux[safe_stem(p)] = p

            items.append(CaseItem(
                dataset_name="prostate158",
                case_id=cid,
                patient_id=cid,
                split=split,
                modality="MRI",
                task_dim="3D",
                class_info="multiclass",
                image_src=t2,
                label_src=main_gt,
                source_format="nii.gz",
                aux_src=aux,
                extra_meta={
                    "main_modality": "t2",
                    "main_gt": "t2_anatomy_reader1"
                }
            ))

    report = (
        "Prostate158: follow provided fixed split; "
        "train cases are read from train/train, test cases from test/prostate158_test/test. "
        "Current baseline uses single-modality T2 anatomy segmentation with "
        "t2_anatomy_reader1 as the primary GT; other modalities and reader labels are stored under aux/."
    )
    split_meta = {
        "dataset_name": "prostate158",
        "split_protocol": "official_fixed_split_single_modality_t2_reader1",
        "train_cases": [x["case_id"] for x in items if x["split"] == "train"],
        "test_cases": [x["case_id"] for x in items if x["split"] == "test"],
    }
    return items, report, split_meta

ADAPTERS = {
    "tn3k": adapter_tn3k,
    "tg3k": adapter_tg3k,
    "kvasir": lambda r, s: adapter_generic_2d_split(r, "kvasirseg", 900, s),
    "kvasirseg": lambda r, s: adapter_generic_2d_split(r, "kvasirseg", 900, s),
    "kvasir_seg": lambda r, s: adapter_generic_2d_split(r, "kvasirseg", 900, s),
    "cvc_clinicdb": adapter_cvc_clinicdb,
    "cvc_clinic_db": adapter_cvc_clinicdb,
    "synapse": adapter_synapse,
    "acdc": adapter_acdc,
    "btcv": adapter_btcv,
    "chasedb1": adapter_chasedb1,
    "chase_db1": adapter_chasedb1,
    "ddti": adapter_ddti,
    "drive": adapter_drive,
    "hrf": adapter_hrf,
    "monuseg": adapter_monuseg,
    "otu_2d": adapter_otu2d,
    "mmotu": adapter_otu2d,
    "ph2": adapter_ph2,
    "prostate158": adapter_prostate158,
}


# -----------------------------
# Organization
# -----------------------------
def build_sanity_rows(items: List[CaseItem]) -> List[Dict]:
    rows = []
    seen = set()
    for x in items:
        key = (x["split"], x["case_id"])
        rows.append({
            "dataset_name": x["dataset_name"],
            "case_id": x["case_id"],
            "patient_id": x["patient_id"],
            "split": x["split"],
            "image_exists": bool(x.get("image_src")) and Path(str(x["image_src"])).exists(),
            "label_exists": True if x.get("label_src") is None else Path(str(x["label_src"])).exists(),
            "duplicate_case_within_split": key in seen,
            "pathology_label": x.get("pathology_label"),
            "source_format": x.get("source_format"),
        })
        seen.add(key)
    return rows


def save_case(case: CaseItem, case_dir: Path, link_mode: str = "copy"):
    ensure_dir(case_dir)
    ensure_dir(case_dir / "aux")
    fmt = case["source_format"]
    label_processing: Dict[str, Any] = {}

    if fmt == "image":
        save_image_as_png(Path(case["image_src"]), case_dir / "image.png", is_mask=False)
        label_processing = save_image_as_png(
            Path(case["label_src"]),
            case_dir / "label.png",
            is_mask=True,
            dataset_name=case["dataset_name"],
            mask_role="label",
        ) or {}
        image_name, label_name = "image.png", "label.png"
    elif fmt == "image+xml":
        ip = Path(case["image_src"])
        if ip.suffix.lower() in {".tif", ".tiff"}:
            copy_or_link(ip, case_dir / "image.tif", mode=link_mode)
            image_name = "image.tif"
        else:
            save_image_as_png(ip, case_dir / "image.png", is_mask=False)
            image_name = "image.png"
        mask = xml_to_binary_mask(Path(case["label_src"]), ip)
        save_binary_mask_from_array(mask, case_dir / "label.png")
        label_name = "label.png"
    elif fmt == "nii.gz":
        copy_or_link(Path(case["image_src"]), case_dir / "image.nii.gz", mode=link_mode)
        copy_or_link(Path(case["label_src"]), case_dir / "label.nii.gz", mode=link_mode)
        image_name, label_name = "image.nii.gz", "label.nii.gz"
    elif fmt == "npz_slices":
        slice_dir = case_dir / "aux" / "slices"
        ensure_dir(slice_dir)
        for sp in case["aux_src"]["slices"]:
            copy_or_link(Path(sp), slice_dir / Path(sp).name, mode=link_mode)
        image_name, label_name = None, None
    elif fmt == "h5_volume":
        copy_or_link(Path(case["image_src"]), case_dir / "image.h5", mode=link_mode)
        image_name, label_name = "image.h5", None
    else:
        raise ValueError(f"Unsupported source_format: {fmt}")

    aux_files = {}
    for k, v in case.get("aux_src", {}).items():
        if fmt == "npz_slices" and k == "slices":
            aux_files["slices_dir"] = "aux/slices"
            continue
        vp = Path(str(v))
        ext = suffix_for_copy(vp)
        dst = case_dir / "aux" / f"{k}{ext}"
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            if ("mask" in k) or ("label" in k) or ("roi" in k):
                out = case_dir / "aux" / f"{k}.png"
                save_image_as_png(vp, out, is_mask=True, dataset_name=case["dataset_name"], mask_role="aux")
                aux_files[k] = f"aux/{out.name}"
            else:
                if ext in {".tif", ".tiff"}:
                    copy_or_link(vp, dst, mode=link_mode)
                    aux_files[k] = f"aux/{dst.name}"
                else:
                    out = case_dir / "aux" / f"{k}.png"
                    save_image_as_png(vp, out, is_mask=False)
                    aux_files[k] = f"aux/{out.name}"
        else:
            copy_or_link(vp, dst, mode=link_mode)
            aux_files[k] = f"aux/{dst.name}"

    meta = {
        "dataset_name": case["dataset_name"],
        "case_id": case["case_id"],
        "patient_id": case["patient_id"],
        "split": case["split"],
        "modality": case["modality"],
        "task_dim": case["task_dim"],
        "class_info": case["class_info"],
        "source_format": fmt,
        "image_file": image_name,
        "label_file": label_name,
        "aux_files": aux_files,
        "pathology_label": case.get("pathology_label"),
        "extra_meta": case.get("extra_meta", {}),
        "label_processing": label_processing,
    }
    write_json(meta, case_dir / "case_meta.json")
    return meta


def organize_one_dataset(raw_root: Path, organized_root: Path, dataset_arg: str, seed: int, link_mode: str = "copy"):
    ds_key = norm_name(dataset_arg)
    if ds_key not in ADAPTERS:
        raise KeyError(f"Unsupported dataset adapter: {dataset_arg}")

    raw_ds_root = raw_root / dataset_arg
    if not raw_ds_root.exists():
        candidates = {norm_name(p.name): p for p in raw_root.iterdir() if p.is_dir()}
        if ds_key not in candidates:
            raise FileNotFoundError(f"Dataset directory not found under raw root: {dataset_arg}")
        raw_ds_root = candidates[ds_key]

    items, report, split_meta = ADAPTERS[ds_key](raw_ds_root, seed)
    out_ds_name = split_meta["dataset_name"]
    ds_out = organized_root / out_ds_name
    meta_dir = ds_out / "meta"
    train_cases_dir = ds_out / "train" / "cases"
    test_cases_dir = ds_out / "test" / "cases"
    ensure_dir(meta_dir)
    ensure_dir(train_cases_dir)
    ensure_dir(test_cases_dir)

    serializable = []
    for x in items:
        y = dict(x)
        if y.get("image_src") is not None:
            y["image_src"] = str(y["image_src"])
        if y.get("label_src") is not None:
            y["label_src"] = str(y["label_src"])
        aux2 = {}
        for k, v in y.get("aux_src", {}).items():
            aux2[k] = [str(t) for t in v] if isinstance(v, list) else str(v)
        y["aux_src"] = aux2
        serializable.append(y)

    write_text(report + "\n", meta_dir / "adapter_report.txt")
    write_json(serializable, meta_dir / "raw_manifest.json")
    write_csv(build_sanity_rows(items), meta_dir / "sanity_check.csv")
    write_json(split_meta, meta_dir / "split_meta.json")
    write_json(split_meta.get("train_cases", []), meta_dir / "train_cases.json")
    write_json(split_meta.get("test_cases", []), meta_dir / "test_cases.json")

    train_index, test_index = [], []
    label_audits: List[Dict[str, Any]] = []
    for x in items:
        split = x["split"]
        case_id = sanitize_case_id(x["case_id"])
        case_dir = (train_cases_dir if split == "train" else test_cases_dir) / case_id
        case_meta = save_case(x, case_dir, link_mode=link_mode)
        rel = f"{split}/cases/{case_id}"
        record = {"case_id": case_id, "patient_id": x["patient_id"], "case_dir": rel}
        if split == "train":
            train_index.append(record)
        elif split == "test":
            test_index.append(record)
        label_processing = dict(case_meta.get("label_processing") or {})
        if label_processing:
            label_processing.update({"case_id": case_id, "split": split})
            label_audits.append(label_processing)

    write_json(train_index, ds_out / "train" / "index.json")
    write_json(test_index, ds_out / "test" / "index.json")
    if label_audits:
        write_json(label_audits, meta_dir / "label_source_audit.json")
        write_csv(label_audits, meta_dir / "label_source_audit.csv")
        summary = {
            "dataset_name": out_ds_name,
            "num_cases": len(label_audits),
            "binarize_modes": sorted({str(x.get("binarize_mode")) for x in label_audits}),
            "cleanup_modes": sorted({str(x.get("cleanup_mode")) for x in label_audits if x.get("cleanup_mode")}),
            "num_cases_with_removed_components": int(sum(bool(x.get("removed_components")) for x in label_audits)),
            "max_source_unique_count": int(max(int(x.get("source_unique_count", 0)) for x in label_audits)),
        }
        write_json(summary, meta_dir / "label_source_audit_summary.json")

    return {"dataset": out_ds_name, "num_train": len(train_index), "num_test": len(test_index), "output_dir": str(ds_out)}


def resolve_dataset_names(raw_root: Path, datasets_arg: str) -> List[str]:
    if datasets_arg.strip().lower() == "all":
        return sorted([p.name for p in raw_root.iterdir() if p.is_dir()])
    return [x.strip() for x in datasets_arg.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", type=str, required=True, help="Root containing raw dataset folders")
    parser.add_argument("--organized_root", type=str, required=True, help="Root to write organized datasets")
    parser.add_argument("--datasets", type=str, default="all", help="Comma separated dataset names or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link_mode", type=str, default="copy", choices=["copy", "link"], help="Use hardlink when possible")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    organized_root = Path(args.organized_root)
    ensure_dir(organized_root)

    requested = resolve_dataset_names(raw_root, args.datasets)
    alias_map = {
        "kvasir_seg": "Kvasir", "kvasirseg": "Kvasir", "kvasir": "Kvasir",
        "cvc_clinicdb": "CVC-ClinicDB", "cvc-clinicdb": "CVC-ClinicDB",
        "chase_db1": "CHASEDB1", "chasedb1": "CHASEDB1",
        "otu_2d": "OTU_2d", "mmotu": "OTU_2d", "tg3k": "tg3k", "tn3k": "tn3k",
        "monuseg": "MoNuSeg", "prostate158": "Prostate158", "synapse": "Synapse",
        "btcv": "BTCV", "acdc": "ACDC", "ddti": "DDTI", "drive": "DRIVE", "hrf": "HRF", "ph2": "PH2",
    }
    folder_lookup = {norm_name(p.name): p.name for p in raw_root.iterdir() if p.is_dir()}
    todo = []
    for name in requested:
        key = norm_name(name)
        if key in folder_lookup:
            todo.append(folder_lookup[key])
        elif key in alias_map and norm_name(alias_map[key]) in folder_lookup:
            todo.append(folder_lookup[norm_name(alias_map[key])])
        elif key in alias_map:
            todo.append(alias_map[key])
        else:
            print(f"[WARN] skip unknown dataset request: {name}", file=sys.stderr)

    summary = []
    for ds in todo:
        print(f"\n[INFO] Organizing dataset: {ds}")
        try:
            result = organize_one_dataset(raw_root=raw_root, organized_root=organized_root, dataset_arg=ds,
                                          seed=args.seed, link_mode=args.link_mode)
            print(f"[OK] {result['dataset']}: train={result['num_train']} test={result['num_test']} -> {result['output_dir']}")
            summary.append(result)
        except Exception as e:
            print(f"[ERROR] {ds}: {repr(e)}", file=sys.stderr)

    write_json(summary, organized_root / "_summary.json")
    print(f"\n[DONE] Summary written to {organized_root / '_summary.json'}")


if __name__ == "__main__":
    main()
