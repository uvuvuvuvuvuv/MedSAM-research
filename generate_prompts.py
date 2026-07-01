#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
generate_prompts.py

Stage 1.5 strict prompt generation for the MedSAM -> Swin-UMamba baseline.

Contract
--------
Input:
    processed/<dataset>/fold_0/
        meta/manifest.json
        meta/geometry_meta.json
        meta/split_meta.json
        native_npy/gts/*.npy

Output:
    processed/<dataset>/fold_0/prompts/prompts_train.json
    processed/<dataset>/fold_0/prompts/prompts_test.json

By default, the script writes only canonical prompt files under:
    processed/<dataset>/fold_0/prompts/

Root-level prompt aliases can be enabled explicitly with --write_root_alias for compatibility,
but the md-aligned formal baseline should use the canonical prompts/ directory only.

Prompt rule
-----------
1) bbox is generated from processed native_gt only.
2) bbox is defined in native space first.
3) bbox_teacher is obtained through geometry_meta[...]["native_to_teacher"].
4) no expansion, no margin, no jitter, no random perturbation.
5) binary / 2D foreground datasets: one connected component -> one bbox, label_id = 1.
6) multi-class datasets: each class connected component -> one bbox, label_id = class id.
7) empty slices are explicitly recorded with empty_slice = true and instances = [].

The resulting prompt JSON is directly compatible with the current strict box-constrained
Stage-2 pseudo-label generator, whose _parse_instances(...) accepts:
    instances[i]["bbox_teacher"]
    instances[i]["bbox_native"]
    instances[i]["label_id"]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

try:
    from utils.stage_timer_utils import StageTimer
except Exception:  # keeps the script usable outside the project package during syntax/debug checks
    class StageTimer:  # type: ignore
        def __init__(self, save_path: str, stage_name: str, dataset: str, fold: str, split: str):
            self.save_path = save_path
            self.stage_name = stage_name
            self.dataset = dataset
            self.fold = fold
            self.split = split
            self.outputs = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            ensure_dir(Path(self.save_path).parent)
            save_json({
                "stage_name": self.stage_name,
                "dataset": self.dataset,
                "fold": self.fold,
                "split": self.split,
                "num_outputs": self.outputs,
                "timer_fallback": True,
            }, Path(self.save_path))
            return False

        def set_outputs(self, n: int):
            self.outputs = int(n)



try:
    from utils.pipeline_progress import ProgressWriter
except Exception:
    try:
        from pipeline_progress import ProgressWriter
    except Exception:
        class ProgressWriter:  # type: ignore
            def __init__(self, *args, **kwargs):
                self.enabled = False
            def start(self, *args, **kwargs): pass
            def update(self, *args, **kwargs): pass
            def finish(self, *args, **kwargs): pass

DEFAULT_PROCESSED_ROOT = "/storage/baiyuting/data/MedSAM-main/data/processed"

# Baseline V1 scope: 3D4 + 2D7. If you still keep monuseg in a later run,
# pass --datasets ... ,monuseg explicitly.
DEFAULT_TARGET_DATASETS = [
    "btcv", "synapse", "acdc", "prostate158",
    "kvasirseg", "cvc_clinicdb", "tn3k", "tg3k", "ddti", "otu_2d", "ph2",
]

MULTICLASS_DATASETS = {"btcv", "synapse", "acdc", "prostate158"}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj: Any, path: Path):
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_list(s: str) -> List[str]:
    s = (s or "").strip()
    return [x.strip() for x in s.split(",") if x.strip()]


def infer_dataset_key(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def load_manifest(fold_root: Path) -> List[Dict[str, Any]]:
    path = fold_root / "meta" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"manifest.json not found: {path}")
    obj = load_json(path)
    if not isinstance(obj, list):
        raise ValueError(f"manifest.json must be a list: {path}")
    return obj


def load_geometry_meta(fold_root: Path) -> Dict[str, Any]:
    path = fold_root / "meta" / "geometry_meta.json"
    if not path.exists():
        raise FileNotFoundError(f"geometry_meta.json not found: {path}")
    obj = load_json(path)
    if not isinstance(obj, dict):
        raise ValueError(f"geometry_meta.json must be a dict: {path}")
    return obj


def load_split_meta(fold_root: Path) -> Dict[str, Any]:
    path = fold_root / "meta" / "split_meta.json"
    if not path.exists():
        return {}
    obj = load_json(path)
    return obj if isinstance(obj, dict) else {}


def infer_is_multiclass(dataset_name: str, split_meta: Dict[str, Any]) -> bool:
    if "multi_class_preserved" in split_meta:
        return bool(split_meta["multi_class_preserved"])
    return infer_dataset_key(dataset_name) in MULTICLASS_DATASETS


def load_native_gt(fold_root: Path, manifest_item: Dict[str, Any], slice_name: str) -> np.ndarray:
    rel = manifest_item.get("native_gt", None)
    candidates: List[Path] = []
    if rel:
        candidates.append(fold_root / str(rel))
    candidates.append(fold_root / "native_npy" / "gts" / slice_name)

    for path in candidates:
        if path.exists():
            gt = np.load(path)
            if gt.ndim == 3:
                gt = np.squeeze(gt)
                if gt.ndim == 3:
                    gt = gt[:, :, 0]
            if gt.ndim != 2:
                raise ValueError(f"native_gt must be 2D after squeeze, got shape={gt.shape}, path={path}")
            return gt.astype(np.uint8)

    raise FileNotFoundError(f"native_gt not found for {slice_name}; tried: {[str(x) for x in candidates]}")


def _clip_box_xyxy(box: Iterable[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0.0, min(float(w - 1), x1))
    x2 = max(0.0, min(float(w - 1), x2))
    y1 = max(0.0, min(float(h - 1), y1))
    y2 = max(0.0, min(float(h - 1), y2))

    # Keep MedSAM-compatible non-degenerate boxes.
    if x2 <= x1:
        x2 = min(float(w - 1), x1 + 1.0)
    if y2 <= y1:
        y2 = min(float(h - 1), y1 + 1.0)
    return [float(x1), float(y1), float(x2), float(y2)]


def bbox_from_mask(mask: np.ndarray) -> Optional[List[float]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def native_box_to_teacher_box(native_box: List[float], geom: Dict[str, Any]) -> List[float]:
    if "native_to_teacher" not in geom:
        raise KeyError("geometry_meta item missing native_to_teacher")
    g = geom["native_to_teacher"]

    scale_x = float(g.get("scale_x", g.get("scale", 1.0)))
    scale_y = float(g.get("scale_y", g.get("scale", 1.0)))
    offset_x = float(g.get("offset_x", 0.0))
    offset_y = float(g.get("offset_y", 0.0))
    target_h = int(g.get("target_h", geom.get("teacher_hw", [1024, 1024])[0]))
    target_w = int(g.get("target_w", geom.get("teacher_hw", [1024, 1024])[1]))

    x1, y1, x2, y2 = native_box
    teacher_box = [
        x1 * scale_x + offset_x,
        y1 * scale_y + offset_y,
        x2 * scale_x + offset_x,
        y2 * scale_y + offset_y,
    ]
    return _clip_box_xyxy(teacher_box, target_w, target_h)


def connected_components_for_label(mask: np.ndarray, connectivity: int = 8) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
    if connectivity not in {4, 8}:
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")
    return cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=connectivity)


def generate_instances_from_gt(
    gt_native: np.ndarray,
    geom: Dict[str, Any],
    is_multiclass: bool,
    min_component_area: int,
    connectivity: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    h, w = int(gt_native.shape[0]), int(gt_native.shape[1])
    instances: List[Dict[str, Any]] = []
    removed_small = 0
    removed_small_pixels = 0

    labels = [int(x) for x in np.unique(gt_native) if int(x) > 0 and int(x) != 255]
    if not labels:
        return [], {
            "num_labels": 0,
            "num_components": 0,
            "num_removed_small_components": 0,
            "num_removed_small_pixels": 0,
        }

    if not is_multiclass:
        labels_to_process = [1]
        masks_by_label = {1: (gt_native > 0) & (gt_native != 255)}
    else:
        labels_to_process = labels
        masks_by_label = {lb: (gt_native == lb) for lb in labels_to_process}

    for label_id in labels_to_process:
        binary = masks_by_label[label_id].astype(np.uint8)
        if binary.max() == 0:
            continue

        num_cc, cc, stats, centroids = connected_components_for_label(binary, connectivity=connectivity)
        for cc_id in range(1, num_cc):
            area = int(stats[cc_id, cv2.CC_STAT_AREA])
            if area < int(min_component_area):
                removed_small += 1
                removed_small_pixels += area
                continue

            component_mask = (cc == cc_id)
            native_box = bbox_from_mask(component_mask)
            if native_box is None:
                continue
            native_box = _clip_box_xyxy(native_box, w=w, h=h)
            teacher_box = native_box_to_teacher_box(native_box, geom)
            cx, cy = centroids[cc_id]

            instances.append({
                "label_id": int(label_id),
                "component_id": int(cc_id),
                "area_native": int(area),
                "bbox_native": [float(v) for v in native_box],
                "bbox_teacher": [float(v) for v in teacher_box],
                # Backward-compatible shortcut. Current pseudo script prefers bbox_teacher,
                # older variants often read bbox directly.
                "bbox": [float(v) for v in teacher_box],
                "centroid_native": [float(cx), float(cy)],
                "prompt_type": "tight_box",
            })

    instances.sort(key=lambda x: (int(x["label_id"]), int(x["component_id"])))
    return instances, {
        "num_labels": int(len(labels)),
        "num_components": int(len(instances)),
        "num_removed_small_components": int(removed_small),
        "num_removed_small_pixels": int(removed_small_pixels),
    }


def validate_manifest_geometry(manifest: List[Dict[str, Any]], geometry_meta: Dict[str, Any]):
    seen = set()
    duplicates = []
    missing_geom = []
    for item in manifest:
        name = str(item.get("slice_name", ""))
        if not name:
            raise ValueError("manifest item missing slice_name")
        if name in seen:
            duplicates.append(name)
        seen.add(name)
        if name not in geometry_meta:
            missing_geom.append(name)
    if duplicates:
        raise ValueError(f"duplicate slice_name in manifest: {duplicates[:20]}")
    if missing_geom:
        raise ValueError(f"slice_name missing in geometry_meta: {missing_geom[:20]}")


def write_prompt_file(
    fold_root: Path,
    split: str,
    prompts: Dict[str, Any],
    write_root_alias: bool,
    remove_stale_root: bool,
) -> Dict[str, str]:
    prompt_dir = fold_root / "prompts"
    ensure_dir(prompt_dir)
    canonical_path = prompt_dir / f"prompts_{split}.json"
    save_json(prompts, canonical_path)

    paths = {"canonical": str(canonical_path)}
    root_alias = fold_root / f"prompts_{split}.json"

    if write_root_alias:
        save_json(prompts, root_alias)
        paths["root_alias"] = str(root_alias)
    elif remove_stale_root and root_alias.exists():
        root_alias.unlink()
        paths["removed_stale_root_alias"] = str(root_alias)

    return paths


def process_one_dataset(
    processed_root: Path,
    dataset_name: str,
    fold_arg: str,
    splits: List[str],
    min_component_area: int,
    connectivity: int,
    write_root_alias: bool,
    remove_stale_root: bool,
    overwrite: bool,
    progress_root: str = "",
    progress_every: int = 25,
) -> Dict[str, Any]:
    dataset_root = processed_root / dataset_name
    if not dataset_root.exists():
        # tolerate case / separator differences
        ds_key = infer_dataset_key(dataset_name)
        candidates = {infer_dataset_key(p.name): p for p in processed_root.iterdir() if p.is_dir()}
        if ds_key in candidates:
            dataset_root = candidates[ds_key]
            dataset_name = dataset_root.name
        else:
            raise FileNotFoundError(f"processed dataset root not found: {dataset_root}")

    fold_roots = sorted([p for p in dataset_root.glob("fold_*") if p.is_dir()])
    if not fold_roots:
        raise FileNotFoundError(f"no fold_* directory found under {dataset_root}")
    if fold_arg != "all":
        fold_roots = [p for p in fold_roots if p.name == fold_arg]
        if not fold_roots:
            raise FileNotFoundError(f"{dataset_name}: fold not found: {fold_arg}")

    ds_summary: Dict[str, Any] = {
        "dataset": dataset_name,
        "folds": [],
        "status": "success",
    }

    for fold_root in fold_roots:
        manifest = load_manifest(fold_root)
        geometry_meta = load_geometry_meta(fold_root)
        split_meta = load_split_meta(fold_root)
        validate_manifest_geometry(manifest, geometry_meta)
        is_multiclass = infer_is_multiclass(dataset_name, split_meta)
        fold_name = fold_root.name

        with StageTimer(
            save_path=str(fold_root / "meta" / "stage_time_prompts.json"),
            stage_name="prompts",
            dataset=dataset_name,
            fold=fold_name,
            split=",".join(splits),
        ) as timer:
            fold_summary: Dict[str, Any] = {
                "fold": fold_name,
                "is_multiclass": bool(is_multiclass),
                "splits": {},
            }

            for split in splits:
                out_path = fold_root / "prompts" / f"prompts_{split}.json"
                root_path = fold_root / f"prompts_{split}.json"
                if (not overwrite) and out_path.exists() and ((not write_root_alias) or root_path.exists()):
                    fold_summary["splits"][split] = {
                        "status": "skipped_existing",
                        "canonical": str(out_path),
                        "root_alias": str(root_path) if root_path.exists() else "",
                    }
                    continue

                prompts: Dict[str, Any] = {}
                split_items = [x for x in manifest if str(x.get("split", "")) == split]
                stats = {
                    "num_manifest_items": int(len(split_items)),
                    "num_prompt_items": 0,
                    "num_empty_slices": 0,
                    "num_nonempty_slices": 0,
                    "num_instances": 0,
                    "num_removed_small_components": 0,
                    "num_removed_small_pixels": 0,
                    "num_errors": 0,
                    "instances_per_label": defaultdict(int),
                }
                errors: List[Dict[str, str]] = []
                progress_writer = ProgressWriter(
                    progress_root,
                    stage="prompts",
                    dataset=dataset_name,
                    fold=fold_name,
                    split=split,
                    enabled=bool(progress_root),
                )
                progress_writer.start(
                    total=len(split_items),
                    message="generating strict tight-box prompts",
                    stats={"num_manifest_items": int(len(split_items))},
                )
                progress_every_i = max(1, int(progress_every or 1))

                for progress_idx, item in enumerate(
                    tqdm(split_items, desc=f"prompts:{dataset_name}:{fold_name}:{split}", dynamic_ncols=True, ascii=True),
                    start=1,
                ):
                    slice_name = str(item["slice_name"])
                    try:
                        geom = geometry_meta[slice_name]
                        gt_native = load_native_gt(fold_root, item, slice_name)
                        instances, local_stats = generate_instances_from_gt(
                            gt_native=gt_native,
                            geom=geom,
                            is_multiclass=is_multiclass,
                            min_component_area=min_component_area,
                            connectivity=connectivity,
                        )

                        empty_slice = len(instances) == 0
                        prompt_item = {
                            "dataset_name": dataset_name,
                            "fold": fold_name,
                            "split": split,
                            "case_id": item.get("case_id", geom.get("case_id", "")),
                            "patient_id": item.get("patient_id", geom.get("patient_id", "")),
                            "slice_name": slice_name,
                            "slice_idx": item.get("slice_idx", geom.get("slice_idx", None)),
                            "empty_slice": bool(empty_slice),
                            "prompt_space": "native_first_then_teacher",
                            "bbox_policy": "tight_box_no_margin_no_jitter",
                            "component_policy": "per_class_component" if is_multiclass else "instance_component",
                            "instances": instances,
                        }
                        prompts[slice_name] = prompt_item

                        stats["num_prompt_items"] += 1
                        stats["num_instances"] += len(instances)
                        stats["num_removed_small_components"] += int(local_stats["num_removed_small_components"])
                        stats["num_removed_small_pixels"] += int(local_stats["num_removed_small_pixels"])
                        if empty_slice:
                            stats["num_empty_slices"] += 1
                        else:
                            stats["num_nonempty_slices"] += 1
                        for ins in instances:
                            stats["instances_per_label"][str(int(ins["label_id"]))] += 1

                    except Exception as e:
                        stats["num_errors"] += 1
                        errors.append({"slice_name": slice_name, "error": repr(e)})

                    if progress_idx == 1 or progress_idx % progress_every_i == 0 or progress_idx == len(split_items):
                        progress_writer.update(
                            current=progress_idx,
                            total=len(split_items),
                            message=f"processed {progress_idx}/{len(split_items)} prompt items",
                            stats={
                                "num_prompt_items": int(stats["num_prompt_items"]),
                                "num_instances": int(stats["num_instances"]),
                                "num_errors": int(stats["num_errors"]),
                            },
                        )

                # Convert defaultdict for JSON.
                stats["instances_per_label"] = dict(stats["instances_per_label"])

                paths = write_prompt_file(
                    fold_root=fold_root,
                    split=split,
                    prompts=prompts,
                    write_root_alias=write_root_alias,
                    remove_stale_root=remove_stale_root,
                )

                split_summary = {
                    "status": "success" if stats["num_errors"] == 0 else "success_with_errors",
                    "paths": paths,
                    "stats": stats,
                    "first_errors": errors[:20],
                }
                progress_writer.finish(
                    status="success" if stats["num_errors"] == 0 else "failed",
                    current=int(stats["num_prompt_items"]) + int(stats["num_errors"]),
                    total=len(split_items),
                    message="prompt generation finished",
                    stats={
                        "num_prompt_items": int(stats["num_prompt_items"]),
                        "num_instances": int(stats["num_instances"]),
                        "num_errors": int(stats["num_errors"]),
                    },
                )
                fold_summary["splits"][split] = split_summary
                timer.set_outputs(timer.outputs + int(stats["num_prompt_items"]) if hasattr(timer, "outputs") else int(stats["num_prompt_items"]))

                print(
                    f"[OK] prompts {dataset_name}/{fold_name}/{split}: "
                    f"items={stats['num_prompt_items']} nonempty={stats['num_nonempty_slices']} "
                    f"empty={stats['num_empty_slices']} instances={stats['num_instances']} "
                    f"errors={stats['num_errors']} -> {paths['canonical']}"
                )

            save_json(fold_summary, fold_root / "meta" / "prompt_generation_summary.json")
            ds_summary["folds"].append(fold_summary)

    return ds_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", type=str, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(DEFAULT_TARGET_DATASETS),
        help="comma-separated datasets; default is baseline V1 3D4+2D7",
    )
    parser.add_argument("--fold", type=str, default="all", help='fold_0/fold_1/... or "all"')
    parser.add_argument("--split", type=str, default="all", choices=["train", "test", "all"])
    parser.add_argument("--min_component_area", type=int, default=1, help="components smaller than this area are ignored; default keeps all components")
    parser.add_argument("--connectivity", type=int, default=8, choices=[4, 8])
    parser.add_argument("--overwrite", action="store_true", help="overwrite existing prompt json files")
    parser.add_argument(
        "--write_root_alias",
        action="store_true",
        default=False,
        help="also write fold_root/prompts_train.json and prompts_test.json for backward compatibility. Formal md-aligned baseline should leave this off.",
    )
    parser.add_argument(
        "--no_root_alias",
        action="store_true",
        help="do not write root prompt aliases; canonical prompts/ directory is still written",
    )
    parser.add_argument(
        "--remove_stale_root",
        action="store_true",
        help="when --no_root_alias is used, remove existing fold_root/prompts_*.json aliases to prevent stale reads",
    )
    parser.add_argument(
        "--progress_root",
        type=str,
        default="",
        help="Optional progress output root. For md-aligned formal runs, prefer data/vis/progress.",
    )
    parser.add_argument(
        "--progress_every",
        type=int,
        default=25,
        help="Update progress every N prompt items.",
    )
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    if not processed_root.exists():
        raise FileNotFoundError(f"processed_root not found: {processed_root}")

    datasets = parse_list(args.datasets)
    if not datasets:
        datasets = DEFAULT_TARGET_DATASETS

    splits = ["train", "test"] if args.split == "all" else [args.split]
    write_root_alias = bool(args.write_root_alias and not args.no_root_alias)

    print("Generating strict prompt JSONs with the following configuration:")
    print(f"   processed_root      = {processed_root}")
    print(f"   datasets            = {datasets}")
    print(f"   fold                = {args.fold}")
    print(f"   split               = {args.split}")
    print(f"   min_component_area  = {args.min_component_area}")
    print(f"   connectivity        = {args.connectivity}")
    print(f"   overwrite           = {args.overwrite}")
    print(f"   write_root_alias    = {write_root_alias}")
    print(f"   progress_root       = {args.progress_root if args.progress_root else '<disabled>'}")
    print(f"   progress_every      = {args.progress_every}")

    summary: List[Dict[str, Any]] = []
    for ds in datasets:
        try:
            summary.append(process_one_dataset(
                processed_root=processed_root,
                dataset_name=ds,
                fold_arg=args.fold,
                splits=splits,
                min_component_area=int(args.min_component_area),
                connectivity=int(args.connectivity),
                write_root_alias=write_root_alias,
                remove_stale_root=bool(args.remove_stale_root),
                overwrite=bool(args.overwrite),
                progress_root=(args.progress_root.strip() if args.progress_root else ""),
                progress_every=int(args.progress_every),
            ))
        except Exception as e:
            print(f"[ERROR] {ds}: {repr(e)}")
            traceback.print_exc()
            summary.append({
                "dataset": ds,
                "status": "failed",
                "error": repr(e),
            })

    save_json(summary, processed_root / "_prompt_generation_summary.json")
    print(f"[DONE ] prompt generation summary -> {processed_root / '_prompt_generation_summary.json'}")


if __name__ == "__main__":
    main()
