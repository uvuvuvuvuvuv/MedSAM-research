#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

from idea1_common import (
    METHOD_DEFAULT,
    RoundPaths,
    atomic_save_json,
    build_case_to_slices,
    current_timestamp,
    full_and_box_from_selection,
    get_slice_name,
    infer_is_3d,
    load_json,
    load_manifest,
    load_selection,
    load_split_meta,
    read_csv,
    safe_float,
    train_items,
    validate_partition,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Select hard Full samples for the next round.")
    parser.add_argument("--fold_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--iou_threshold", type=float, default=0.5)
    parser.add_argument("--add_2d", type=int, default=5)
    parser.add_argument("--max_full_2d", type=int, default=20)
    parser.add_argument("--add_3d", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    current_paths = RoundPaths(args.fold_root, args.method, args.round)
    current_selection_path = args.selection or (current_paths.selection_dir / "selection.json")
    current = load_selection(current_selection_path)
    manifest = load_manifest(args.fold_root)
    split_meta = load_split_meta(args.fold_root)
    is_3d = infer_is_3d(args.dataset, split_meta)
    train = train_items(manifest)
    train_names = {get_slice_name(x) for x in train}
    full_names, box_names, full_cases = full_and_box_from_selection(current, manifest, is_3d)

    budget_path = args.fold_root / "meta" / f"annotation_budget_{args.method}.json"
    budget_obj = load_json(budget_path)
    max_full = int(budget_obj["max_full"])
    stop = False
    stop_reason = None
    selected_new: list[str] = []

    if is_3d:
        if len(full_cases) >= max_full:
            stop, stop_reason = True, "budget_reached"
        else:
            metrics_path = current_paths.diagnosis_dir / "per_case_metrics.csv"
            rows = read_csv(metrics_path)
            candidates = []
            for row in rows:
                case_id = str(row["case_id"])
                if case_id in full_cases:
                    continue
                macro = safe_float(row.get("case_macro_3d_iou"), 1.0)
                missed = int(float(row.get("empty_prediction_classes", 0) or 0))
                worst20 = safe_float(row.get("worst20_slice_class_iou"), 1.0)
                if macro < args.iou_threshold or missed > 0:
                    candidates.append((macro, -missed, worst20, case_id))
            candidates.sort()
            capacity = max_full - len(full_cases)
            count = min(args.add_3d, capacity, len(candidates))
            selected_new = [x[3] for x in candidates[:count]]
            if not selected_new:
                stop, stop_reason = True, "no_hard_candidate"
    else:
        if len(full_names) >= min(args.max_full_2d, max_full):
            stop, stop_reason = True, "budget_reached"
        else:
            metrics_path = current_paths.diagnosis_dir / "per_image_metrics.csv"
            rows = read_csv(metrics_path)
            candidates = []
            for row in rows:
                name = str(row["slice_name"])
                if name in full_names:
                    continue
                macro = safe_float(row.get("image_macro_iou"), 1.0)
                missed = int(float(row.get("empty_instance_count", 0) or 0))
                if macro < args.iou_threshold:
                    candidates.append((macro, -missed, name))
            candidates.sort()
            actual_max = min(args.max_full_2d, max_full)
            capacity = actual_max - len(full_names)
            count = min(args.add_2d, capacity, len(candidates))
            selected_new = [x[2] for x in candidates[:count]]
            if not selected_new:
                stop, stop_reason = True, "no_hard_candidate"

    result = {
        "dataset": args.dataset,
        "fold": current.get("fold", "fold_0"),
        "method": args.method,
        "current_round": args.round,
        "is_3d": is_3d,
        "selected_new_ids": selected_new,
        "stop": stop,
        "stop_reason": stop_reason,
        "current_full_slices": len(full_names),
        "current_full_cases": len(full_cases),
        "budget_max_full": max_full,
    }

    if not stop:
        next_round = args.round + 1
        next_paths = RoundPaths(args.fold_root, args.method, next_round)
        next_path = next_paths.selection_dir / "selection.json"
        if next_path.exists() and not args.overwrite:
            raise FileExistsError(next_path)
        if is_3d:
            case_to_slices = build_case_to_slices(train)
            next_full_cases = set(full_cases) | set(selected_new)
            next_full_names = {name for cid in next_full_cases for name in case_to_slices[cid]}
        else:
            next_full_cases = set()
            next_full_names = set(full_names) | set(selected_new)
        next_box_names = train_names - next_full_names
        validate_partition(train_names, next_full_names, next_box_names)
        payload = {
            "selection_version": "idea1_selection_v1",
            "created_at": current_timestamp(),
            "dataset": args.dataset,
            "fold": current.get("fold", "fold_0"),
            "method": args.method,
            "round": next_round,
            "selection_type": "hard_gt_iou",
            "source_round": args.round,
            "difficulty_iou_threshold": args.iou_threshold,
            "is_3d": is_3d,
            "selected_new_ids": selected_new,
            "cumulative_full_case_ids": sorted(next_full_cases),
            "cumulative_full_slice_names": sorted(next_full_names),
            "remaining_box_slice_names": sorted(next_box_names),
            "num_full_slices": len(next_full_names),
            "num_box_slices": len(next_box_names),
            "num_full_cases": len(next_full_cases),
        }
        atomic_save_json(payload, next_path)
        atomic_save_json(selected_new, next_paths.selection_dir / "selected_new_ids.json")
        atomic_save_json(sorted(next_full_names), next_paths.selection_dir / "cumulative_full_ids.json")
        atomic_save_json(sorted(next_box_names), next_paths.selection_dir / "remaining_box_ids.json")
        result["next_selection"] = str(next_path)

    atomic_save_json(result, current_paths.diagnosis_dir / "selection_result.json")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
