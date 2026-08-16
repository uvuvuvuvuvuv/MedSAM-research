#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    IGNORE_LABEL,
    METHOD_DEFAULT,
    RoundPaths,
    atomic_save_json,
    full_and_box_from_selection,
    infer_is_3d,
    load_manifest,
    load_npy_2d,
    load_prompts,
    load_selection,
    load_split_meta,
    manifest_index,
    resolve_path,
    train_items,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Idea1 hybrid supervision without touching frozen baseline.")
    parser.add_argument("--fold_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--final_round", type=int, required=True)
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--max_errors", type=int, default=20)
    args = parser.parse_args()

    paths = RoundPaths(args.fold_root, args.method, args.final_round)
    selection_path = args.selection or (paths.selection_dir / "selection.json")
    selection = load_selection(selection_path)
    manifest = load_manifest(args.fold_root)
    split_meta = load_split_meta(args.fold_root)
    is_3d = infer_is_3d(args.dataset, split_meta)
    full_names, box_names, full_cases = full_and_box_from_selection(selection, manifest, is_3d)
    mindex = manifest_index(manifest)
    train = train_items(manifest)
    target_dir = args.fold_root / "pseudo_student" / f"tri_train_{args.method}"
    boxonly_dir = args.fold_root / "pseudo_student" / "tri_train_boxonly"

    errors: list[str] = []
    total_error_count = 0

    def add_error(message: str) -> None:
        nonlocal total_error_count
        total_error_count += 1
        if len(errors) < args.max_errors:
            errors.append(message)

    counts = {"full": 0, "box": 0, "box_with_255": 0, "full_with_255": 0}
    files = {p.name for p in target_dir.glob("*.npy")}
    train_names = {Path(str(x.get("slice_name", ""))).name for x in train}
    if files != train_names:
        add_error(
            f"file set mismatch: missing={sorted(train_names-files)[:5]}, extra={sorted(files-train_names)[:5]}"
        )

    for name in sorted(train_names & files):
        label = load_npy_2d(target_dir / name).astype(np.int64)
        item = mindex[name]
        expected_hw = load_npy_2d(resolve_path(args.fold_root, item, "student_gt")).shape
        if tuple(label.shape) != tuple(expected_hw):
            add_error(f"shape mismatch {name}: label={label.shape}, expected_student_gt={expected_hw}")
            continue
        unique = set(int(x) for x in np.unique(label))
        if any(x < 0 or x > 255 for x in unique):
            add_error(f"invalid values {name}: {sorted(unique)}")
        if name in full_names:
            counts["full"] += 1
            if IGNORE_LABEL in unique:
                counts["full_with_255"] += 1
                add_error(f"Full label contains 255: {name}")
            gt = load_npy_2d(resolve_path(args.fold_root, item, "student_gt")).astype(np.int64)
            if not np.array_equal(label, gt):
                add_error(f"Full label is not exact student GT: {name}")
        else:
            counts["box"] += 1
            if IGNORE_LABEL in unique:
                counts["box_with_255"] += 1

            # Audit against the canonical box union already rasterized by the
            # frozen baseline generator in the same student space. Baseline
            # generation rasterizes native boxes first, builds the native
            # tri-map, and then applies nearest-neighbour native->student
            # mapping. Directly mapping continuous box coordinates to student
            # space and rasterizing again is not discretely equivalent at box
            # borders and creates false one-pixel "outside" reports.
            boxonly_path = boxonly_dir / name
            if not boxonly_path.is_file():
                add_error(f"boxonly reference file missing: {boxonly_path}")
                continue

            boxonly_label = load_npy_2d(boxonly_path).astype(np.int64)
            if tuple(boxonly_label.shape) != tuple(label.shape):
                add_error(
                    f"boxonly shape mismatch {name}: "
                    f"boxonly={boxonly_label.shape}, method={label.shape}"
                )
                continue

            boxonly_unique = set(int(x) for x in np.unique(boxonly_label))
            if any(x < 0 or x > 255 for x in boxonly_unique):
                add_error(f"invalid boxonly values {name}: {sorted(boxonly_unique)}")

            # In canonical baseline tri labels, the complete box union is
            # non-zero: confirmed foreground is 1..K and unconfirmed interior
            # is 255; all pixels outside the box union are exactly zero.
            canonical_union = boxonly_label != 0

            outside_unknown = (label == IGNORE_LABEL) & (~canonical_union)
            if outside_unknown.any():
                add_error(
                    f"255 outside canonical box union: {name}, "
                    f"pixels={int(outside_unknown.sum())}"
                )

            outside_foreground = (label > 0) & (label != IGNORE_LABEL) & (~canonical_union)
            if outside_foreground.any():
                add_error(
                    f"foreground outside canonical box union: {name}, "
                    f"pixels={int(outside_foreground.sum())}"
                )

    if not boxonly_dir.is_dir():
        add_error(f"boxonly reference missing: {boxonly_dir}")

    report = {
        "dataset": args.dataset,
        "method": args.method,
        "final_round": args.final_round,
        "is_3d": is_3d,
        "num_train": len(train_names),
        "num_full_slices": len(full_names),
        "num_full_cases": len(full_cases),
        "num_box_slices": len(box_names),
        "counts": counts,
        "target_dir": str(target_dir),
        "boxonly_reference": str(boxonly_dir),
        "errors": errors,
        "error_count_total": total_error_count,
        "errors_truncated": total_error_count > len(errors),
        "box_union_audit": "pseudo_student/tri_train_boxonly != 0 in student space",
        "passed": total_error_count == 0,
    }
    output = args.fold_root / "meta" / f"method_validation_{args.method}.json"
    atomic_save_json(report, output)
    print(json.dumps(report, indent=2))
    if total_error_count:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
