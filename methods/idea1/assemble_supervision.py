#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from common import (
    METHOD_DEFAULT,
    RoundPaths,
    atomic_save_json,
    full_and_box_from_selection,
    get_slice_name,
    infer_is_3d,
    load_manifest,
    load_selection,
    load_split_meta,
    manifest_index,
    resolve_path,
    train_items,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a new Idea1 hybrid supervision directory; never modify baseline pseudo.")
    parser.add_argument("--fold_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--final_round", type=int, required=True)
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = RoundPaths(args.fold_root, args.method, args.final_round)
    selection_path = args.selection or (paths.selection_dir / "selection.json")
    selection = load_selection(selection_path)
    manifest = load_manifest(args.fold_root)
    split_meta = load_split_meta(args.fold_root)
    is_3d = infer_is_3d(args.dataset, split_meta)
    full_names, box_names, full_cases = full_and_box_from_selection(selection, manifest, is_3d)
    mindex = manifest_index(manifest)

    summary = {
        "dataset": args.dataset,
        "method": args.method,
        "final_round": args.final_round,
        "selection": str(selection_path),
        "num_full_slices": len(full_names),
        "num_full_cases": len(full_cases),
        "num_box_slices": len(box_names),
        "spaces": {},
    }

    for space, gt_key in (("pseudo_student", "student_gt"), ("pseudo_teacher", "teacher_gt")):
        box_source = args.fold_root / space / f"tri_box_final_{args.method}"
        target = args.fold_root / space / f"tri_train_{args.method}"
        if target.exists():
            if not args.overwrite:
                raise FileExistsError(target)
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)

        copied_box = 0
        copied_full = 0
        for name in sorted(box_names):
            src = box_source / name
            if not src.is_file():
                raise FileNotFoundError(src)
            shutil.copy2(src, target / name)
            copied_box += 1
        for name in sorted(full_names):
            item = mindex[name]
            src = resolve_path(args.fold_root, item, gt_key)
            shutil.copy2(src, target / name)
            copied_full += 1
        summary["spaces"][space] = {
            "target": str(target),
            "box_source": str(box_source),
            "copied_box": copied_box,
            "copied_full_gt": copied_full,
        }

    train_count = len(train_items(manifest))
    actual = len(list((args.fold_root / "pseudo_student" / f"tri_train_{args.method}").glob("*.npy")))
    if actual != train_count:
        raise RuntimeError(f"Hybrid supervision count mismatch: expected={train_count}, actual={actual}")
    summary["train_count"] = train_count
    summary["student_count"] = actual
    summary["statement"] = "This creates a new method directory; frozen baseline tri_train and tri_train_boxonly remain untouched."
    atomic_save_json(summary, args.fold_root / "meta" / f"hybrid_supervision_{args.method}.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
