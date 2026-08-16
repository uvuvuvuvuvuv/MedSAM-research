#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

from idea1_common import METHOD_DEFAULT, RoundPaths, atomic_save_json, current_timestamp, load_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the final immutable Idea1 method record.")
    parser.add_argument("--fold_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--final_round", type=int, required=True)
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--student_view", type=Path, required=True)
    args = parser.parse_args()

    paths = RoundPaths(args.fold_root, args.method, args.final_round)
    checkpoint = paths.teacher_dir / "medsam_ft.pth"
    validation_path = args.fold_root / "meta" / f"method_validation_{args.method}.json"
    validation = load_json(validation_path)
    if not validation.get("passed", False):
        raise RuntimeError(f"Method validation did not pass: {validation_path}")
    record = {
        "method": args.method,
        "parent_method": "frozen_baseline_v1",
        "dataset": args.dataset,
        "final_round": args.final_round,
        "created_at": current_timestamp(),
        "teacher_checkpoint": str(checkpoint),
        "teacher_checkpoint_sha256": sha256_file(checkpoint),
        "student_supervision": str(args.fold_root / "pseudo_student" / f"tri_train_{args.method}"),
        "teacher_supervision": str(args.fold_root / "pseudo_teacher" / f"tri_train_{args.method}"),
        "boxonly_reference": str(args.fold_root / "pseudo_student" / "tri_train_boxonly"),
        "student_view": str(args.student_view),
        "validation": str(validation_path),
        "comparison_groups": ["frozen_baseline_v1", args.method, "upper"],
    }
    output = args.fold_root / "meta" / f"final_method_{args.method}.json"
    atomic_save_json(record, output)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
