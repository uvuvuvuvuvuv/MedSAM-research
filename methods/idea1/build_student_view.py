#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import METHOD_DEFAULT, atomic_save_json, copy_or_symlink


def build_view(fold_root: Path, view_fold: Path, pseudo_name: str, overwrite: bool) -> None:
    view_fold.mkdir(parents=True, exist_ok=True)
    for name in ("native_npy", "teacher_npy", "student_npy", "prompts", "meta"):
        copy_or_symlink(fold_root / name, view_fold / name, "symlink", overwrite=overwrite)
    pseudo_root = view_fold / "pseudo_student"
    pseudo_root.mkdir(parents=True, exist_ok=True)
    copy_or_symlink(
        fold_root / "pseudo_student" / pseudo_name,
        pseudo_root / "tri_train",
        "symlink",
        overwrite=overwrite,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Expose method-specific labels through the unchanged baseline student fold interface.")
    parser.add_argument("--fold_root", type=Path, required=True)
    parser.add_argument("--view_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--build_boxonly", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    method_view = args.view_root / args.method / args.dataset / args.fold
    build_view(args.fold_root, method_view, f"tri_train_{args.method}", args.overwrite)
    outputs = {"method_view": str(method_view)}
    if args.build_boxonly:
        box_view = args.view_root / "boxonly" / args.dataset / args.fold
        build_view(args.fold_root, box_view, "tri_train_boxonly", args.overwrite)
        outputs["boxonly_view"] = str(box_view)
    atomic_save_json(outputs, args.fold_root / "meta" / f"student_views_{args.method}.json")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
