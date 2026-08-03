#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build TG3K Hard20-FullOnly using the exact Idea1 Full20 selection."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

import sys

IDEA_CODE_ROOT = Path(__file__).resolve().parents[1]
if str(IDEA_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(IDEA_CODE_ROOT))

from idea1_common import (
    atomic_save_json,
    full_and_box_from_selection,
    infer_is_3d,
    load_manifest,
    load_npy_2d,
    load_selection,
    load_split_meta,
    manifest_index,
    resolve_path,
    train_items,
)

METHOD = "hard20_fullonly"


def values(path: Path) -> set[int]:
    return {int(x) for x in np.unique(load_npy_2d(path)).tolist()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold_root", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    fold = args.fold_root.resolve()
    manifest = load_manifest(fold)
    if infer_is_3d("tg3k", load_split_meta(fold)):
        raise RuntimeError("TG3K unexpectedly detected as 3D")
    full, box, _ = full_and_box_from_selection(
        load_selection(args.selection), manifest, is_3d=False
    )
    train = train_items(manifest)
    index = manifest_index(manifest)
    if len(full) != 20 or len(full) + len(box) != len(train):
        raise RuntimeError(
            f"Invalid partition: full={len(full)} box={len(box)} train={len(train)}"
        )

    report: dict[str, Any] = {
        "dataset": "tg3k",
        "method": METHOD,
        "selection": str(args.selection.resolve()),
        "contract": "same Full20 GT; remaining Box labels from frozen tri_train_boxonly; no teacher fine-tuning",
        "num_train": len(train),
        "num_full": len(full),
        "num_box": len(box),
        "spaces": {},
    }

    for space, gt_key in (("pseudo_student", "student_gt"), ("pseudo_teacher", "teacher_gt")):
        src_root = fold / space / "tri_train_boxonly"
        dst_root = fold / space / f"tri_train_{METHOD}"
        if dst_root.exists():
            if not args.overwrite:
                raise FileExistsError(dst_root)
            shutil.rmtree(dst_root)
        dst_root.mkdir(parents=True)

        box_ignore = 0
        for name in sorted(box):
            src = src_root / name
            shutil.copy2(src, dst_root / name)
            box_ignore += int(255 in values(src))

        full_ignore = 0
        for name in sorted(full):
            src = resolve_path(fold, index[name], gt_key)
            full_ignore += int(255 in values(src))
            shutil.copy2(src, dst_root / name)

        if full_ignore:
            raise RuntimeError(f"Full GT contains ignore label in {space}")
        if len(list(dst_root.glob("*.npy"))) != len(train):
            raise RuntimeError(f"Output count mismatch in {space}")

        report["spaces"][space] = {
            "box_source": str(src_root),
            "target": str(dst_root),
            "box_with_255": box_ignore,
            "full_with_255": full_ignore,
        }

    out = fold / "meta" / f"hybrid_supervision_{METHOD}.json"
    atomic_save_json(report, out)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
