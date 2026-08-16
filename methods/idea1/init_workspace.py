#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from common import (
    METHOD_DEFAULT,
    assert_output_isolated,
    atomic_save_json,
    compute_3d_budget,
    copy_or_symlink,
    current_timestamp,
    ensure_method_not_baseline,
    get_case_id,
    infer_is_3d,
    load_manifest,
    load_split_meta,
    normalize_workspace_spacing_metadata,
    sha256_tree,
    train_items,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an isolated Idea1 workspace from frozen baseline inputs.")
    parser.add_argument("--frozen_processed_root", type=Path, required=True)
    parser.add_argument("--idea_processed_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--shared_mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument("--boxonly_mode", choices=["copy", "symlink"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--baseline_audit", type=Path)
    args = parser.parse_args()

    ensure_method_not_baseline(args.method)
    frozen_fold = args.frozen_processed_root / args.dataset / args.fold
    idea_fold = args.idea_processed_root / args.dataset / args.fold
    assert_output_isolated(idea_fold, [args.frozen_processed_root])
    idea_fold.mkdir(parents=True, exist_ok=True)

    if args.baseline_audit:
        audit = json.loads(args.baseline_audit.read_text(encoding="utf-8"))
        if not audit.get("passed", False):
            raise RuntimeError(f"Baseline audit did not pass: {args.baseline_audit}")

    # Large immutable arrays can be symlinked. Small contracts are copied so
    # the method workspace remains self-describing and cannot mutate baseline metadata.
    for name in ("native_npy", "teacher_npy", "student_npy"):
        copy_or_symlink(frozen_fold / name, idea_fold / name, args.shared_mode, args.overwrite)
    for name in ("prompts", "meta"):
        copy_or_symlink(frozen_fold / name, idea_fold / name, "copy", args.overwrite)

    for space in ("pseudo_student", "pseudo_teacher"):
        src = frozen_fold / space / "tri_train"
        dst = idea_fold / space / "tri_train_boxonly"
        copy_or_symlink(src, dst, args.boxonly_mode, args.overwrite)
        (idea_fold / space).mkdir(parents=True, exist_ok=True)

    (idea_fold / "rounds" / args.method).mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(idea_fold)
    split_meta = load_split_meta(idea_fold)
    train = train_items(manifest)
    is_3d = infer_is_3d(args.dataset, split_meta)
    spacing_normalization = None
    if is_3d:
        spacing_normalization = normalize_workspace_spacing_metadata(idea_fold)
        manifest = load_manifest(idea_fold)
        train = train_items(manifest)
    train_cases = sorted(
        {get_case_id(x, required=True) for x in train}
    ) if is_3d else []

    # ======================================================
    # Active-learning V2 annotation budget
    #
    # Per-round annotation unit:
    #   max(ceil(1% * N), 5)
    #
    # 2D unit = image
    # 3D unit = case
    #
    # For 3D, the cumulative Full-case annotation budget
    # is 5% of all train cases, rounded UP to the nearest
    # whole case because case-level annotation is indivisible.
    # ======================================================

    annotation_unit_count = (
        len(train_cases)
        if is_3d
        else len(train)
    )

    if annotation_unit_count <= 0:
        raise RuntimeError(
            "No train annotation units were found."
        )

    # ======================================================
    # Annotation budget
    #
    # 3D:
    #   Keep the committed Active Learning V2 behavior.
    #
    # 2D:
    #   q = min(max(1, ceil(0.01 * N)), 5)
    #
    #   cumulative Full cap =
    #       min(
    #           ceil(0.05 * N),
    #           5 * q,
    #       )
    #
    #   acquisition rounds <= 5.
    # ======================================================

    if is_3d:
        # --------------------------------------------------
        # Preserve committed 3D behavior unchanged.
        # --------------------------------------------------
        requested_round_quota = max(
            int(math.ceil(
                0.01 * annotation_unit_count
            )),
            5,
        )

        requested_round_quota = min(
            requested_round_quota,
            annotation_unit_count,
        )

        max_full = max(
            1,
            int(math.ceil(
                0.05 * annotation_unit_count
            )),
        )

    else:
        # --------------------------------------------------
        # New 2D protocol.
        # --------------------------------------------------
        requested_round_quota = min(
            max(
                1,
                int(math.ceil(
                    0.01 * annotation_unit_count
                )),
            ),
            5,
            annotation_unit_count,
        )

        five_percent_cap = max(
            1,
            int(math.ceil(
                0.05 * annotation_unit_count
            )),
        )

        max_full = min(
            five_percent_cap,
            5 * requested_round_quota,
        )

    effective_round_quota = min(
        requested_round_quota,
        max_full,
    )

    budget = {
        "budget_version": "active_learning_v2",
        "dataset": args.dataset,
        "fold": args.fold,
        "method": args.method,
        "is_3d": is_3d,

        "annotation_unit": (
            "case" if is_3d else "image"
        ),

        "num_train_slices_or_images": len(train),
        "num_train_cases": len(train_cases),
        "num_annotation_units": annotation_unit_count,

        "per_round_ratio": 0.01,
        "minimum_per_round": (5 if is_3d else 1),
        "rounding_per_round": "ceil",
        "requested_round_quota": requested_round_quota,

        "round0": effective_round_quota,
        "add_per_round": effective_round_quota,

        "max_full": max_full,

        "max_acquisition_rounds": (
            None if is_3d else 5
        ),

        "two_d_round_quota_formula": (
            "min(max(1,ceil(0.01*N)),5)"
            if not is_3d
            else None
        ),

        "two_d_max_full_formula": (
            "min(max(1,ceil(0.05*N)),"
            "5*round_quota)"
            if not is_3d
            else None
        ),

        # Unified 2D/3D cumulative annotation budget.
        "cumulative_full_ratio": 0.05,
        "cumulative_cap_rounding": "ceil",
        "cumulative_formula": (
            "max(1,ceil(0.05*N_annotation_units))"
            if is_3d
            else
            "min(max(1,ceil(0.05*N_images)),"
            "5*round_quota)"
        ),

        # Retained for backward-compatible metadata.
        "three_d_max_ratio": (
            0.05 if is_3d else None
        ),
        "three_d_cap_rounding": (
            "ceil" if is_3d else None
        ),
        "three_d_formula": (
            "max(1,ceil(0.05*N_train_cases))"
            if is_3d
            else None
        ),

        "stop_iou_threshold": 0.5,
        "convergence_rule": (
            "all_remaining_gt_targets_iou_ge_0.5"
        ),
    }
    atomic_save_json(budget, idea_fold / "meta" / f"annotation_budget_{args.method}.json")

    lineage = {
        "method": args.method,
        "parent_method": "frozen_baseline_v1",
        "created_at": current_timestamp(),
        "frozen_fold_root": str(frozen_fold.resolve()),
        "idea_fold_root": str(idea_fold.resolve()),
        "shared_mode": args.shared_mode,
        "boxonly_mode": args.boxonly_mode,
        "baseline_audit": str(args.baseline_audit.resolve()) if args.baseline_audit else None,
        "spacing_normalization": spacing_normalization,
        "prompt_hashes": sha256_tree(idea_fold / "prompts", {".json", ".csv"}),
        "meta_contract_hashes": sha256_tree(idea_fold / "meta", {".json", ".csv"}),
    }
    atomic_save_json(lineage, idea_fold / "meta" / f"method_lineage_{args.method}.json")
    print(json.dumps({"idea_fold": str(idea_fold), "budget": budget}, indent=2))


if __name__ == "__main__":
    main()
