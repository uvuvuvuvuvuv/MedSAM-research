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
    parser = argparse.ArgumentParser(
        description=(
            "Select the next Active Learning V2 hard Full samples "
            "using minimum GT IoU."
        )
    )
    parser.add_argument(
        "--fold_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        required=True,
    )
    parser.add_argument(
        "--round",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--method",
        default=METHOD_DEFAULT,
    )
    parser.add_argument(
        "--selection",
        type=Path,
    )

    # Kept only as an explicit contract check.
    # The annotation budget remains the source of truth.
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    current_paths = RoundPaths(
        args.fold_root,
        args.method,
        args.round,
    )

    current_selection_path = (
        args.selection
        or (
            current_paths.selection_dir
            / "selection.json"
        )
    )

    current = load_selection(
        current_selection_path
    )

    manifest = load_manifest(
        args.fold_root
    )

    split_meta = load_split_meta(
        args.fold_root
    )

    is_3d = infer_is_3d(
        args.dataset,
        split_meta,
    )

    train = train_items(
        manifest
    )

    train_names = {
        get_slice_name(x)
        for x in train
    }

    (
        full_names,
        box_names,
        full_cases,
    ) = full_and_box_from_selection(
        current,
        manifest,
        is_3d,
    )

    # ========================================================
    # Active Learning V2 budget
    # ========================================================

    budget_path = (
        args.fold_root
        / "meta"
        / f"annotation_budget_{args.method}.json"
    )

    if not budget_path.exists():
        raise FileNotFoundError(
            f"Annotation budget missing: {budget_path}"
        )

    budget = load_json(
        budget_path
    )

    if (
        budget.get("budget_version")
        != "active_learning_v2"
    ):
        raise RuntimeError(
            "06_select_next_hard_samples.py requires "
            "budget_version='active_learning_v2', got "
            f"{budget.get('budget_version')!r}"
        )

    if bool(budget.get("is_3d")) != is_3d:
        raise RuntimeError(
            "Budget dimensionality mismatch: "
            f"is_3d={is_3d}, "
            f"budget_is_3d={budget.get('is_3d')}"
        )

    threshold = float(
        budget.get(
            "stop_iou_threshold",
            0.5,
        )
    )

    if (
        args.iou_threshold is not None
        and abs(
            float(args.iou_threshold)
            - threshold
        ) > 1e-12
    ):
        raise ValueError(
            "--iou_threshold disagrees with "
            "Active Learning V2 budget: "
            f"cli={args.iou_threshold}, "
            f"budget={threshold}"
        )

    per_round_quota = int(
        budget["add_per_round"]
    )

    max_full = int(
        budget["max_full"]
    )

    if per_round_quota <= 0:
        raise RuntimeError(
            f"Invalid add_per_round={per_round_quota}"
        )

    if max_full <= 0:
        raise RuntimeError(
            f"Invalid max_full={max_full}"
        )

    selected_new: list[str] = []

    decision_status = None
    stop_reason = None

    minimum_remaining_iou = 1.0
    num_hard_candidates = 0

    # ========================================================
    # 3D:
    # annotation unit = case
    #
    # difficulty =
    # minimum valid slice/class IoU in that case.
    # ========================================================

    if is_3d:
        metrics_path = (
            current_paths.diagnosis_dir
            / "per_case_metrics.csv"
        )

        rows = read_csv(
            metrics_path
        )

        candidates = []

        remaining_ious = []

        for row in rows:
            case_id = str(
                row["case_id"]
            )

            if case_id in full_cases:
                continue

            case_min = safe_float(
                row.get(
                    "case_min_slice_class_iou"
                ),
                1.0,
            )

            missed = int(
                float(
                    row.get(
                        "empty_prediction_classes",
                        0,
                    )
                    or 0
                )
            )

            case_macro = safe_float(
                row.get(
                    "case_macro_3d_iou"
                ),
                1.0,
            )

            remaining_ious.append(
                case_min
            )

            hard = (
                case_min < threshold
                or missed > 0
            )

            if hard:
                # Primary ordering:
                #   lower minimum IoU first.
                #
                # Secondary:
                #   more missed classes first.
                #
                # Macro IoU is diagnostic/tie-break only.
                candidates.append(
                    (
                        case_min,
                        -missed,
                        case_macro,
                        case_id,
                    )
                )

        candidates.sort()

        minimum_remaining_iou = min(
            remaining_ious,
            default=1.0,
        )

        num_hard_candidates = len(
            candidates
        )

        # ----------------------------------------------------
        # Priority 1:
        # convergence is checked BEFORE budget exhaustion.
        #
        # If all remaining pseudo targets pass, this is
        # CONVERGED even when the annotation budget happens
        # to have been fully consumed.
        # ----------------------------------------------------

        if not candidates:
            decision_status = "CONVERGED"
            stop_reason = (
                "all_remaining_targets_iou_ge_threshold"
            )

        # ----------------------------------------------------
        # Priority 2:
        # hard targets remain, but the 3D cumulative
        # case-level budget has been exhausted.
        # ----------------------------------------------------

        elif len(full_cases) >= max_full:
            decision_status = "BUDGET_EXHAUSTED"
            stop_reason = (
                "3d_full_case_budget_exhausted_"
                "with_hard_targets_remaining"
            )

        # ----------------------------------------------------
        # Otherwise select another hard-case batch.
        # ----------------------------------------------------

        else:
            capacity = (
                max_full
                - len(full_cases)
            )

            count = min(
                per_round_quota,
                capacity,
                len(candidates),
            )

            if count <= 0:
                decision_status = (
                    "BUDGET_EXHAUSTED"
                )
                stop_reason = (
                    "3d_no_remaining_annotation_capacity"
                )

            else:
                selected_new = [
                    x[3]
                    for x in candidates[:count]
                ]

                decision_status = "CONTINUE"
                stop_reason = None

    # ========================================================
    # 2D:
    # annotation unit = image
    #
    # Active Learning V2 uses the SAME cumulative 5% Full
    # annotation budget as 3D. The only difference is the
    # annotation unit:
    #
    #   2D -> image
    #   3D -> case
    # ========================================================

    else:
        metrics_path = (
            current_paths.diagnosis_dir
            / "per_image_metrics.csv"
        )

        rows = read_csv(
            metrics_path
        )

        candidates = []

        remaining_ious = []

        for row in rows:
            name = str(
                row["slice_name"]
            )

            if name in full_names:
                continue

            image_min = safe_float(
                row.get(
                    "image_min_iou"
                ),
                1.0,
            )

            empty_count = int(
                float(
                    row.get(
                        "empty_instance_count",
                        0,
                    )
                    or 0
                )
            )

            image_macro = safe_float(
                row.get(
                    "image_macro_iou"
                ),
                1.0,
            )

            remaining_ious.append(
                image_min
            )

            if image_min < threshold:
                candidates.append(
                    (
                        image_min,
                        -empty_count,
                        image_macro,
                        name,
                    )
                )

        candidates.sort()

        minimum_remaining_iou = min(
            remaining_ious,
            default=1.0,
        )

        num_hard_candidates = len(
            candidates
        )

        if not candidates:
            decision_status = "CONVERGED"
            stop_reason = (
                "all_remaining_targets_iou_ge_threshold"
            )

        elif (
            budget.get("max_acquisition_rounds")
            is not None
            and args.round >= (
                int(
                    budget[
                        "max_acquisition_rounds"
                    ]
                )
                - 1
            )
        ):
            # The current round itself has already been
            # trained and evaluated.  For a five-round 2D
            # protocol, round_04 is therefore the final
            # allowed acquisition round.
            #
            # Do NOT select samples for round_05.
            decision_status = (
                "BUDGET_EXHAUSTED"
            )

            stop_reason = (
                "2d_max_acquisition_rounds_reached_"
                "with_hard_targets_remaining"
            )

        else:
            capacity = (
                max_full
                - len(full_names)
            )

            if capacity <= 0:
                # Active Learning V2:
                # the cumulative 5% Full-image budget has
                # been consumed, but hard pseudo labels
                # still remain.
                decision_status = (
                    "BUDGET_EXHAUSTED"
                )

                stop_reason = (
                    "2d_full_image_budget_exhausted_"
                    "with_hard_targets_remaining"
                )

            else:
                count = min(
                    per_round_quota,
                    capacity,
                    len(candidates),
                )

                selected_new = [
                    x[3]
                    for x in candidates[:count]
                ]

                if not selected_new:
                    raise RuntimeError(
                        "2D hard candidates exist but "
                        "no new sample was selected."
                    )

                decision_status = "CONTINUE"
                stop_reason = None

    # ========================================================
    # Decision result
    # ========================================================

    stop = (
        decision_status
        in {
            "CONVERGED",
            "BUDGET_EXHAUSTED",
        }
    )

    current_full_units = (
        len(full_cases)
        if is_3d
        else len(full_names)
    )

    result = {
        "selection_version": (
            "active_learning_v2"
        ),
        "dataset": args.dataset,
        "fold": current.get(
            "fold",
            "fold_0",
        ),
        "method": args.method,
        "current_round": args.round,
        "is_3d": is_3d,

        "annotation_unit": (
            "case"
            if is_3d
            else "image"
        ),

        "iou_threshold": threshold,

        "difficulty_metric": (
            "case_min_slice_class_iou"
            if is_3d
            else "image_min_iou"
        ),

        "per_round_quota": (
            per_round_quota
        ),

        "budget_max_full": (
            max_full
        ),

        "current_full_units": (
            current_full_units
        ),

        "remaining_budget_capacity": max(
            0,
            max_full
            - current_full_units,
        ),

        "num_hard_candidates": (
            num_hard_candidates
        ),

        "minimum_remaining_iou": float(
            minimum_remaining_iou
        ),

        "all_remaining_targets_pass": bool(
            num_hard_candidates == 0
        ),

        "decision_status": (
            decision_status
        ),

        "selected_new_ids": (
            selected_new
        ),

        # Backward-compatible stop flag for
        # run_iterative_teacher.py.
        "stop": stop,

        "stop_reason": (
            stop_reason
        ),

        "current_full_slices": (
            len(full_names)
        ),

        "current_full_cases": (
            len(full_cases)
        ),

        "current_box_slices": (
            len(box_names)
        ),
    }

    # ========================================================
    # Build next round only for CONTINUE
    # ========================================================

    if decision_status == "CONTINUE":
        next_round = (
            args.round
            + 1
        )

        next_paths = RoundPaths(
            args.fold_root,
            args.method,
            next_round,
        )

        next_path = (
            next_paths.selection_dir
            / "selection.json"
        )

        if (
            next_path.exists()
            and not args.overwrite
        ):
            raise FileExistsError(
                next_path
            )

        if is_3d:
            case_to_slices = (
                build_case_to_slices(
                    train
                )
            )

            next_full_cases = (
                set(full_cases)
                | set(selected_new)
            )

            next_full_names = {
                name
                for cid in next_full_cases
                for name in case_to_slices[cid]
            }

        else:
            next_full_cases = set()

            next_full_names = (
                set(full_names)
                | set(selected_new)
            )

        next_box_names = (
            train_names
            - next_full_names
        )

        validate_partition(
            train_names,
            next_full_names,
            next_box_names,
        )

        payload = {
            "selection_version": (
                "active_learning_v2"
            ),

            "created_at": (
                current_timestamp()
            ),

            "dataset": args.dataset,
            "fold": current.get(
                "fold",
                "fold_0",
            ),
            "method": args.method,

            "round": next_round,

            "selection_type": (
                "hard_min_gt_iou"
            ),

            "source_round": (
                args.round
            ),

            "difficulty_iou_threshold": (
                threshold
            ),

            "difficulty_metric": (
                "case_min_slice_class_iou"
                if is_3d
                else "image_min_iou"
            ),

            "is_3d": is_3d,

            "annotation_unit": (
                "case"
                if is_3d
                else "image"
            ),

            "per_round_quota": (
                per_round_quota
            ),

            "budget_max_full": (
                max_full
            ),

            "selected_new_ids": (
                selected_new
            ),

            "cumulative_full_case_ids": sorted(
                next_full_cases
            ),

            "cumulative_full_slice_names": sorted(
                next_full_names
            ),

            "remaining_box_slice_names": sorted(
                next_box_names
            ),

            "num_full_slices": len(
                next_full_names
            ),

            "num_box_slices": len(
                next_box_names
            ),

            "num_full_cases": len(
                next_full_cases
            ),
        }

        atomic_save_json(
            payload,
            next_path,
        )

        atomic_save_json(
            selected_new,
            (
                next_paths.selection_dir
                / "selected_new_ids.json"
            ),
        )

        atomic_save_json(
            sorted(next_full_names),
            (
                next_paths.selection_dir
                / "cumulative_full_ids.json"
            ),
        )

        atomic_save_json(
            sorted(next_box_names),
            (
                next_paths.selection_dir
                / "remaining_box_ids.json"
            ),
        )

        result["next_selection"] = str(
            next_path
        )

    atomic_save_json(
        result,
        (
            current_paths.diagnosis_dir
            / "selection_result.json"
        ),
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
