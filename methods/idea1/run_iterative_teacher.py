#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

from idea1_common import (
    METHOD_DEFAULT,
    RoundPaths,
    atomic_save_json,
    infer_is_3d,
    load_json,
    load_split_meta,
)


def run(cmd: list[str], cwd: Path) -> None:
    print(
        "\n[RUN]",
        " ".join(cmd),
        flush=True,
    )
    subprocess.run(
        cmd,
        cwd=cwd,
        check=True,
    )


def validate_v2_selection(path: Path) -> dict:
    obj = load_json(path)

    if (
        obj.get("selection_version")
        != "active_learning_v2"
    ):
        raise RuntimeError(
            "Existing selection does not belong to "
            "Active Learning V2 and must not be reused:\n"
            f"{path}\n"
            f"selection_version="
            f"{obj.get('selection_version')!r}"
        )

    return obj


def validate_v2_metrics(
    path: Path,
    is_3d: bool,
) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    header = (
        path.open(
            "r",
            encoding="utf-8",
        )
        .readline()
        .strip()
        .split(",")
    )

    required = (
        "case_min_slice_class_iou"
        if is_3d
        else "image_min_iou"
    )

    if required not in header:
        raise RuntimeError(
            "Existing diagnosis metrics belong to an "
            "older protocol and must not be reused:\n"
            f"{path}\n"
            f"missing required V2 field: {required}"
        )


def validate_v2_selection_result(
    path: Path,
) -> dict:
    obj = load_json(path)

    if (
        obj.get("selection_version")
        != "active_learning_v2"
    ):
        raise RuntimeError(
            "Existing selection_result is not "
            "Active Learning V2:\n"
            f"{path}"
        )

    status = obj.get(
        "decision_status"
    )

    if status not in {
        "CONTINUE",
        "CONVERGED",
        "BUDGET_EXHAUSTED",
    }:
        raise RuntimeError(
            "Invalid/missing Active Learning V2 "
            f"decision_status={status!r} in {path}"
        )

    return obj


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Active Learning V2 MedSAM teacher iteration: "
            "random Round0 followed by minimum-IoU hard-sample "
            "selection until convergence or the cumulative "
            "5% Full-annotation budget is exhausted."
        )
    )

    parser.add_argument(
        "--repo_root",
        type=Path,
        required=True,
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
        "--fold",
        default="fold_0",
    )

    parser.add_argument(
        "--method",
        default=METHOD_DEFAULT,
    )

    parser.add_argument(
        "--base_checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--steps_2d",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--steps_3d",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )

    # Kept for backward CLI compatibility.
    #
    # Under V2 this is NOT an experimental stopping rule.
    # The automatically derived annotation-budget horizon
    # always takes precedence if it is larger.
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=0,
        help=(
            "Optional safety horizon. Active Learning V2 "
            "automatically enlarges this value when needed "
            "to cover the complete annotation budget."
        ),
    )

    parser.add_argument(
        "--start_round",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--amp_init_scale",
        type=float,
        default=1024.0,
    )

    parser.add_argument(
        "--disable_amp",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = parser.parse_args()

    script_root = (
        Path(__file__)
        .resolve()
        .parent
    )

    python = sys.executable

    is_3d = infer_is_3d(
        args.dataset,
        load_split_meta(
            args.fold_root
        ),
    )

    steps = (
        args.steps_3d
        if is_3d
        else args.steps_2d
    )

    # ========================================================
    # Load the single source of truth for AL V2.
    # ========================================================

    budget_path = (
        args.fold_root
        / "meta"
        / f"annotation_budget_{args.method}.json"
    )

    if not budget_path.exists():
        raise FileNotFoundError(
            "Active Learning V2 annotation budget "
            f"is missing: {budget_path}"
        )

    budget = load_json(
        budget_path
    )

    if (
        budget.get("budget_version")
        != "active_learning_v2"
    ):
        raise RuntimeError(
            "Teacher iteration requires "
            "budget_version='active_learning_v2', "
            f"got {budget.get('budget_version')!r}"
        )

    if bool(
        budget.get("is_3d")
    ) != is_3d:
        raise RuntimeError(
            "Dataset dimensionality disagrees "
            "with annotation budget."
        )

    threshold = float(
        budget.get(
            "stop_iou_threshold",
            0.5,
        )
    )

    if abs(
        threshold - 0.5
    ) > 1e-12:
        raise RuntimeError(
            "V2 contract fixes the convergence "
            f"IoU threshold at 0.5, got {threshold}"
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

    # ========================================================
    # Automatic safety horizon.
    #
    # This is NOT a stopping criterion.
    #
    # In the worst case:
    # Round0 consumes one quota, and each following round
    # consumes another quota until max_full is reached.
    #
    # At or before this horizon Stage 06 must therefore return
    # CONVERGED or BUDGET_EXHAUSTED.
    # ========================================================

    requested_round_horizon = max(
        0,
        int(args.max_rounds),
    )

    if is_3d:
        # Preserve the committed 3D controller behavior.
        protocol_round_horizon = max(
            1,
            int(math.ceil(
                max_full
                / per_round_quota
            )),
        )

        effective_max_rounds = max(
            protocol_round_horizon,
            requested_round_horizon,
        )

    else:
        # 2D scientific protocol:
        # no more than five acquisition/training rounds.
        protocol_round_horizon = int(
            budget.get(
                "max_acquisition_rounds",
                5,
            )
        )

        if protocol_round_horizon <= 0:
            raise RuntimeError(
                "Invalid 2D max_acquisition_rounds="
                f"{protocol_round_horizon}"
            )

        # A CLI safety horizon must never enlarge the
        # scientific 2D five-round protocol.
        effective_max_rounds = (
            protocol_round_horizon
        )

    if (
        args.start_round < 0
        or args.start_round >= effective_max_rounds
    ):
        raise ValueError(
            "--start_round must satisfy "
            "0 <= start_round < effective_max_rounds; "
            f"start_round={args.start_round}, "
            f"effective_max_rounds="
            f"{effective_max_rounds}"
        )

    print(
        json.dumps(
            {
                "active_learning_protocol": (
                    "active_learning_v2"
                ),
                "dataset": args.dataset,
                "is_3d": is_3d,
                "iou_threshold": threshold,
                "per_round_quota": (
                    per_round_quota
                ),
                "max_full": max_full,
                "protocol_round_horizon": (
                    protocol_round_horizon
                ),
                "requested_max_rounds": (
                    requested_round_horizon
                ),
                "effective_max_rounds": (
                    effective_max_rounds
                ),
                "three_d_budget_formula": (
                    budget.get(
                        "three_d_formula"
                    )
                ),
            },
            indent=2,
        ),
        flush=True,
    )

    # ========================================================
    # Round0
    # ========================================================

    round0_selection = (
        RoundPaths(
            args.fold_root,
            args.method,
            0,
        ).selection_dir
        / "selection.json"
    )

    if not round0_selection.exists():
        cmd = [
            python,
            str(
                script_root
                / "02_select_round0_random.py"
            ),
            "--fold_root",
            str(args.fold_root),
            "--dataset",
            args.dataset,
            "--fold",
            args.fold,
            "--method",
            args.method,
            "--seed",
            str(args.seed),
        ]

        if args.overwrite:
            cmd.append(
                "--overwrite"
            )

        run(
            cmd,
            args.repo_root,
        )

    # Never silently reuse a V1 Round0 selection.
    validate_v2_selection(
        round0_selection
    )

    # ========================================================
    # Resume checkpoint
    # ========================================================

    if args.start_round == 0:
        checkpoint = (
            args.base_checkpoint
        )

    else:
        checkpoint = (
            RoundPaths(
                args.fold_root,
                args.method,
                args.start_round - 1,
            ).teacher_dir
            / "medsam_ft.pth"
        )

        if not checkpoint.exists():
            raise FileNotFoundError(
                "Previous-round checkpoint "
                f"required for resume: {checkpoint}"
            )

    final_round = None
    final_decision_status = None
    final_selection_result = None

    # ========================================================
    # Active-learning loop
    # ========================================================

    for round_index in range(
        args.start_round,
        effective_max_rounds,
    ):
        paths = RoundPaths(
            args.fold_root,
            args.method,
            round_index,
        )

        selection_path = (
            paths.selection_dir
            / "selection.json"
        )

        if not selection_path.exists():
            raise RuntimeError(
                "Selection missing for round "
                f"{round_index}: "
                f"{paths.selection_dir}"
            )

        validate_v2_selection(
            selection_path
        )

        # ----------------------------------------------------
        # Stage 03: Full pairs
        # ----------------------------------------------------

        pairs_path = (
            paths.pair_dir
            / "pairs.json"
        )

        if (
            pairs_path.exists()
            and not args.overwrite
        ):
            print(
                "[REUSE] finetune pairs: "
                f"{pairs_path}",
                flush=True,
            )

        else:
            cmd = [
                python,
                str(
                    script_root
                    / "03_build_full_finetune_pairs.py"
                ),
                "--fold_root",
                str(args.fold_root),
                "--dataset",
                args.dataset,
                "--round",
                str(round_index),
                "--method",
                args.method,
            ]

            if args.overwrite:
                cmd.append(
                    "--overwrite"
                )

            run(
                cmd,
                args.repo_root,
            )

        # ----------------------------------------------------
        # Stage 04: MedSAM mask-decoder FT
        # ----------------------------------------------------

        teacher_ckpt = (
            paths.teacher_dir
            / "medsam_ft.pth"
        )

        if (
            teacher_ckpt.exists()
            and not args.overwrite
        ):
            print(
                "[REUSE] teacher checkpoint: "
                f"{teacher_ckpt}",
                flush=True,
            )

        else:
            cmd = [
                python,
                str(
                    script_root
                    / "04_finetune_medsam_mask_decoder.py"
                ),

                "--repo_root",
                str(args.repo_root),

                "--fold_root",
                str(args.fold_root),

                "--dataset",
                args.dataset,

                "--round",
                str(round_index),

                "--method",
                args.method,

                "--checkpoint",
                str(checkpoint),

                "--device",
                args.device,

                "--seed",
                str(args.seed),

                "--max_steps",
                str(steps),

                "--batch_size",
                str(args.batch_size),

                "--lr",
                "1e-5",

                "--weight_decay",
                "0.01",

                "--dice_weight",
                "1.0",

                "--bce_weight",
                "1.0",

                "--max_grad_norm",
                "1.0",

                "--amp_init_scale",
                str(args.amp_init_scale),
            ]

            if not args.disable_amp:
                cmd.append(
                    "--amp"
                )

            if args.overwrite:
                cmd.append(
                    "--overwrite"
                )

            run(
                cmd,
                args.repo_root,
            )

        checkpoint = teacher_ckpt

        # ----------------------------------------------------
        # Stage 05:
        # score every remaining Box target.
        #
        # V2 uses minimum IoU, not macro IoU.
        # ----------------------------------------------------

        metrics_path = (
            paths.diagnosis_dir
            / (
                "per_case_metrics.csv"
                if is_3d
                else "per_image_metrics.csv"
            )
        )

        if (
            metrics_path.exists()
            and not args.overwrite
        ):
            validate_v2_metrics(
                metrics_path,
                is_3d,
            )

            print(
                "[REUSE] V2 diagnosis metrics: "
                f"{metrics_path}",
                flush=True,
            )

        else:
            cmd = [
                python,
                str(
                    script_root
                    / "05_score_remaining_box_pool.py"
                ),

                "--repo_root",
                str(args.repo_root),

                "--baseline_generator",
                str(
                    script_root.parent
                    / "generate_pseudo_labels.py"
                ),

                "--fold_root",
                str(args.fold_root),

                "--dataset",
                args.dataset,

                "--round",
                str(round_index),

                "--method",
                args.method,

                "--checkpoint",
                str(checkpoint),

                "--device",
                args.device,

                "--threshold",
                "0.5",

                "--hard_threshold",
                "0.5",

                "--save_probability",
                "hard",
            ]

            run(
                cmd,
                args.repo_root,
            )

            validate_v2_metrics(
                metrics_path,
                is_3d,
            )

        # ----------------------------------------------------
        # Stage 06:
        # CONVERGED / BUDGET_EXHAUSTED / CONTINUE.
        #
        # No add_2d/max_full_2d/add_3d parameters remain.
        # Stage 06 reads annotation_budget_*.json directly.
        # ----------------------------------------------------

        selection_result_path = (
            paths.diagnosis_dir
            / "selection_result.json"
        )

        if (
            selection_result_path.exists()
            and not args.overwrite
        ):
            selection_result = (
                validate_v2_selection_result(
                    selection_result_path
                )
            )

            print(
                "[REUSE] V2 selection result: "
                f"{selection_result_path}",
                flush=True,
            )

        else:
            cmd = [
                python,
                str(
                    script_root
                    / "06_select_next_hard_samples.py"
                ),

                "--fold_root",
                str(args.fold_root),

                "--dataset",
                args.dataset,

                "--round",
                str(round_index),

                "--method",
                args.method,
            ]

            if args.overwrite:
                cmd.append(
                    "--overwrite"
                )

            run(
                cmd,
                args.repo_root,
            )

            selection_result = (
                validate_v2_selection_result(
                    selection_result_path
                )
            )

        decision_status = (
            selection_result[
                "decision_status"
            ]
        )

        print(
            "[ACTIVE-LEARNING] "
            f"round={round_index} "
            f"status={decision_status} "
            f"min_remaining_iou="
            f"{selection_result.get('minimum_remaining_iou')} "
            f"hard_candidates="
            f"{selection_result.get('num_hard_candidates')} "
            f"selected_new="
            f"{len(selection_result.get('selected_new_ids', []))}",
            flush=True,
        )

        if decision_status in {
            "CONVERGED",
            "BUDGET_EXHAUSTED",
        }:
            final_round = (
                round_index
            )

            final_decision_status = (
                decision_status
            )

            final_selection_result = (
                selection_result
            )

            break

        if (
            decision_status
            != "CONTINUE"
        ):
            raise RuntimeError(
                "Unexpected Active Learning V2 "
                f"decision: {decision_status}"
            )

    # ========================================================
    # A protocol-complete run must stop through Stage 06.
    # ========================================================

    if final_round is None:
        raise RuntimeError(
            "Active Learning V2 reached its derived "
            "annotation-budget round horizon without "
            "CONVERGED or BUDGET_EXHAUSTED. "
            "This indicates an internal protocol/state "
            "inconsistency rather than a normal stop. "
            f"effective_max_rounds={effective_max_rounds}"
        )

    final = {
        "protocol": (
            "active_learning_v2"
        ),

        "dataset": (
            args.dataset
        ),

        "method": (
            args.method
        ),

        "is_3d": (
            is_3d
        ),

        "iou_threshold": (
            threshold
        ),

        "final_round": (
            final_round
        ),

        "final_decision_status": (
            final_decision_status
        ),

        "converged": (
            final_decision_status
            == "CONVERGED"
        ),

        "budget_exhausted": (
            final_decision_status
            == "BUDGET_EXHAUSTED"
        ),

        "minimum_remaining_iou": (
            final_selection_result.get(
                "minimum_remaining_iou"
            )
        ),

        "num_hard_candidates_remaining": (
            final_selection_result.get(
                "num_hard_candidates"
            )
        ),

        "per_round_quota": (
            per_round_quota
        ),

        "budget_max_full": (
            max_full
        ),

        "protocol_round_horizon": (
            protocol_round_horizon
        ),

        "effective_max_rounds": (
            effective_max_rounds
        ),

        "annotation_budget": str(
            budget_path
        ),

        "final_checkpoint": str(
            RoundPaths(
                args.fold_root,
                args.method,
                final_round,
            ).teacher_dir
            / "medsam_ft.pth"
        ),

        "stop_result": str(
            RoundPaths(
                args.fold_root,
                args.method,
                final_round,
            ).diagnosis_dir
            / "selection_result.json"
        ),
    }

    atomic_save_json(
        final,
        (
            args.fold_root
            / "meta"
            / (
                "teacher_iteration_final_"
                f"{args.method}.json"
            )
        ),
    )

    print(
        json.dumps(
            final,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
