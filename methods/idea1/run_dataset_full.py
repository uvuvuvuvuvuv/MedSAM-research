#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Resume-safe MedSAM/label stages for one Idea1 dataset.

This wrapper does not alter the frozen baseline. It resumes round-by-round and
rebuilds only incomplete method outputs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from idea1_common import METHOD_DEFAULT, atomic_save_json, load_json


SUPPORTED = {
    "btcv", "synapse", "acdc", "prostate158",
    "kvasirseg", "cvc_clinicdb", "tn3k", "tg3k",
    "ddti", "otu_2d", "ph2",
}


def run(cmd: list[str], cwd: Path) -> None:
    print("\n[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def round_root(fold_root: Path, method: str, round_index: int) -> Path:
    """Return the on-disk round directory used by the existing Idea1 pipeline."""
    return fold_root / "rounds" / method / f"round_{round_index:02d}"


def find_pairs_marker(root: Path) -> Path | None:
    """Locate the pair manifest without depending on a RoundPaths property name."""
    preferred = (
        root / "finetune_pairs" / "pairs.json",
        root / "finetune" / "pairs.json",
        root / "pairs" / "pairs.json",
    )
    for path in preferred:
        if path.is_file():
            return path

    found = sorted(root.rglob("pairs.json")) if root.is_dir() else []
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise RuntimeError(
            "Ambiguous pairs.json markers under "
            f"{root}: {[str(path) for path in found]}"
        )
    return None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", type=Path, required=True)
    p.add_argument("--frozen_processed_root", type=Path, required=True)
    p.add_argument("--idea_processed_root", type=Path, required=True)
    p.add_argument("--view_root", type=Path, required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--fold", default="fold_0")
    p.add_argument("--method", default=METHOD_DEFAULT)
    p.add_argument("--base_checkpoint", type=Path, required=True)
    p.add_argument("--baseline_audit", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--steps_2d", type=int, default=300)
    p.add_argument("--steps_3d", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument(
        "--pseudo_protocol",
        default="student_v2_probability_arbitration",
        choices=["student_v2_probability_arbitration"],
    )
    args = p.parse_args()

    if args.dataset not in SUPPORTED:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(args.base_checkpoint)
    audit = load_json(args.baseline_audit)
    if not audit.get("passed", False):
        raise RuntimeError(f"Baseline audit did not pass: {args.baseline_audit}")

    repo = args.repo_root.resolve()
    code = Path(__file__).resolve().parent
    py = sys.executable
    fold_root = args.idea_processed_root / args.dataset / args.fold
    lineage = fold_root / "meta" / f"method_lineage_{args.method}.json"

    # Stage 1: isolated workspace.
    if not lineage.is_file():
        run([
            py, str(code / "01_init_idea1_workspace.py"),
            "--frozen_processed_root", str(args.frozen_processed_root),
            "--idea_processed_root", str(args.idea_processed_root),
            "--dataset", args.dataset,
            "--fold", args.fold,
            "--method", args.method,
            "--shared_mode", "symlink",
            "--boxonly_mode", "symlink",
            "--baseline_audit", str(args.baseline_audit),
        ], repo)
    else:
        print(f"[SKIP] workspace exists: {fold_root}")

    # ========================================================
    # Active Learning V2 contract audit
    # ========================================================

    budget_path = (
        fold_root
        / "meta"
        / f"annotation_budget_{args.method}.json"
    )

    if not budget_path.is_file():
        raise FileNotFoundError(
            "Active Learning V2 budget is missing: "
            f"{budget_path}"
        )

    budget = load_json(
        budget_path
    )

    if (
        budget.get("budget_version")
        != "active_learning_v2"
    ):
        raise RuntimeError(
            "This run_dataset_full.py is the V2 pipeline, "
            "but the workspace contains a non-V2 annotation "
            "budget. Do not reuse the frozen V1 run tree. "
            "Use a fresh V2 idea_processed_root.\n"
            f"budget={budget_path}\n"
            f"budget_version="
            f"{budget.get('budget_version')!r}"
        )

    # ========================================================
    # Stage 2:
    # delegate ALL teacher active-learning logic to the
    # single V2 controller.
    #
    # run_dataset_full.py no longer contains:
    #   - fixed add_2d / add_3d values
    #   - max_full_2d=20
    #   - fixed max_rounds=8
    #   - macro-IoU stopping
    #
    # The authoritative controller is:
    #   run_iterative_teacher.py
    # ========================================================

    final_teacher_meta = (
        fold_root
        / "meta"
        / f"teacher_iteration_final_{args.method}.json"
    )

    if not final_teacher_meta.is_file():
        run(
            [
                py,
                str(
                    code
                    / "run_iterative_teacher.py"
                ),

                "--repo_root",
                str(repo),

                "--fold_root",
                str(fold_root),

                "--dataset",
                args.dataset,

                "--fold",
                args.fold,

                "--method",
                args.method,

                "--base_checkpoint",
                str(args.base_checkpoint),

                "--device",
                args.device,

                "--seed",
                str(args.seed),

                "--steps_2d",
                str(args.steps_2d),

                "--steps_3d",
                str(args.steps_3d),

                "--batch_size",
                str(args.batch_size),
            ],
            repo,
        )

    else:
        print(
            "[SKIP] Active Learning V2 teacher "
            f"iteration marker exists: {final_teacher_meta}"
        )

    if not final_teacher_meta.is_file():
        raise RuntimeError(
            "run_iterative_teacher.py completed without "
            "creating the final teacher metadata: "
            f"{final_teacher_meta}"
        )

    teacher_meta = load_json(
        final_teacher_meta
    )

    if (
        teacher_meta.get("protocol")
        != "active_learning_v2"
    ):
        raise RuntimeError(
            "Final teacher metadata does not belong to "
            "Active Learning V2. Refusing to mix V1/V2 runs. "
            f"protocol={teacher_meta.get('protocol')!r}"
        )

    final_decision_status = (
        teacher_meta.get(
            "final_decision_status"
        )
    )

    if final_decision_status not in {
        "CONVERGED",
        "BUDGET_EXHAUSTED",
        "MAX_ROUNDS_REACHED",
    }:
        raise RuntimeError(
            "Invalid Active Learning V2 terminal state: "
            f"{final_decision_status!r}"
        )

    final_round = int(
        teacher_meta["final_round"]
    )

    print(
        "[ACTIVE-LEARNING V2 COMPLETE] "
        f"dataset={args.dataset} "
        f"round={final_round} "
        f"status={final_decision_status} "
        f"min_remaining_iou="
        f"{teacher_meta.get('minimum_remaining_iou')} "
        f"hard_remaining="
        f"{teacher_meta.get('num_hard_candidates_remaining')}"
    )

    # Stage 3: final pseudo/hybrid/student view. Rebuild only if final marker is absent.
    final_method = fold_root / "meta" / f"final_method_{args.method}.json"
    if not final_method.is_file():
        run([
            py, str(code / "run_final_pipeline.py"),
            "--repo_root", str(repo),
            "--fold_root", str(fold_root),
            "--view_root", str(args.view_root),
            "--dataset", args.dataset,
            "--fold", args.fold,
            "--final_round", str(final_round),
            "--method", args.method,
            "--pseudo_protocol", args.pseudo_protocol,
            "--overwrite",
        ], repo)
    else:
        print(f"[SKIP] final method marker exists: {final_method}")

    print(json.dumps({
        "dataset": args.dataset,
        "fold_root": str(fold_root),
        "final_round": final_round,
        "active_learning_status": final_decision_status,
        "final_method": str(final_method),
        "student_view": str(args.view_root / args.method / args.dataset / args.fold),
        "pseudo_protocol": args.pseudo_protocol,
        "status": "MEDSAM_AND_LABEL_STAGES_COMPLETE",
    }, indent=2))


if __name__ == "__main__":
    main()
