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
    p.add_argument("--max_rounds", type=int, default=8)
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

    final_teacher_meta = fold_root / "meta" / f"teacher_iteration_final_{args.method}.json"
    final_round: int | None = None

    # Stage 2: resume-safe teacher iteration.
    if final_teacher_meta.is_file():
        final_round = int(load_json(final_teacher_meta)["final_round"])
        print(f"[SKIP] teacher iteration complete at round {final_round}")
    else:
        round0_root = round_root(fold_root, args.method, 0)
        round0_selection = round0_root / "selection" / "selection.json"
        if not round0_selection.is_file():
            run([
                py, str(code / "02_select_round0_random.py"),
                "--fold_root", str(fold_root),
                "--dataset", args.dataset,
                "--fold", args.fold,
                "--method", args.method,
                "--seed", str(args.seed),
            ], repo)

        for r in range(args.max_rounds):
            current_round = round_root(fold_root, args.method, r)
            selection = current_round / "selection" / "selection.json"
            if not selection.is_file():
                raise RuntimeError(f"Missing selection for round {r}: {selection}")

            pairs = find_pairs_marker(current_round)
            if pairs is None:
                run([
                    py, str(code / "03_build_full_finetune_pairs.py"),
                    "--fold_root", str(fold_root),
                    "--dataset", args.dataset,
                    "--round", str(r),
                    "--method", args.method,
                ], repo)
                pairs = find_pairs_marker(current_round)
                if pairs is None:
                    raise RuntimeError(
                        f"Round {r:02d} pair builder completed without a pairs.json marker "
                        f"under {current_round}"
                    )
            else:
                print(f"[SKIP] round {r:02d} pairs complete: {pairs}")

            input_ckpt = (
                args.base_checkpoint
                if r == 0
                else round_root(fold_root, args.method, r - 1)
                / "teacher"
                / "medsam_ft.pth"
            )
            output_ckpt = current_round / "teacher" / "medsam_ft.pth"
            if not output_ckpt.is_file():
                if not input_ckpt.is_file():
                    raise FileNotFoundError(input_ckpt)
                steps = args.steps_3d if args.dataset in {"btcv", "synapse", "acdc", "prostate158"} else args.steps_2d
                run([
                    py, str(code / "04_finetune_medsam_mask_decoder.py"),
                    "--repo_root", str(repo),
                    "--fold_root", str(fold_root),
                    "--dataset", args.dataset,
                    "--round", str(r),
                    "--method", args.method,
                    "--checkpoint", str(input_ckpt),
                    "--device", args.device,
                    "--seed", str(args.seed),
                    "--max_steps", str(steps),
                    "--batch_size", str(args.batch_size),
                    "--lr", "1e-5",
                    "--weight_decay", "0.01",
                    "--dice_weight", "1.0",
                    "--bce_weight", "1.0",
                    "--max_grad_norm", "1.0",
                    "--amp",
                ], repo)
            else:
                print(f"[SKIP] round {r:02d} teacher checkpoint complete")

            diagnosis_dir = current_round / "diagnosis"
            selection_result = diagnosis_dir / "selection_result.json"
            if not selection_result.is_file():
                run([
                    py, str(code / "05_score_remaining_box_pool.py"),
                    "--repo_root", str(repo),
                    "--baseline_generator", str(repo / "generate_pseudo_labels.py"),
                    "--fold_root", str(fold_root),
                    "--dataset", args.dataset,
                    "--round", str(r),
                    "--method", args.method,
                    "--checkpoint", str(output_ckpt),
                    "--device", args.device,
                    "--threshold", "0.5",
                    "--hard_threshold", "0.5",
                    "--save_probability", "hard",
                ], repo)

                next_selection = (
                    round_root(fold_root, args.method, r + 1)
                    / "selection"
                    / "selection.json"
                )
                cmd = [
                    py, str(code / "06_select_next_hard_samples.py"),
                    "--fold_root", str(fold_root),
                    "--dataset", args.dataset,
                    "--round", str(r),
                    "--method", args.method,
                    "--iou_threshold", "0.5",
                    "--add_2d", "5",
                    "--max_full_2d", "20",
                    "--add_3d", "1",
                ]
                if next_selection.exists():
                    cmd.append("--overwrite")
                run(cmd, repo)
                if not selection_result.is_file():
                    candidates = sorted(diagnosis_dir.rglob("selection_result.json"))
                    if len(candidates) == 1:
                        selection_result = candidates[0]
                    else:
                        raise RuntimeError(
                            f"Round {r:02d} selection completed without a unique "
                            f"selection_result.json under {diagnosis_dir}"
                        )
            else:
                print(f"[SKIP] round {r:02d} diagnosis/selection complete")

            result = load_json(selection_result)
            if bool(result.get("stop", False)):
                final_round = r
                payload = {
                    "dataset": args.dataset,
                    "method": args.method,
                    "final_round": final_round,
                    "final_checkpoint": str(output_ckpt),
                    "stop_result": str(selection_result),
                }
                atomic_save_json(payload, final_teacher_meta)
                break

        if final_round is None:
            raise RuntimeError(f"No stop condition reached within max_rounds={args.max_rounds}")

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
            "--overwrite",
        ], repo)
    else:
        print(f"[SKIP] final method marker exists: {final_method}")

    print(json.dumps({
        "dataset": args.dataset,
        "fold_root": str(fold_root),
        "final_round": final_round,
        "final_method": str(final_method),
        "student_view": str(args.view_root / args.method / args.dataset / args.fold),
        "status": "MEDSAM_AND_LABEL_STAGES_COMPLETE",
    }, indent=2))


if __name__ == "__main__":
    main()
