#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from idea1_common import METHOD_DEFAULT, RoundPaths, atomic_save_json, infer_is_3d, load_json, load_split_meta


def run(cmd: list[str], cwd: Path) -> None:
    print("\n[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Round0 random + iterative hard-Full MedSAM fine-tuning.")
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--fold_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--base_checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--steps_2d", type=int, default=300)
    parser.add_argument("--steps_3d", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_rounds", type=int, default=8)
    parser.add_argument("--start_round", type=int, default=0)
    parser.add_argument("--amp_init_scale", type=float, default=1024.0)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    script_root = Path(__file__).resolve().parent
    python = sys.executable
    is_3d = infer_is_3d(args.dataset, load_split_meta(args.fold_root))
    steps = args.steps_3d if is_3d else args.steps_2d
    if args.start_round < 0 or args.start_round >= args.max_rounds:
        raise ValueError("--start_round must satisfy 0 <= start_round < max_rounds")

    round0_selection = RoundPaths(args.fold_root, args.method, 0).selection_dir / "selection.json"
    if not round0_selection.exists():
        cmd = [
            python, str(script_root / "02_select_round0_random.py"),
            "--fold_root", str(args.fold_root), "--dataset", args.dataset,
            "--fold", args.fold, "--method", args.method, "--seed", str(args.seed),
        ]
        if args.overwrite:
            cmd.append("--overwrite")
        run(cmd, args.repo_root)

    if args.start_round == 0:
        checkpoint = args.base_checkpoint
    else:
        checkpoint = RoundPaths(
            args.fold_root, args.method, args.start_round - 1
        ).teacher_dir / "medsam_ft.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(
                f"Previous-round checkpoint required for resume: {checkpoint}"
            )

    final_round = None
    for round_index in range(args.start_round, args.max_rounds):
        paths = RoundPaths(args.fold_root, args.method, round_index)
        if not (paths.selection_dir / "selection.json").exists():
            raise RuntimeError(f"Selection missing for round {round_index}: {paths.selection_dir}")

        pairs_path = paths.pair_dir / "pairs.json"
        if pairs_path.exists() and not args.overwrite:
            print(f"[REUSE] finetune pairs: {pairs_path}", flush=True)
        else:
            cmd = [
                python, str(script_root / "03_build_full_finetune_pairs.py"),
                "--fold_root", str(args.fold_root), "--dataset", args.dataset,
                "--round", str(round_index), "--method", args.method,
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            run(cmd, args.repo_root)

        teacher_ckpt = paths.teacher_dir / "medsam_ft.pth"
        if teacher_ckpt.exists() and not args.overwrite:
            print(f"[REUSE] teacher checkpoint: {teacher_ckpt}", flush=True)
        else:
            cmd = [
                python, str(script_root / "04_finetune_medsam_mask_decoder.py"),
                "--repo_root", str(args.repo_root), "--fold_root", str(args.fold_root),
                "--dataset", args.dataset, "--round", str(round_index), "--method", args.method,
                "--checkpoint", str(checkpoint), "--device", args.device,
                "--seed", str(args.seed), "--max_steps", str(steps),
                "--batch_size", str(args.batch_size), "--lr", "1e-5", "--weight_decay", "0.01",
                "--dice_weight", "1.0", "--bce_weight", "1.0", "--max_grad_norm", "1.0",
                "--amp_init_scale", str(args.amp_init_scale),
            ]
            if not args.disable_amp:
                cmd.append("--amp")
            if args.overwrite:
                cmd.append("--overwrite")
            run(cmd, args.repo_root)
        checkpoint = teacher_ckpt

        metrics_path = paths.diagnosis_dir / (
            "per_case_metrics.csv" if is_3d else "per_image_metrics.csv"
        )
        if metrics_path.exists() and not args.overwrite:
            print(f"[REUSE] diagnosis metrics: {metrics_path}", flush=True)
        else:
            cmd = [
                python, str(script_root / "05_score_remaining_box_pool.py"),
                "--repo_root", str(args.repo_root),
                "--baseline_generator", str(args.repo_root / "generate_pseudo_labels.py"),
                "--fold_root", str(args.fold_root), "--dataset", args.dataset,
                "--round", str(round_index), "--method", args.method,
                "--checkpoint", str(checkpoint), "--device", args.device,
                "--threshold", "0.5", "--hard_threshold", "0.5", "--save_probability", "hard",
            ]
            run(cmd, args.repo_root)

        selection_result_path = paths.diagnosis_dir / "selection_result.json"
        if selection_result_path.exists() and not args.overwrite:
            print(f"[REUSE] selection result: {selection_result_path}", flush=True)
        else:
            cmd = [
                python, str(script_root / "06_select_next_hard_samples.py"),
                "--fold_root", str(args.fold_root), "--dataset", args.dataset,
                "--round", str(round_index), "--method", args.method,
                "--iou_threshold", "0.5", "--add_2d", "5", "--max_full_2d", "20", "--add_3d", "1",
            ]
            if args.overwrite:
                cmd.append("--overwrite")
            run(cmd, args.repo_root)
        selection_result = load_json(selection_result_path)
        if selection_result.get("stop", False):
            final_round = round_index
            break

    if final_round is None:
        raise RuntimeError(f"No stop condition reached within max_rounds={args.max_rounds}")
    final = {
        "dataset": args.dataset,
        "method": args.method,
        "final_round": final_round,
        "final_checkpoint": str(RoundPaths(args.fold_root, args.method, final_round).teacher_dir / "medsam_ft.pth"),
        "stop_result": str(RoundPaths(args.fold_root, args.method, final_round).diagnosis_dir / "selection_result.json"),
    }
    atomic_save_json(final, args.fold_root / "meta" / f"teacher_iteration_final_{args.method}.json")
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
