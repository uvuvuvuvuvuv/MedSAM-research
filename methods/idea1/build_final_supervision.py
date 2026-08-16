#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import METHOD_DEFAULT, RoundPaths


def run(cmd: list[str], cwd: Path) -> None:
    print("\n[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final pseudo, hybrid supervision, student view and validation.")
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--fold_root", type=Path, required=True)
    parser.add_argument("--view_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--final_round", type=int, required=True)
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--pseudo_protocol",
        default="student_v2_probability_arbitration",
        choices=["student_v2_probability_arbitration"],
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    python = sys.executable
    checkpoint = args.checkpoint or RoundPaths(args.fold_root, args.method, args.final_round).teacher_dir / "medsam_ft.pth"

    commands = [
        [python, str(root / "generate_final_pseudo.py"), "--repo_root", str(args.repo_root),
         "--fold_root", str(args.fold_root), "--dataset", args.dataset, "--fold", args.fold,
         "--final_round", str(args.final_round), "--method", args.method,
         "--checkpoint", str(checkpoint),
         "--pseudo_protocol", args.pseudo_protocol],
        [python, str(root / "assemble_supervision.py"), "--fold_root", str(args.fold_root),
         "--dataset", args.dataset, "--final_round", str(args.final_round), "--method", args.method],
        [python, str(root / "build_student_view.py"), "--fold_root", str(args.fold_root),
         "--view_root", str(args.view_root), "--dataset", args.dataset, "--fold", args.fold,
         "--method", args.method, "--build_boxonly"],
        [python, str(root / "validate_data.py"), "--fold_root", str(args.fold_root),
         "--dataset", args.dataset, "--final_round", str(args.final_round), "--method", args.method],
    ]
    for cmd in commands:
        if args.overwrite and Path(cmd[1]).name in {
            "generate_final_pseudo.py", "assemble_supervision.py", "build_student_view.py"
        }:
            cmd.append("--overwrite")
        run(cmd, args.repo_root)

    student_view = args.view_root / args.method / args.dataset / args.fold
    run(
        [python, str(root / "finalize.py"), "--fold_root", str(args.fold_root),
         "--dataset", args.dataset, "--final_round", str(args.final_round), "--method", args.method,
         "--student_view", str(student_view)],
        args.repo_root,
    )


if __name__ == "__main__":
    main()
