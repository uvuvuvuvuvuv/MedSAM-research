#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from idea1_common import (
    METHOD_DEFAULT,
    RoundPaths,
    atomic_save_json,
    copy_or_symlink,
    full_and_box_from_selection,
    get_slice_name,
    infer_is_3d,
    load_manifest,
    load_prompts,
    load_selection,
    load_split_meta,
    train_items,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final Box-only pseudo with frozen baseline generator in an isolated view.")
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--fold_root", type=Path, required=True, help="Idea1 workspace fold")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--final_round", type=int, required=True)
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--pseudo_protocol",
        default="student_v2_probability_arbitration",
        choices=["student_v2_probability_arbitration"],
        help=(
            "Locked V2 tri-pseudo protocol: original refined "
            "MedSAM support + pixel-probability multiclass arbitration."
        ),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = RoundPaths(args.fold_root, args.method, args.final_round)
    selection_path = args.selection or (paths.selection_dir / "selection.json")
    selection = load_selection(selection_path)
    manifest = load_manifest(args.fold_root)
    prompts = load_prompts(args.fold_root)
    split_meta = load_split_meta(args.fold_root)
    is_3d = infer_is_3d(args.dataset, split_meta)
    _, box_names, _ = full_and_box_from_selection(selection, manifest, is_3d)

    # Isolated generation view. The frozen baseline generator writes canonical
    # tri_train directories here, never into the frozen baseline or final method directory.
    generation_base = args.fold_root / "_generation_views" / args.method / "data"
    generation_fold = generation_base / "processed" / args.dataset / args.fold
    if generation_fold.exists() and args.overwrite:
        shutil.rmtree(generation_fold)
    generation_fold.mkdir(parents=True, exist_ok=True)

    for name in ("native_npy", "teacher_npy", "student_npy"):
        copy_or_symlink(args.fold_root / name, generation_fold / name, "symlink", overwrite=True)
    copy_or_symlink(args.fold_root / "meta", generation_fold / "meta", "copy", overwrite=True)
    (generation_fold / "prompts").mkdir(parents=True, exist_ok=True)

    filtered_prompts = {name: prompts[name] for name in sorted(box_names) if name in prompts}
    missing = sorted(box_names - set(filtered_prompts))
    if missing:
        raise RuntimeError(f"Box samples missing prompt records: {missing[:10]}")
    prompt_name = f"prompts_train_{args.method}_box_only.json"
    atomic_save_json(filtered_prompts, generation_fold / "prompts" / prompt_name)

    # Use the Student/Teacher V2 generator shipped with this
    # experiment repository.  repo_root remains the frozen
    # MedSAM runtime root and must not be overwritten.
    generator = (
        Path(__file__).resolve().parent.parent
        / "generate_pseudo_labels.py"
    )
    if not generator.is_file():
        raise FileNotFoundError(generator)
    cmd = [
        args.python,
        str(generator),
        "--base_dir",
        str(generation_base),
        "--checkpoint",
        str(args.checkpoint),
        "--datasets",
        args.dataset,
        "--fold",
        args.fold,
        "--split",
        "train",
        "--overwrite",
        "--vis_limit",
        "0",
        "--prompt_name",
        prompt_name,
        "--stage_tag",
        args.method,
    ]
    print("[RUN]", " ".join(cmd))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.repo_root.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, cwd=args.repo_root, env=env, check=True)

    outputs = {}
    for space in ("pseudo_teacher", "pseudo_student"):
        src = generation_fold / space / "tri_train"
        dst = args.fold_root / space / f"tri_box_final_{args.method}"
        if not src.is_dir():
            raise RuntimeError(f"Baseline generator did not create {src}")
        if dst.exists():
            if not args.overwrite:
                raise FileExistsError(dst)
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        outputs[space] = str(dst)

    expected = len(box_names)
    actual = len(list((args.fold_root / "pseudo_student" / f"tri_box_final_{args.method}").glob("*.npy")))
    if actual != expected:
        raise RuntimeError(f"Final Box pseudo count mismatch: expected={expected}, actual={actual}")
    summary = {
        "dataset": args.dataset,
        "method": args.method,
        "final_round": args.final_round,
        "selection": str(selection_path),
        "checkpoint": str(args.checkpoint),
        "num_box_samples": expected,
        "baseline_generator": str(generator),
        "generation_view": str(generation_fold),
        "outputs": outputs,
        "pseudo_protocol": args.pseudo_protocol,
        "contract": {
            "foreground_support": "selected_refined_medsam_best_mask",
            "probability_source": "same_selected_multimask_candidate",
            "same_class_overlap": "support_or_probability_max",
            "cross_class_overlap": "pixel_probability_argmax",
            "numerical_tie": 255,
            "inside_box_unconfirmed": 255,
            "outside_box_union": 0,
            "additional_margin_threshold": None,
        },
    }
    atomic_save_json(summary, args.fold_root / "meta" / f"final_box_pseudo_{args.method}.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
