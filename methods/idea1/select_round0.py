#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import (
    METHOD_DEFAULT,
    RoundPaths,
    atomic_save_json,
    build_case_to_slices,
    current_timestamp,
    foreground_slice_names,
    get_slice_name,
    infer_is_3d,
    load_json,
    load_manifest,
    load_prompts,
    load_split_meta,
    prompt_instances,
    seeded_sample,
    train_items,
    validate_partition,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Round 0 fixed random Full selection.")
    parser.add_argument("--fold_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = RoundPaths(args.fold_root, args.method, 0)
    output = paths.selection_dir / "selection.json"
    if output.exists() and not args.overwrite:
        raise FileExistsError(output)

    manifest = load_manifest(args.fold_root)
    prompts = load_prompts(args.fold_root)
    split_meta = load_split_meta(args.fold_root)
    train = train_items(manifest)
    train_names = {get_slice_name(x) for x in train}
    is_3d = infer_is_3d(args.dataset, split_meta)

    budget_path = (
        args.fold_root
        / "meta"
        / f"annotation_budget_{args.method}.json"
    )

    if not budget_path.exists():
        raise FileNotFoundError(
            "Active-learning annotation budget is missing: "
            f"{budget_path}. "
            "Run init_workspace.py first."
        )

    budget = load_json(
        budget_path
    )

    if budget.get("budget_version") != "active_learning_v2":
        raise RuntimeError(
            "Round0 requires budget_version="
            "'active_learning_v2', got "
            f"{budget.get('budget_version')!r}"
        )

    budget_is_3d = bool(
        budget.get("is_3d", False)
    )

    if budget_is_3d != is_3d:
        raise RuntimeError(
            "Dataset dimensionality disagrees with "
            f"annotation budget: is_3d={is_3d}, "
            f"budget_is_3d={budget_is_3d}"
        )

    round0_quota = int(
        budget["round0"]
    )

    max_full = int(
        budget["max_full"]
    )

    if round0_quota <= 0:
        raise RuntimeError(
            f"Invalid Round0 quota: {round0_quota}"
        )

    if max_full <= 0:
        raise RuntimeError(
            f"Invalid cumulative Full budget: {max_full}"
        )

    if is_3d:
        case_to_slices = build_case_to_slices(train)
        eligible_cases = [
            case_id
            for case_id, names in case_to_slices.items()
            if any(name in prompts and len(prompt_instances(prompts[name])) > 0 for name in names)
        ]
        round0_count = min(
            round0_quota,
            max_full,
            len(eligible_cases),
        )

        if round0_count <= 0:
            raise RuntimeError(
                "No eligible 3D case can be selected "
                "for Round0."
            )

        selected_cases = seeded_sample(
            eligible_cases,
            round0_count,
            args.seed,
        )

        full_names = {
            name
            for cid in selected_cases
            for name in case_to_slices[cid]
        }

        selected_new = selected_cases
    else:
        eligible = foreground_slice_names(
            prompts,
            train_names,
        )

        round0_count = min(
            round0_quota,
            len(eligible),
        )

        if round0_count <= 0:
            raise RuntimeError(
                "No eligible 2D image can be selected "
                "for Round0."
            )

        selected_slices = seeded_sample(
            eligible,
            round0_count,
            args.seed,
        )

        selected_cases = []
        full_names = set(selected_slices)
        selected_new = selected_slices

    box_names = train_names - full_names
    validate_partition(train_names, full_names, box_names)
    payload = {
        "selection_version": "active_learning_v2",
        "created_at": current_timestamp(),
        "dataset": args.dataset,
        "fold": args.fold,
        "method": args.method,
        "round": 0,
        "selection_type": "random_round0",
        "seed": args.seed,
        "is_3d": is_3d,

        "annotation_budget_version": (
            budget["budget_version"]
        ),
        "annotation_unit": (
            budget["annotation_unit"]
        ),
        "per_round_ratio": float(
            budget["per_round_ratio"]
        ),
        "minimum_per_round": int(
            budget.get("minimum_per_round") or 0
        ),
        "requested_round_quota": int(
            budget["requested_round_quota"]
        ),
        "effective_round0_quota": int(
            round0_quota
        ),
        "cumulative_full_budget": int(
            max_full
        ),
        "three_d_max_ratio": (
            budget.get("three_d_max_ratio")
        ),

        "selected_new_ids": selected_new,
        "cumulative_full_case_ids": selected_cases,
        "cumulative_full_slice_names": sorted(full_names),
        "remaining_box_slice_names": sorted(box_names),
        "num_full_slices": len(full_names),
        "num_box_slices": len(box_names),
        "num_full_cases": len(selected_cases),
    }
    atomic_save_json(payload, output)
    atomic_save_json(
        {
            "seed": args.seed,
            "budget_version": "active_learning_v2",
            "round0_quota": round0_quota,
            "actual_selected_count": len(selected_new),
        },
        paths.selection_dir / "random_seed.json",
    )
    atomic_save_json(selected_new, paths.selection_dir / "selected_new_ids.json")
    atomic_save_json(sorted(full_names), paths.selection_dir / "cumulative_full_ids.json")
    atomic_save_json(sorted(box_names), paths.selection_dir / "remaining_box_ids.json")
    print(json.dumps({"output": str(output), "selected": selected_new}, indent=2))


if __name__ == "__main__":
    main()
