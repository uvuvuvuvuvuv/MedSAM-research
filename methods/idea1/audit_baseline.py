#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

from common import (
    ContractError,
    atomic_save_json,
    current_timestamp,
    get_case_id,
    get_slice_name,
    infer_is_3d,
    load_geometry,
    load_manifest,
    load_prompts,
    load_split_meta,
    prompt_instances,
    resolve_spacing_zyx,
    sha256_file,
    test_items,
    train_items,
)

REQUIRED_BASELINE_FILES = [
    "generate_prompts.py",
    "generate_pseudo_labels.py",
]


def parse_python(path: Path) -> None:
    try:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        raise ContractError(f"Python syntax failure: {path}: {exc}") from exc


def audit_generator_contract(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    checks = {
        "has_tri_inside_unknown_255": bool(re.search(r"tri(?:_map)?\s*\[.*union.*\]\s*=\s*255", text)),
        "has_confirmed_foreground_write": "confirmed_fg" in text and "tri" in text,
        "has_conflict_to_255": "conflict" in text and "255" in text,
        "uses_sam_predictor": "SamPredictor" in text,
        "supports_checkpoint_argument": "--checkpoint" in text,
        "supports_prompt_name_argument": "--prompt_name" in text,
        "supports_train_split": "--split" in text,
        "does_not_require_historical_sac": "idea1_sac_medsam" not in text,
    }
    # SamPredictor converts mask logits to masks at model.mask_threshold=0.0,
    # which corresponds to sigmoid probability 0.5. The baseline file may not
    # contain a literal "0.5", so the audit records the mechanism explicitly.
    checks["single_threshold_semantics"] = (
        "predictor.predict" in text and ("masks[i] > 0" in text or "masks > 0" in text)
    )
    checks["all_required_true"] = all(checks.values())
    return checks


def audit_dataset(frozen_processed_root: Path, dataset: str, fold: str) -> dict[str, Any]:
    fold_root = frozen_processed_root / dataset / fold
    manifest = load_manifest(fold_root)
    prompts = load_prompts(fold_root)
    geometry = load_geometry(fold_root)
    split_meta = load_split_meta(fold_root)
    train = train_items(manifest)
    test = test_items(manifest)
    train_names = {get_slice_name(x) for x in train}
    test_names = {get_slice_name(x) for x in test}
    prompt_names = set(prompts)
    is_3d = infer_is_3d(dataset, split_meta)

    errors: list[str] = []
    if train_names & test_names:
        errors.append("manifest train/test overlap")
    if prompt_names - train_names:
        errors.append(f"prompt contains non-train samples: {sorted(prompt_names-train_names)[:5]}")
    missing_prompt = train_names - prompt_names
    # The canonical baseline prompt file should include empty 3D slices explicitly.
    if missing_prompt:
        errors.append(f"train samples missing prompt records: {sorted(missing_prompt)[:5]}")
    missing_geometry = train_names - set(geometry)
    if missing_geometry:
        errors.append(f"train samples missing geometry: {sorted(missing_geometry)[:5]}")

    num_prompt_instances = 0
    invalid_prompt_split = 0
    for name, meta in prompts.items():
        if not isinstance(meta, dict):
            errors.append(f"prompt record is not object: {name}")
            continue
        if str(meta.get("split", "train")) != "train":
            invalid_prompt_split += 1
        num_prompt_instances += len(prompt_instances(meta))
    if invalid_prompt_split:
        errors.append(f"prompt split mismatch count={invalid_prompt_split}")

    case_count = 0
    spacing_ok = True
    spacing_source_counts: dict[str, int] = {}
    if is_3d:
        case_ids = {get_case_id(x, required=True) for x in train}
        case_count = len(case_ids)
        for item in train:
            name = get_slice_name(item)
            geom = geometry.get(name, {})
            if not isinstance(geom, dict):
                geom = {}
            try:
                spacing, source = resolve_spacing_zyx(item, geom)
            except ContractError as exc:
                spacing_ok = False
                errors.append(f"missing/invalid physical spacing: {name}: {exc}")
                break
            spacing_source_counts[source] = spacing_source_counts.get(source, 0) + 1

    forbidden = [
        fold_root / "prompts" / "prompts_test.json",
        fold_root / "pseudo_student" / "tri_test",
        fold_root / "pseudo_teacher" / "tri_test",
    ]
    forbidden_present = [str(x) for x in forbidden if x.exists()]
    if forbidden_present:
        errors.append(f"forbidden test artifacts present: {forbidden_present}")

    boxonly_student = fold_root / "pseudo_student" / "tri_train"
    boxonly_teacher = fold_root / "pseudo_teacher" / "tri_train"
    if not boxonly_student.is_dir():
        errors.append(f"baseline student pseudo missing: {boxonly_student}")
    if not boxonly_teacher.is_dir():
        errors.append(f"baseline teacher pseudo missing: {boxonly_teacher}")

    return {
        "dataset": dataset,
        "fold": fold,
        "fold_root": str(fold_root),
        "is_3d": is_3d,
        "num_manifest": len(manifest),
        "num_train": len(train),
        "num_test": len(test),
        "num_train_cases": case_count,
        "num_prompt_records": len(prompts),
        "num_prompt_instances": num_prompt_instances,
        "spacing_zyx_ok": spacing_ok,
        "spacing_source_counts": spacing_source_counts,
        "boxonly_student_pseudo": str(boxonly_student),
        "boxonly_teacher_pseudo": str(boxonly_teacher),
        "errors": errors,
        "passed": len(errors) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen baseline before Idea1 writes any output.")
    parser.add_argument("--baseline_repo", type=Path, required=True)
    parser.add_argument("--frozen_processed_root", type=Path, required=True)
    parser.add_argument("--datasets", required=True, help="comma-separated")
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow_fail", action="store_true")
    args = parser.parse_args()

    file_records: dict[str, Any] = {}
    for rel in REQUIRED_BASELINE_FILES:
        path = args.baseline_repo / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        parse_python(path)
        file_records[rel] = {"path": str(path), "sha256": sha256_file(path)}

    generator_contract = audit_generator_contract(args.baseline_repo / "generate_pseudo_labels.py")
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    dataset_records = [audit_dataset(args.frozen_processed_root, ds, args.fold) for ds in datasets]
    passed = generator_contract["all_required_true"] and all(x["passed"] for x in dataset_records)
    report = {
        "audit_version": "idea1_baseline_contract_v1",
        "created_at": current_timestamp(),
        "baseline_repo": str(args.baseline_repo.resolve()),
        "frozen_processed_root": str(args.frozen_processed_root.resolve()),
        "files": file_records,
        "generator_contract": generator_contract,
        "datasets": dataset_records,
        "passed": passed,
    }
    atomic_save_json(report, args.output)
    print(json.dumps({"passed": passed, "output": str(args.output)}, indent=2))
    if not passed and not args.allow_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
