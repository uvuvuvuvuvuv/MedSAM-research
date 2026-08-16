#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import (
    METHOD_DEFAULT,
    RoundPaths,
    atomic_save_json,
    component_mask_native,
    full_and_box_from_selection,
    get_case_id,
    get_slice_name,
    geometry_item,
    infer_is_3d,
    instance_native_box,
    instance_teacher_box,
    load_geometry,
    load_manifest,
    load_npy_2d,
    load_prompts,
    load_selection,
    load_split_meta,
    map_native_mask_to_target,
    manifest_index,
    prompt_instances,
    resolve_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Full-only MedSAM image-box-component-mask pairs.")
    parser.add_argument("--fold_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--connectivity", type=int, choices=[4, 8], default=8)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    paths = RoundPaths(args.fold_root, args.method, args.round)
    selection_path = args.selection or (paths.selection_dir / "selection.json")
    selection = load_selection(selection_path)
    manifest = load_manifest(args.fold_root)
    prompts = load_prompts(args.fold_root)
    geometry = load_geometry(args.fold_root)
    split_meta = load_split_meta(args.fold_root)
    is_3d = infer_is_3d(args.dataset, split_meta)
    full_names, _, _ = full_and_box_from_selection(selection, manifest, is_3d)
    mindex = manifest_index(manifest)

    pair_dir = paths.pair_dir
    masks_dir = pair_dir / "masks"
    pairs_path = pair_dir / "pairs.json"
    if pairs_path.exists() and not args.overwrite:
        raise FileExistsError(pairs_path)
    if args.overwrite and masks_dir.exists():
        import shutil
        shutil.rmtree(masks_dir)
    masks_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[dict] = []
    skipped_empty = 0
    for slice_name in sorted(full_names):
        item = mindex[slice_name]
        meta = prompts.get(slice_name)
        if not isinstance(meta, dict):
            raise RuntimeError(f"Missing prompt record for Full sample: {slice_name}")
        instances = prompt_instances(meta)
        if not instances:
            skipped_empty += 1
            continue
        gt_native_path = resolve_path(args.fold_root, item, "native_gt")
        gt_native = load_npy_2d(gt_native_path)
        geom = geometry_item(geometry, slice_name)
        teacher_img_path = resolve_path(args.fold_root, item, "teacher_img")

        for instance_index, ins in enumerate(instances):
            label_id = int(ins.get("label_id", 1))
            component_id = ins.get("component_id")
            component_id = int(component_id) if component_id is not None else None
            native_box = instance_native_box(ins)
            component_native = component_mask_native(
                gt_native=gt_native,
                label_id=label_id,
                component_id=component_id,
                native_box=native_box,
                connectivity=args.connectivity,
            )
            target_teacher = map_native_mask_to_target(component_native, geom, "native_to_teacher")
            if target_teacher.max(initial=0) == 0:
                raise RuntimeError(f"Mapped teacher target is empty: {slice_name}, instance={instance_index}")
            mask_name = f"{Path(slice_name).stem}__i{instance_index:03d}_c{label_id}.npy"
            mask_path = masks_dir / mask_name
            np.save(mask_path, target_teacher.astype(np.uint8))
            pairs.append(
                {
                    "pair_id": f"{Path(slice_name).stem}__i{instance_index:03d}_c{label_id}",
                    "slice_name": slice_name,
                    "case_id": get_case_id(item),
                    "label_id": label_id,
                    "component_id": component_id,
                    "teacher_img": str(teacher_img_path),
                    "target_mask": str(mask_path),
                    "bbox_teacher": instance_teacher_box(ins),
                    "bbox_native": native_box,
                }
            )

    if not pairs:
        raise RuntimeError("No Full fine-tuning pairs were generated")
    atomic_save_json(pairs, pairs_path)
    audit = {
        "dataset": args.dataset,
        "round": args.round,
        "method": args.method,
        "selection": str(selection_path),
        "is_3d": is_3d,
        "num_full_samples_or_slices": len(full_names),
        "num_pairs": len(pairs),
        "num_empty_full_slices_skipped_for_medsam": skipped_empty,
        "target_definition": "one connected GT component per tight baseline box",
    }
    atomic_save_json(audit, pair_dir / "audit.json")
    print(json.dumps({"pairs": len(pairs), "output": str(pairs_path)}, indent=2))


if __name__ == "__main__":
    main()
