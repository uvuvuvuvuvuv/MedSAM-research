#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from common import (
    METHOD_DEFAULT,
    RoundPaths,
    atomic_save_json,
    binary_iou,
    box_mask,
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
    load_rgb_uint8,
    load_selection,
    load_split_meta,
    map_native_mask_to_target,
    map_teacher_mask_to_native,
    manifest_index,
    prompt_instances,
    resolve_path,
    write_csv,
)


def load_baseline_generator(path: Path):
    # V2_DYNAMIC_GENERATOR_IMPORT_PATH
    generator_path = Path(path).resolve()
    generator_repo_root = generator_path.parent
    repo_root_str = str(generator_repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    spec = importlib.util.spec_from_file_location("frozen_baseline_generate_pseudo_labels", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load baseline generator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_predictor(checkpoint: Path, repo_root: Path, device: str):
    sys.path.insert(0, str(repo_root.resolve()))
    from segment_anything import SamPredictor, sam_model_registry

    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    model.to(device=device)
    model.eval()
    return SamPredictor(model)


# Student/Teacher V2:
# Probability diagnostics must come from the SAME formal proposal
# selected by generate_pseudo_labels.select_and_build_proposal().
# Do not issue a second independent predictor.predict() here.


def save_probability_bundle(
    output: Path,
    probabilities: list[np.ndarray],
    boxes: list[list[float]],
    labels: list[int],
    ious: list[float],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "boxes": np.asarray(boxes, dtype=np.float32),
        "label_ids": np.asarray(labels, dtype=np.int32),
        "ious": np.asarray(ious, dtype=np.float32),
    }
    for idx, probability in enumerate(probabilities):
        payload[f"prob_{idx:03d}"] = probability.astype(np.float16)
    np.savez_compressed(output, **payload)


def save_probability_visual(image: np.ndarray, probability: np.ndarray, box: list[float], path: Path) -> None:
    heat = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    base = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score remaining Box pool with current MedSAM and hidden train GT.")
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--baseline_generator", type=Path, required=True)
    parser.add_argument("--fold_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--hard_threshold", type=float, default=0.5)
    parser.add_argument("--save_probability", choices=["none", "hard"], default="hard")
    args = parser.parse_args()

    if abs(args.threshold - 0.5) > 1e-12:
        raise ValueError("Idea1 contract fixes probability threshold at 0.5")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; using CPU")
        args.device = "cpu"

    paths = RoundPaths(args.fold_root, args.method, args.round)
    selection_path = args.selection or (paths.selection_dir / "selection.json")
    selection = load_selection(selection_path)
    manifest = load_manifest(args.fold_root)
    prompts = load_prompts(args.fold_root)
    geometry = load_geometry(args.fold_root)
    split_meta = load_split_meta(args.fold_root)
    is_3d = infer_is_3d(args.dataset, split_meta)
    _, box_names, _ = full_and_box_from_selection(selection, manifest, is_3d)
    mindex = manifest_index(manifest)

    sys.path.insert(0, str(args.repo_root.resolve()))
    baseline = load_baseline_generator(args.baseline_generator)
    predictor = build_predictor(args.checkpoint, args.repo_root, args.device)
    cfg = baseline.get_dataset_cfg(args.dataset)
    is_multiclass = bool(baseline.infer_is_multiclass(args.dataset, str(args.fold_root)))

    instance_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    case_acc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "class_intersection": defaultdict(int),
            "class_union": defaultdict(int),
            "class_gt": defaultdict(int),
            "class_pred": defaultdict(int),
            "slice_class_ious": [],
            "num_slices": 0,
        }
    )
    start = time.perf_counter()

    for slice_name in sorted(box_names):
        item = mindex[slice_name]
        meta = prompts.get(slice_name)
        if not isinstance(meta, dict):
            raise RuntimeError(f"Missing prompt for Box sample: {slice_name}")
        instances = prompt_instances(meta)
        image = load_rgb_uint8(resolve_path(args.fold_root, item, "teacher_img"))
        gt_native = load_npy_2d(resolve_path(args.fold_root, item, "native_gt"))
        geom = geometry_item(geometry, slice_name)
        enhanced = baseline.enhance_image_by_dataset(image, cfg.enhancement)
        predictor.set_image(enhanced)

        per_ious: list[float] = []
        probabilities: list[np.ndarray] = []
        probability_boxes: list[list[float]] = []
        probability_labels: list[int] = []
        class_pred_native: dict[int, np.ndarray] = {}
        empty_count = 0

        for instance_index, ins in enumerate(instances):
            label_id = int(ins.get("label_id", 1)) if is_multiclass else 1
            teacher_box = instance_teacher_box(ins)
            native_box = instance_native_box(ins)
            component_id = ins.get("component_id")
            component_id = int(component_id) if component_id is not None else None
            component_native = component_mask_native(
                gt_native,
                label_id,
                component_id,
                native_box,
                connectivity=8,
            )
            component_teacher = map_native_mask_to_target(component_native, geom, "native_to_teacher") > 0

            prop = baseline.select_and_build_proposal(
                predictor=predictor,
                teacher_box=np.asarray(teacher_box, dtype=np.float32),
                native_box=(np.asarray(native_box, dtype=np.float32) if native_box is not None else None),
                label_id=label_id,
                point_coords=None,
                point_labels=None,
                cfg=cfg,
                teacher_hw=(image.shape[0], image.shape[1]),
                is_multiclass=is_multiclass,
            )
            if prop is None:
                pred_teacher = np.zeros(
                    component_teacher.shape,
                    dtype=bool,
                )
                probability = np.zeros(
                    component_teacher.shape,
                    dtype=np.float32,
                )
                empty_count += 1
                proposal_score = float("nan")
                model_score = float("nan")
            else:
                pred_teacher = np.asarray(
                    prop.best_mask,
                    dtype=bool,
                )
                probability = np.asarray(
                    prop.best_probability,
                    dtype=np.float32,
                )
                proposal_score = float(
                    prop.score
                )
                model_score = float(
                    prop.best_model_score
                )

                if probability.shape != pred_teacher.shape:
                    raise RuntimeError(
                        "Formal proposal mask/probability "
                        "shape mismatch: "
                        f"{pred_teacher.shape} vs "
                        f"{probability.shape}"
                    )

            # Keep the frozen Box constraint for formal selection.
            pred_teacher &= box_mask(
                teacher_box,
                image.shape[0],
                image.shape[1],
            )

            # Probability is diagnostic only, but it now belongs
            # to exactly the same selected multimask candidate.
            probability = np.where(
                pred_teacher,
                probability,
                0.0,
            ).astype(np.float32)
            iou = binary_iou(pred_teacher, component_teacher)
            if not np.isfinite(iou):
                iou = 0.0
            per_ious.append(iou)

            pred_native = map_teacher_mask_to_native(pred_teacher.astype(np.uint8), geom) > 0
            if label_id not in class_pred_native:
                class_pred_native[label_id] = pred_native.copy()
            else:
                class_pred_native[label_id] |= pred_native

            probabilities.append(
                probability
            )
            probability_boxes.append(
                teacher_box
            )
            probability_labels.append(
                label_id
            )
            instance_rows.append(
                {
                    "dataset": args.dataset,
                    "round": args.round,
                    "slice_name": slice_name,
                    "case_id": get_case_id(item),
                    "instance_index": instance_index,
                    "label_id": label_id,
                    "component_id": component_id if component_id is not None else "",
                    "iou": iou,
                    "prediction_empty": int(pred_teacher.sum() == 0),
                    "proposal_score": proposal_score,
                    "model_score_probability_map": model_score,
                    "gt_pixels_teacher": int(component_teacher.sum()),
                    "pred_pixels_teacher": int(pred_teacher.sum()),
                }
            )

        image_macro = (
            float(np.mean(per_ious))
            if per_ious
            else 1.0
        )

        image_min = (
            float(np.min(per_ious))
            if per_ious
            else 1.0
        )

        image_all_pass = int(
            image_min >= args.hard_threshold
        )

        image_rows.append(
            {
                "dataset": args.dataset,
                "round": args.round,
                "slice_name": slice_name,
                "case_id": get_case_id(item),
                "num_instances": len(instances),

                # Diagnostic only.
                "image_macro_iou": image_macro,

                # Active Learning V2 criterion.
                "image_min_iou": image_min,
                "all_instance_iou_ge_threshold": (
                    image_all_pass
                ),

                "empty_instance_count": empty_count,

                "hard_candidate": int(
                    image_min < args.hard_threshold
                ),
            }
        )

        if args.save_probability == "hard" and per_ious and min(per_ious) < args.hard_threshold:
            raw_path = paths.diagnosis_dir / "probability_raw" / f"{Path(slice_name).stem}.npz"
            save_probability_bundle(raw_path, probabilities, probability_boxes, probability_labels, per_ious)
            worst = int(np.argmin(per_ious))
            save_probability_visual(
                image,
                probabilities[worst],
                probability_boxes[worst],
                paths.diagnosis_dir / "probability_vis" / f"{Path(slice_name).stem}.png",
            )

        if is_3d:
            case_id = get_case_id(item, required=True)
            acc = case_acc[case_id]
            acc["num_slices"] += 1
            gt_labels = [int(x) for x in np.unique(gt_native) if int(x) > 0 and int(x) != 255]
            for label_id in gt_labels:
                gt_class = gt_native == label_id
                pred_class = class_pred_native.get(label_id, np.zeros_like(gt_class, dtype=bool))
                inter = int(np.logical_and(gt_class, pred_class).sum())
                union = int(np.logical_or(gt_class, pred_class).sum())
                acc["class_intersection"][label_id] += inter
                acc["class_union"][label_id] += union
                acc["class_gt"][label_id] += int(gt_class.sum())
                acc["class_pred"][label_id] += int(pred_class.sum())
                if union > 0:
                    acc["slice_class_ious"].append(inter / union)

    case_rows: list[dict[str, Any]] = []
    if is_3d:
        for case_id in sorted(case_acc):
            acc = case_acc[case_id]
            class_ious = []
            missed = 0
            for label_id in sorted(acc["class_gt"]):
                if acc["class_gt"][label_id] <= 0:
                    continue
                union = acc["class_union"][label_id]
                iou = acc["class_intersection"][label_id] / union if union > 0 else 0.0
                class_ious.append(float(iou))
                if acc["class_pred"][label_id] == 0:
                    missed += 1
            case_macro = (
                float(np.mean(class_ious))
                if class_ious
                else 1.0
            )

            ordered = sorted(
                float(x)
                for x in acc["slice_class_ious"]
            )

            worst_n = (
                max(
                    1,
                    int(math.ceil(
                        0.20 * len(ordered)
                    )),
                )
                if ordered
                else 0
            )

            worst20 = (
                float(np.mean(
                    ordered[:worst_n]
                ))
                if worst_n
                else 1.0
            )

            # Active Learning V2:
            # convergence requires EVERY evaluated
            # slice/class target in the case to meet
            # the IoU threshold.
            case_min = (
                float(ordered[0])
                if ordered
                else 1.0
            )

            case_all_pass = int(
                case_min >= args.hard_threshold
                and missed == 0
            )

            case_rows.append(
                {
                    "dataset": args.dataset,
                    "round": args.round,
                    "case_id": case_id,
                    "num_slices": acc["num_slices"],
                    "num_gt_classes": len(class_ious),
                    "num_slice_class_evaluations": (
                        len(ordered)
                    ),

                    # Diagnostic statistics.
                    "case_macro_3d_iou": case_macro,
                    "worst20_slice_class_iou": worst20,

                    # Active Learning V2 criterion.
                    "case_min_slice_class_iou": (
                        case_min
                    ),
                    "all_slice_class_iou_ge_threshold": (
                        case_all_pass
                    ),

                    "empty_prediction_classes": missed,

                    "hard_candidate": int(
                        case_min < args.hard_threshold
                        or missed > 0
                    ),
                }
            )

    instance_fields = [
        "dataset", "round", "slice_name", "case_id", "instance_index", "label_id", "component_id",
        "iou", "prediction_empty", "proposal_score", "model_score_probability_map",
        "gt_pixels_teacher", "pred_pixels_teacher",
    ]
    image_fields = [
        "dataset",
        "round",
        "slice_name",
        "case_id",
        "num_instances",
        "image_macro_iou",
        "image_min_iou",
        "all_instance_iou_ge_threshold",
        "empty_instance_count",
        "hard_candidate",
    ]
    case_fields = [
        "dataset",
        "round",
        "case_id",
        "num_slices",
        "num_gt_classes",
        "num_slice_class_evaluations",
        "case_macro_3d_iou",
        "worst20_slice_class_iou",
        "case_min_slice_class_iou",
        "all_slice_class_iou_ge_threshold",
        "empty_prediction_classes",
        "hard_candidate",
    ]
    write_csv(instance_rows, paths.diagnosis_dir / "per_instance_metrics.csv", instance_fields)
    write_csv(image_rows, paths.diagnosis_dir / "per_image_metrics.csv", image_fields)
    if is_3d:
        write_csv(case_rows, paths.diagnosis_dir / "per_case_metrics.csv", case_fields)

    # ======================================================
    # Active Learning V2 convergence audit
    #
    # This is NOT the final stopping decision because 3D also
    # has a cumulative annotation budget. Stage 06 combines
    # this signal with the remaining Full budget.
    # ======================================================

    if is_3d:
        remaining_all_pass = all(
            int(row["hard_candidate"]) == 0
            for row in case_rows
        )

        minimum_remaining_iou = min(
            (
                float(row["case_min_slice_class_iou"])
                for row in case_rows
            ),
            default=1.0,
        )

        num_evaluation_units = len(case_rows)

    else:
        remaining_all_pass = all(
            int(row["hard_candidate"]) == 0
            for row in image_rows
        )

        minimum_remaining_iou = min(
            (
                float(row["image_min_iou"])
                for row in image_rows
            ),
            default=1.0,
        )

        num_evaluation_units = len(image_rows)

    elapsed = time.perf_counter() - start

    summary = {
        "dataset": args.dataset,
        "round": args.round,
        "method": args.method,
        "checkpoint": str(args.checkpoint),
        "selection": str(selection_path),
        "threshold": 0.5,

        "active_learning_protocol": (
            "active_learning_v2"
        ),

        "difficulty_rule": (
            "minimum_iou_below_threshold"
        ),

        "convergence_rule": (
            "all_remaining_gt_targets_iou_ge_threshold"
        ),

        "selection_mask_source": (
            "generate_pseudo_labels."
            "select_and_build_proposal"
        ),

        "probability_map_role": (
            "diagnosis_only_same_selected_candidate"
        ),

        "num_box_slices": len(box_names),
        "num_instances": len(instance_rows),

        "num_evaluation_units": (
            num_evaluation_units
        ),

        "minimum_remaining_iou": float(
            minimum_remaining_iou
        ),

        "all_remaining_targets_pass": bool(
            remaining_all_pass
        ),

        "num_hard_images": sum(
            int(x["hard_candidate"])
            for x in image_rows
        ),

        "num_hard_cases": sum(
            int(x["hard_candidate"])
            for x in case_rows
        ),

        "elapsed_seconds": elapsed,
    }
    atomic_save_json(summary, paths.diagnosis_dir / "diagnosis_summary.json")
    atomic_save_json(
        {"stage": "score_remaining_box_pool", "elapsed_seconds": elapsed, "num_outputs": len(instance_rows)},
        paths.diagnosis_dir / "stage_time_score.json",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
