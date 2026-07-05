from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Import from shared modules
# ---------------------------------------------------------------------------
try:
    from .pipeline_common import (
        METHOD_DEFAULT,
        build_fold_paths,
        is_3d_dataset,
        save_json_atomic,
        validate_writable_path,
    )
except ImportError:
    from pipeline_common import (  # type: ignore[no-redef]
        METHOD_DEFAULT,
        build_fold_paths,
        is_3d_dataset,
        save_json_atomic,
        validate_writable_path,
    )


UNKNOWN_LABEL = 255


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(
    rows: list[dict[str, Any]],
    path: Path,
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_slice_name(item: dict[str, Any]) -> str:
    if item.get("slice_name"):
        return str(item["slice_name"])

    for key in ("teacher_img", "student_img", "teacher_gt", "student_gt"):
        if item.get(key):
            return Path(str(item[key])).name

    raise KeyError(
        f"Cannot infer slice_name from keys={list(item.keys())}"
    )


def get_case_id(
    item: dict[str, Any],
    *,
    required: bool = False,
) -> str:
    value = item.get("case_id")

    if value is None or not str(value).strip():
        if required:
            raise KeyError(
                "3D sample has no valid case_id: "
                f"{get_slice_name(item)}"
            )
        return "unknown_case"

    return str(value)


def resolve_path(
    fold_root: Path,
    item: dict[str, Any],
    key: str,
) -> Path:
    value = item.get(key)

    if not value:
        raise KeyError(
            f"Missing manifest key '{key}' for {get_slice_name(item)}"
        )

    path = Path(str(value))
    if not path.is_absolute():
        path = fold_root / path

    if not path.exists():
        raise FileNotFoundError(path)

    return path


def load_mask(path: Path) -> np.ndarray:
    mask = np.load(path)

    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask[..., 0]
        elif mask.shape[0] == 1:
            mask = mask[0]
        else:
            raise ValueError(
                f"Unexpected mask shape {mask.shape} from {path}"
            )

    if mask.ndim != 2:
        raise ValueError(
            f"Expected a 2D mask, got {mask.shape} from {path}"
        )

    return mask.astype(np.uint8)


def infer_is_3d(
    dataset: str,
    split_meta: dict[str, Any],
) -> bool:
    if is_3d_dataset(dataset):
        return True

    for key in ("is_3d", "volume_level_split", "has_volume"):
        if key in split_meta:
            return bool(split_meta[key])

    return False


def parse_class_ids(label_meta: Any) -> list[int]:
    if not isinstance(label_meta, dict):
        return [1]

    candidates: list[int] = []

    for key in ("unique_labels", "class_ids"):
        values = label_meta.get(key)
        if not isinstance(values, list):
            continue

        for value in values:
            try:
                class_id = int(value)
            except (TypeError, ValueError):
                continue

            if 0 < class_id < UNKNOWN_LABEL:
                candidates.append(class_id)

    labels = label_meta.get("labels")
    if isinstance(labels, dict):
        for value in labels.keys():
            try:
                class_id = int(value)
            except (TypeError, ValueError):
                continue

            if 0 < class_id < UNKNOWN_LABEL:
                candidates.append(class_id)

    if not candidates and "num_classes" in label_meta:
        num_classes = int(label_meta["num_classes"])
        candidates.extend(range(1, max(num_classes, 2)))

    class_ids = sorted(set(candidates))
    return class_ids or [1]


def build_unique_index(
    records: list[dict[str, Any]],
    source_name: str,
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}

    for record in records:
        if not isinstance(record, dict):
            raise TypeError(
                f"Each record in {source_name} must be a dict"
            )

        slice_name = get_slice_name(record)

        if slice_name in index:
            raise RuntimeError(
                f"Duplicate slice_name in {source_name}: {slice_name}"
            )

        index[slice_name] = record

    return index


def compute_binary_metrics(
    pseudo: np.ndarray,
    gt: np.ndarray,
) -> tuple[float, float, int, int, int]:
    pred_fg = (pseudo > 0) & (pseudo != UNKNOWN_LABEL)
    gt_fg = gt > 0

    intersection = int(np.logical_and(pred_fg, gt_fg).sum())
    union = int(np.logical_or(pred_fg, gt_fg).sum())
    pred_pixels = int(pred_fg.sum())
    gt_pixels = int(gt_fg.sum())

    iou = 1.0 if union == 0 else intersection / union

    denominator = pred_pixels + gt_pixels
    dice = (
        1.0
        if denominator == 0
        else (2.0 * intersection) / denominator
    )

    return (
        float(iou),
        float(dice),
        intersection,
        pred_pixels,
        gt_pixels,
    )


def compute_macro_class_iou(
    pseudo: np.ndarray,
    gt: np.ndarray,
    class_ids: list[int],
) -> tuple[float, int]:
    present_classes = [
        class_id
        for class_id in class_ids
        if np.any(gt == class_id)
    ]

    if not present_classes:
        return float("nan"), 0

    values: list[float] = []

    for class_id in present_classes:
        pred_class = pseudo == class_id
        gt_class = gt == class_id

        intersection = int(
            np.logical_and(pred_class, gt_class).sum()
        )
        union = int(
            np.logical_or(pred_class, gt_class).sum()
        )

        values.append(
            1.0 if union == 0 else intersection / union
        )

    return float(np.mean(values)), len(present_classes)


def compute_slice_metrics(
    pseudo: np.ndarray,
    gt: np.ndarray,
    class_ids: list[int],
    is_multiclass: bool,
) -> dict[str, Any]:
    if pseudo.shape != gt.shape:
        raise ValueError(
            "Pseudo/GT shape mismatch: "
            f"pseudo={pseudo.shape}, gt={gt.shape}"
        )

    pseudo_values = set(np.unique(pseudo).tolist())
    allowed_values = {0, UNKNOWN_LABEL, *class_ids}
    invalid_values = pseudo_values - allowed_values

    if invalid_values:
        raise ValueError(
            f"Pseudo contains invalid values: {sorted(invalid_values)}"
        )

    (
        binary_iou,
        dice,
        intersection,
        pseudo_fg_pixels,
        gt_fg_pixels,
    ) = compute_binary_metrics(pseudo, gt)

    macro_iou, num_present_classes = compute_macro_class_iou(
        pseudo,
        gt,
        class_ids,
    )

    unknown_pixels = int((pseudo == UNKNOWN_LABEL).sum())
    unknown_ratio = unknown_pixels / max(int(pseudo.size), 1)

    main_iou = (
        macro_iou
        if is_multiclass and not math.isnan(macro_iou)
        else binary_iou
    )

    return {
        "main_iou": float(main_iou),
        "binary_fg_iou": float(binary_iou),
        "macro_class_iou": float(macro_iou),
        "dice": float(dice),
        "intersection_pixels": int(intersection),
        "pseudo_fg_pixels": int(pseudo_fg_pixels),
        "gt_fg_pixels": int(gt_fg_pixels),
        "unknown_pixels": int(unknown_pixels),
        "unknown_ratio": float(unknown_ratio),
        "gt_has_foreground": bool(gt_fg_pixels > 0),
        "num_present_classes": int(num_present_classes),
    }


def csv_float(value: float) -> float | str:
    return "" if math.isnan(value) else float(value)


def worst20_mean(values: list[float]) -> float:
    if not values:
        return float("nan")

    values = sorted(values)
    count = max(1, int(math.ceil(len(values) * 0.20)))
    return float(np.mean(values[:count]))


def check_output_paths(
    paths: list[Path],
    overwrite: bool,
) -> None:
    existing = [path for path in paths if path.exists()]

    if existing and not overwrite:
        raise FileExistsError(
            "Outputs already exist. Use --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )


def process_dataset(
    args: argparse.Namespace,
    dataset: str,
) -> None:
    paths = build_fold_paths(
        processed_root=args.processed_root,
        dataset=dataset,
        fold=args.fold,
        method=args.method,
        round_tag=args.round_tag,
    )
    fold_root = paths["fold_root"]
    meta_dir = paths["meta_dir"]
    run_id = paths["run_id"]

    manifest_path = meta_dir / "manifest.json"
    split_meta_path = meta_dir / "split_meta.json"
    label_meta_path = meta_dir / "label_meta.json"
    split_path = paths["split_path"]
    pseudo_dir = paths["pseudo_teacher_dir"]

    for required_path in (
        manifest_path,
        split_path,
        pseudo_dir,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required input does not exist: {required_path}"
            )

    manifest = load_json(manifest_path)
    split_records = load_json(split_path)
    split_meta = (
        load_json(split_meta_path)
        if split_meta_path.exists()
        else {}
    )
    label_meta = (
        load_json(label_meta_path)
        if label_meta_path.exists()
        else {}
    )

    if not isinstance(manifest, list):
        raise TypeError("manifest.json must contain a list")
    if not isinstance(split_records, list):
        raise TypeError("Full/Box split must contain a list")
    if not isinstance(split_meta, dict):
        raise TypeError("split_meta.json must contain a JSON object")

    manifest_by_slice = build_unique_index(
        manifest,
        "manifest.json",
    )
    split_by_slice = build_unique_index(
        split_records,
        split_path.name,
    )

    class_ids = parse_class_ids(label_meta)
    is_multiclass = len(class_ids) > 1
    is_3d = infer_is_3d(dataset, split_meta)

    train_names = {
        name
        for name, item in manifest_by_slice.items()
        if item.get("split") == "train"
    }

    if set(split_by_slice) != train_names:
        raise RuntimeError(
            "Full/Box split does not exactly match train manifest"
        )

    invalid_modes = {
        name: record.get("label_mode")
        for name, record in split_by_slice.items()
        if str(record.get("label_mode")) not in {"full", "box"}
    }
    if invalid_modes:
        raise ValueError(
            f"Invalid label_mode values: {list(invalid_modes.items())[:10]}"
        )

    current_full_slices = {
        name
        for name, record in split_by_slice.items()
        if str(record["label_mode"]) == "full"
    }
    current_box_slices = {
        name
        for name, record in split_by_slice.items()
        if str(record["label_mode"]) == "box"
    }

    if args.select_count > len(current_box_slices):
        raise ValueError(
            f"select_count={args.select_count} exceeds "
            f"remaining Box samples={len(current_box_slices)}"
        )

    slice_rows: list[dict[str, Any]] = []
    missing_pseudo: list[str] = []
    missing_gt: list[str] = []

    for slice_name in sorted(current_box_slices):
        item = manifest_by_slice[slice_name]

        pseudo_path = pseudo_dir / (
            slice_name
            if slice_name.endswith(".npy")
            else f"{slice_name}.npy"
        )

        if not pseudo_path.exists():
            missing_pseudo.append(slice_name)
            continue

        try:
            gt_path = resolve_path(
                fold_root,
                item,
                "teacher_gt",
            )
        except (KeyError, FileNotFoundError):
            missing_gt.append(slice_name)
            continue

        pseudo = load_mask(pseudo_path)
        gt = load_mask(gt_path)

        metrics = compute_slice_metrics(
            pseudo,
            gt,
            class_ids,
            is_multiclass,
        )

        slice_rows.append(
            {
                "slice_name": slice_name,
                "case_id": get_case_id(
                    item,
                    required=is_3d,
                ),
                "slice_idx": item.get("slice_idx", ""),
                "label_mode": "box",
                **metrics,
            }
        )

    if missing_pseudo or missing_gt:
        raise RuntimeError(
            "Scoring inputs are incomplete:\n"
            + json.dumps(
                {
                    "missing_pseudo_count": len(missing_pseudo),
                    "missing_pseudo_examples": missing_pseudo[:10],
                    "missing_gt_count": len(missing_gt),
                    "missing_gt_examples": missing_gt[:10],
                },
                indent=2,
                ensure_ascii=False,
            )
        )

    if len(slice_rows) != len(current_box_slices):
        raise RuntimeError(
            "Scored sample count mismatch: "
            f"expected={len(current_box_slices)}, "
            f"actual={len(slice_rows)}"
        )

    next_round = args.round_id + 1
    hard_selection_path = paths["hard_selection_path"]
    hard_summary_path = paths["hard_summary_path"]
    next_selection_path = (
        meta_dir / f"full_selection_round_{next_round}.json"
    )

    if is_3d:
        slice_ranking_path = (
            meta_dir / f"hard_slice_ranking_{run_id}.csv"
        )
        case_ranking_path = (
            meta_dir / f"hard_case_ranking_{run_id}.csv"
        )
        for out_path in (
            hard_selection_path,
            hard_summary_path,
            next_selection_path,
            slice_ranking_path,
            case_ranking_path,
        ):
            validate_writable_path(out_path)
        check_output_paths(
            [
                hard_selection_path,
                hard_summary_path,
                next_selection_path,
                slice_ranking_path,
                case_ranking_path,
            ],
            args.overwrite,
        )

        case_to_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in slice_rows:
            case_to_rows[str(row["case_id"])].append(row)

        case_rows: list[dict[str, Any]] = []

        for case_id, rows in sorted(case_to_rows.items()):
            valid_rows = [
                row
                for row in rows
                if bool(row["gt_has_foreground"])
            ]
            scores = [
                float(row["main_iou"])
                for row in valid_rows
            ]

            case_rows.append(
                {
                    "case_id": case_id,
                    "num_total_slices": len(rows),
                    "num_foreground_slices": len(valid_rows),
                    "num_empty_slices": len(rows) - len(valid_rows),
                    "mean_iou": (
                        float(np.mean(scores))
                        if scores
                        else float("nan")
                    ),
                    "median_iou": (
                        float(np.median(scores))
                        if scores
                        else float("nan")
                    ),
                    "worst20_mean_iou": worst20_mean(scores),
                    "valid_for_selection": bool(scores),
                }
            )

        valid_cases = [
            row
            for row in case_rows
            if bool(row["valid_for_selection"])
        ]

        if args.select_count > len(valid_cases):
            raise ValueError(
                f"select_count={args.select_count} exceeds "
                f"valid Box cases={len(valid_cases)}"
            )

        valid_cases.sort(
            key=lambda row: (
                float(row["mean_iou"]),
                float(row["median_iou"]),
                str(row["case_id"]),
            )
        )

        selected_cases = {
            str(row["case_id"])
            for row in valid_cases[: args.select_count]
        }

        sorted_case_rows = sorted(
            case_rows,
            key=lambda row: (
                0 if row["valid_for_selection"] else 1,
                (
                    float(row["mean_iou"])
                    if row["valid_for_selection"]
                    else float("inf")
                ),
                str(row["case_id"]),
            ),
        )

        case_csv_rows: list[dict[str, Any]] = []
        rank = 0
        for row in sorted_case_rows:
            if row["valid_for_selection"]:
                rank += 1
                rank_value: int | str = rank
            else:
                rank_value = ""

            case_csv_rows.append(
                {
                    "rank": rank_value,
                    "case_id": row["case_id"],
                    "num_total_slices": row["num_total_slices"],
                    "num_foreground_slices": row[
                        "num_foreground_slices"
                    ],
                    "num_empty_slices": row["num_empty_slices"],
                    "mean_iou": csv_float(float(row["mean_iou"])),
                    "median_iou": csv_float(float(row["median_iou"])),
                    "worst20_mean_iou": csv_float(
                        float(row["worst20_mean_iou"])
                    ),
                    "valid_for_selection": row[
                        "valid_for_selection"
                    ],
                    "selected": str(row["case_id"])
                    in selected_cases,
                }
            )

        slice_rows.sort(
            key=lambda row: (
                float(row["main_iou"]),
                float(row["dice"]),
                str(row["slice_name"]),
            )
        )

        slice_csv_rows: list[dict[str, Any]] = []
        for rank, row in enumerate(slice_rows, start=1):
            slice_csv_rows.append(
                {
                    "rank": rank,
                    "slice_name": row["slice_name"],
                    "case_id": row["case_id"],
                    "slice_idx": row["slice_idx"],
                    "label_mode": row["label_mode"],
                    "main_iou": row["main_iou"],
                    "binary_fg_iou": row["binary_fg_iou"],
                    "macro_class_iou": csv_float(
                        float(row["macro_class_iou"])
                    ),
                    "dice": row["dice"],
                    "gt_has_foreground": row["gt_has_foreground"],
                    "num_present_classes": row[
                        "num_present_classes"
                    ],
                    "gt_fg_pixels": row["gt_fg_pixels"],
                    "pseudo_fg_pixels": row["pseudo_fg_pixels"],
                    "unknown_pixels": row["unknown_pixels"],
                    "unknown_ratio": row["unknown_ratio"],
                    "selected_case": str(row["case_id"])
                    in selected_cases,
                }
            )

        save_csv(
            slice_csv_rows,
            slice_ranking_path,
            [
                "rank",
                "slice_name",
                "case_id",
                "slice_idx",
                "label_mode",
                "main_iou",
                "binary_fg_iou",
                "macro_class_iou",
                "dice",
                "gt_has_foreground",
                "num_present_classes",
                "gt_fg_pixels",
                "pseudo_fg_pixels",
                "unknown_pixels",
                "unknown_ratio",
                "selected_case",
            ],
        )

        save_csv(
            case_csv_rows,
            case_ranking_path,
            [
                "rank",
                "case_id",
                "num_total_slices",
                "num_foreground_slices",
                "num_empty_slices",
                "mean_iou",
                "median_iou",
                "worst20_mean_iou",
                "valid_for_selection",
                "selected",
            ],
        )

        new_slices = {
            name
            for name in train_names
            if get_case_id(
                manifest_by_slice[name],
                required=True,
            )
            in selected_cases
        }

        current_full_cases = {
            get_case_id(
                manifest_by_slice[name],
                required=True,
            )
            for name in current_full_slices
        }

        cumulative_cases = current_full_cases | selected_cases
        cumulative_slices = current_full_slices | new_slices

        selection_payload = {
            "schema_version": 1,
            "dataset": dataset,
            "fold": args.fold,
            "source_method": args.method,
            "source_round_id": int(args.round_id),
            "round_id": int(next_round),
            "selection_strategy": "gt_iou_hard",
            "sampling_unit": "case",
            "score_space": "teacher",
            "score_name": (
                "mean_main_iou_on_gt_foreground_slices"
            ),
            "class_ids": class_ids,
            "previous_full_count": len(current_full_cases),
            "new_selected_count": len(selected_cases),
            "cumulative_full_count": len(cumulative_cases),
            "new_selected_case_ids": sorted(selected_cases),
            "cumulative_full_case_ids": sorted(cumulative_cases),
            "new_selected_slice_names": sorted(new_slices),
            "cumulative_full_slice_names": sorted(
                cumulative_slices
            ),
        }

        hard_payload = {
            **selection_payload,
            "selected_case_details": [
                row for row in case_csv_rows if row["selected"]
            ],
            "slice_ranking_path": str(slice_ranking_path),
            "case_ranking_path": str(case_ranking_path),
        }

        previous_full_units = len(current_full_cases)
        cumulative_full_units = len(cumulative_cases)
        selected_units = sorted(selected_cases)
        candidate_units = len(valid_cases)

    else:
        ranking_path = (
            meta_dir / f"hard_ranking_{run_id}.csv"
        )
        for out_path in (
            hard_selection_path,
            hard_summary_path,
            next_selection_path,
            ranking_path,
        ):
            validate_writable_path(out_path)
        check_output_paths(
            [
                hard_selection_path,
                hard_summary_path,
                next_selection_path,
                ranking_path,
            ],
            args.overwrite,
        )

        slice_rows.sort(
            key=lambda row: (
                float(row["main_iou"]),
                float(row["dice"]),
                str(row["slice_name"]),
            )
        )

        selected_slices = {
            str(row["slice_name"])
            for row in slice_rows[: args.select_count]
        }

        ranking_rows: list[dict[str, Any]] = []
        for rank, row in enumerate(slice_rows, start=1):
            ranking_rows.append(
                {
                    "rank": rank,
                    "slice_name": row["slice_name"],
                    "case_id": row["case_id"],
                    "label_mode": row["label_mode"],
                    "main_iou": row["main_iou"],
                    "binary_fg_iou": row["binary_fg_iou"],
                    "macro_class_iou": csv_float(
                        float(row["macro_class_iou"])
                    ),
                    "dice": row["dice"],
                    "gt_has_foreground": row["gt_has_foreground"],
                    "num_present_classes": row[
                        "num_present_classes"
                    ],
                    "gt_fg_pixels": row["gt_fg_pixels"],
                    "pseudo_fg_pixels": row["pseudo_fg_pixels"],
                    "unknown_pixels": row["unknown_pixels"],
                    "unknown_ratio": row["unknown_ratio"],
                    "selected": str(row["slice_name"])
                    in selected_slices,
                }
            )

        save_csv(
            ranking_rows,
            ranking_path,
            [
                "rank",
                "slice_name",
                "case_id",
                "label_mode",
                "main_iou",
                "binary_fg_iou",
                "macro_class_iou",
                "dice",
                "gt_has_foreground",
                "num_present_classes",
                "gt_fg_pixels",
                "pseudo_fg_pixels",
                "unknown_pixels",
                "unknown_ratio",
                "selected",
            ],
        )

        cumulative_slices = current_full_slices | selected_slices

        selection_payload = {
            "schema_version": 1,
            "dataset": dataset,
            "fold": args.fold,
            "source_method": args.method,
            "source_round_id": int(args.round_id),
            "round_id": int(next_round),
            "selection_strategy": "gt_iou_hard",
            "sampling_unit": "image",
            "score_space": "teacher",
            "score_name": (
                "macro_class_iou"
                if is_multiclass
                else "binary_fg_iou"
            ),
            "class_ids": class_ids,
            "previous_full_count": len(current_full_slices),
            "new_selected_count": len(selected_slices),
            "cumulative_full_count": len(cumulative_slices),
            "new_selected_slice_names": sorted(selected_slices),
            "cumulative_full_slice_names": sorted(
                cumulative_slices
            ),
            "new_selected_case_ids": None,
            "cumulative_full_case_ids": None,
        }

        hard_payload = {
            **selection_payload,
            "selected_image_details": [
                row for row in ranking_rows if row["selected"]
            ],
            "ranking_path": str(ranking_path),
        }

        previous_full_units = len(current_full_slices)
        cumulative_full_units = len(cumulative_slices)
        selected_units = sorted(selected_slices)
        candidate_units = len(slice_rows)

    if (
        cumulative_full_units
        != previous_full_units + args.select_count
    ):
        raise RuntimeError(
            "Cumulative Full count mismatch: "
            f"previous={previous_full_units}, "
            f"selected={args.select_count}, "
            f"cumulative={cumulative_full_units}"
        )

    summary_payload = {
        "schema_version": 1,
        "dataset": dataset,
        "fold": args.fold,
        "method": args.method,
        "round_tag": args.round_tag,
        "run_id": run_id,
        "round_id": int(args.round_id),
        "next_round_id": int(next_round),
        "is_3d": bool(is_3d),
        "sampling_unit": "case" if is_3d else "image",
        "score_space": "teacher",
        "class_ids": class_ids,
        "num_train_items": len(train_names),
        "num_current_full_slices": len(current_full_slices),
        "num_current_box_slices": len(current_box_slices),
        "num_scored_box_slices": len(slice_rows),
        "num_candidate_units": candidate_units,
        "select_count": int(args.select_count),
        "previous_full_unit_count": previous_full_units,
        "cumulative_full_unit_count": cumulative_full_units,
        "new_selected_units": selected_units,
        "pseudo_dir": str(pseudo_dir),
        "current_selection_path": str(hard_selection_path),
        "next_full_selection_path": str(next_selection_path),
        "gt_usage": (
            "Teacher-space train GT is used only in this selector "
            "as a proxy for doctor judgment."
        ),
        "test_samples_used": 0,
        "selection_complete": True,
    }

    save_json_atomic(hard_payload, hard_selection_path)
    save_json_atomic(selection_payload, next_selection_path)
    save_json_atomic(summary_payload, hard_summary_path)

    print(f"[OK] hard-sample selection: {dataset}/{args.fold}")
    print(
        f"     round={args.round_id} -> {next_round} "
        f"is_3d={is_3d}"
    )
    print(
        f"     current_full_units={previous_full_units} "
        f"selected={args.select_count} "
        f"next_full_units={cumulative_full_units}"
    )
    print(f"     selected={selected_units}")

    if is_3d:
        print(f"     slice_ranking={slice_ranking_path}")
        print(f"     case_ranking={case_ranking_path}")
    else:
        print(f"     ranking={ranking_path}")

    print(f"     hard_selection={hard_selection_path}")
    print(f"     next_selection={next_selection_path}")
    print(f"     summary={hard_summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select the lowest-IoU remaining Box samples using "
            "Teacher-space train GT as a doctor-proxy signal. "
            "2D selects images; 3D selects complete cases."
        )
    )

    parser.add_argument(
        "--processed_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated dataset names",
    )
    parser.add_argument(
        "--fold",
        default="fold_0",
    )
    parser.add_argument(
        "--method",
        default=METHOD_DEFAULT,
        help="Method identifier (default: %(default)s)",
    )
    parser.add_argument(
        "--round_tag",
        required=True,
        help="Round tag, e.g. r00_full5 (2D) or r00_case1 (3D)",
    )
    parser.add_argument(
        "--round_id",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--select_count",
        type=int,
        required=True,
        help=(
            "2D: number of new images. "
            "3D: number of new complete cases."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.round_id < 0:
        raise ValueError("--round_id must be >= 0")

    if args.select_count <= 0:
        raise ValueError("--select_count must be positive")


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)

    datasets = [
        name.strip()
        for name in args.datasets.split(",")
        if name.strip()
    ]

    if not datasets:
        raise ValueError("--datasets is empty")

    for dataset in datasets:
        process_dataset(args, dataset)


if __name__ == "__main__":
    main()
