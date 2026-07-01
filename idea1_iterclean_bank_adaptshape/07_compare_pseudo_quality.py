from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


METHOD_DEFAULT = "idea1_sac_medsam_final"

HIGHER_IS_BETTER = {
    "coverage_non255",
    "certain_acc",
    "fg_precision",
    "fg_recall_strict",
    "dice_strict_255_as_bg",
    "iou_strict_255_as_bg",
    "dice_macro_strict",
    "iou_macro_strict",
    "largest_component_ratio",
}

LOWER_IS_BETTER = {
    "unknown_ratio",
    "false_fg_ratio",
    "wrong_class_ratio",
    "under_fg_ratio",
    "unknown_on_fg_ratio",
    "unknown_on_bg_ratio",
    "num_connected_components",
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_slice_name(item: dict[str, Any]) -> str:
    if item.get("slice_name"):
        return str(item["slice_name"])
    for key in ("teacher_img", "student_img", "teacher_gt", "student_gt"):
        if item.get(key):
            return Path(str(item[key])).name
    raise KeyError(f"Cannot infer slice_name from keys={list(item.keys())}")


def resolve_path(fold_root: Path, item: dict[str, Any], key: str) -> Path:
    value = item.get(key)
    if not value:
        raise KeyError(f"Missing manifest key '{key}' for {get_slice_name(item)}")
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
            raise ValueError(f"Unexpected mask shape {mask.shape} from {path}")
    return mask.astype(np.uint8)


def read_class_ids(meta_dir: Path, method: str) -> list[int]:
    class_ids: list[int] = []
    label_meta_path = meta_dir / "label_meta.json"
    summary_path = meta_dir / f"full_box_split_summary_{method}.json"

    if label_meta_path.exists():
        label_meta = load_json(label_meta_path)
        if isinstance(label_meta, dict):
            for key in ("unique_labels", "class_ids"):
                values = label_meta.get(key, [])
                if isinstance(values, list):
                    class_ids.extend(int(value) for value in values if int(value) > 0)

    if not class_ids and summary_path.exists():
        summary = load_json(summary_path)
        class_ids.extend(
            int(value) for value in summary.get("class_ids", []) if int(value) > 0
        )

    result = sorted(set(class_ids))
    return result or [1]


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def component_metrics(
    tri: np.ndarray,
    class_ids: list[int],
) -> tuple[int, float]:
    total_components = 0
    largest_ratios: list[float] = []

    for class_id in class_ids:
        foreground = (tri == class_id).astype(np.uint8)
        if foreground.max() == 0:
            continue
        n_labels, labels = cv2.connectedComponents(foreground)
        areas = [(labels == index).sum() for index in range(1, n_labels)]
        total_components += len(areas)
        if areas:
            largest_ratios.append(float(max(areas)) / max(float(sum(areas)), 1.0))

    return total_components, float(np.mean(largest_ratios)) if largest_ratios else 0.0


def sample_metrics(
    tri: np.ndarray,
    gt: np.ndarray,
    class_ids: list[int],
) -> dict[str, float]:
    pred_unknown = tri == 255
    pred_background = tri == 0
    pred_foreground = (tri > 0) & (~pred_unknown)
    gt_foreground = gt > 0
    gt_background = gt == 0

    certain = ~pred_unknown
    correct_certain = (tri == gt) & certain

    correct_foreground = pred_foreground & gt_foreground & (tri == gt)
    false_foreground = pred_foreground & gt_background
    wrong_class = pred_foreground & gt_foreground & (tri != gt)
    under_foreground = pred_background & gt_foreground

    pred_fg_count = int(pred_foreground.sum())
    gt_fg_count = int(gt_foreground.sum())
    correct_fg_count = int(correct_foreground.sum())

    strict_dice = safe_div(2.0 * correct_fg_count, pred_fg_count + gt_fg_count)
    strict_iou = safe_div(
        correct_fg_count,
        pred_fg_count + gt_fg_count - correct_fg_count,
    )

    per_class_dice: list[float] = []
    per_class_iou: list[float] = []
    for class_id in class_ids:
        pred_class = tri == class_id
        gt_class = gt == class_id
        pred_count = int(pred_class.sum())
        gt_count = int(gt_class.sum())
        if pred_count == 0 and gt_count == 0:
            continue
        intersection = int((pred_class & gt_class).sum())
        union = int((pred_class | gt_class).sum())
        per_class_dice.append(safe_div(2.0 * intersection, pred_count + gt_count))
        per_class_iou.append(safe_div(intersection, union))

    components, largest_ratio = component_metrics(tri, class_ids)

    return {
        "fg_ratio": float(pred_foreground.mean()),
        "bg_ratio": float(pred_background.mean()),
        "unknown_ratio": float(pred_unknown.mean()),
        "coverage_non255": float(certain.mean()),
        "certain_acc": safe_div(int(correct_certain.sum()), int(certain.sum())),
        "fg_precision": safe_div(correct_fg_count, pred_fg_count),
        "fg_recall_strict": safe_div(correct_fg_count, gt_fg_count),
        "dice_strict_255_as_bg": strict_dice,
        "iou_strict_255_as_bg": strict_iou,
        "dice_macro_strict": float(np.mean(per_class_dice)) if per_class_dice else 0.0,
        "iou_macro_strict": float(np.mean(per_class_iou)) if per_class_iou else 0.0,
        "false_fg_ratio": safe_div(int(false_foreground.sum()), pred_fg_count),
        "wrong_class_ratio": safe_div(int(wrong_class.sum()), pred_fg_count),
        "under_fg_ratio": safe_div(int(under_foreground.sum()), gt_fg_count),
        "unknown_on_fg_ratio": safe_div(int((pred_unknown & gt_foreground).sum()), gt_fg_count),
        "unknown_on_bg_ratio": safe_div(int((pred_unknown & gt_background).sum()), int(gt_background.sum())),
        "num_connected_components": float(components),
        "largest_component_ratio": largest_ratio,
    }


def validate_labels(tri: np.ndarray, class_ids: list[int]) -> int:
    allowed = {0, 255, *class_ids}
    return int(not set(np.unique(tri).tolist()).issubset(allowed))


def build_per_class_summary(
    valid_rows: list[tuple[str, str, np.ndarray, np.ndarray]],
    class_ids: list[int],
) -> pd.DataFrame:
    accumulators: dict[tuple[str, int], dict[str, int]] = {}

    for method_name, _, tri, gt in valid_rows:
        for class_id in class_ids:
            key = (method_name, class_id)
            acc = accumulators.setdefault(
                key,
                {"intersection": 0, "pred": 0, "gt": 0, "union": 0},
            )
            pred_class = tri == class_id
            gt_class = gt == class_id
            acc["intersection"] += int((pred_class & gt_class).sum())
            acc["pred"] += int(pred_class.sum())
            acc["gt"] += int(gt_class.sum())
            acc["union"] += int((pred_class | gt_class).sum())

    rows: list[dict[str, Any]] = []
    for (method_name, class_id), acc in sorted(accumulators.items()):
        rows.append(
            {
                "method": method_name,
                "class_id": class_id,
                "dice": safe_div(2.0 * acc["intersection"], acc["pred"] + acc["gt"]),
                "iou": safe_div(acc["intersection"], acc["union"]),
                "precision": safe_div(acc["intersection"], acc["pred"]),
                "recall": safe_div(acc["intersection"], acc["gt"]),
                "intersection_pixels": acc["intersection"],
                "pred_pixels": acc["pred"],
                "gt_pixels": acc["gt"],
            }
        )
    return pd.DataFrame(rows)


def paired_delta_summary(df: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    valid = df[(df["missing"] == 0) & (df["bad_label"] == 0)].copy()
    pivot = valid.pivot(index="slice_name", columns="method", values=metric_columns)
    baseline_name = "baseline_box_only"
    sac_name = "sac_medsam_final"

    rows: list[dict[str, Any]] = []
    for metric in metric_columns:
        if baseline_name not in pivot[metric].columns or sac_name not in pivot[metric].columns:
            continue

        delta = pivot[metric][sac_name] - pivot[metric][baseline_name]
        if metric in HIGHER_IS_BETTER:
            improvement = delta
            direction = "higher_is_better"
        elif metric in LOWER_IS_BETTER:
            improvement = -delta
            direction = "lower_is_better"
        else:
            improvement = pd.Series(np.nan, index=delta.index)
            direction = "descriptive_only"

        rows.append(
            {
                "metric": metric,
                "direction": direction,
                "delta_mean_sac_minus_baseline": float(delta.mean()),
                "delta_median_sac_minus_baseline": float(delta.median()),
                "sac_better_count": int((improvement > 0).sum()) if direction != "descriptive_only" else np.nan,
                "sac_worse_count": int((improvement < 0).sum()) if direction != "descriptive_only" else np.nan,
                "tie_count": int((improvement == 0).sum()) if direction != "descriptive_only" else np.nan,
                "paired_samples": int(delta.notna().sum()),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline and SAC tri-state pseudo-labels for binary or multi-class datasets."
    )
    parser.add_argument("--fold_root", type=Path, required=True)
    parser.add_argument("--baseline_tri_dir", type=Path, required=True)
    parser.add_argument("--sac_tri_dir", type=Path, required=True)
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--space", choices=("teacher", "student"), default="teacher")
    parser.add_argument("--only_box", action="store_true")
    parser.add_argument("--out_csv", type=Path, required=True)
    args = parser.parse_args()

    meta_dir = args.fold_root / "meta"
    manifest = load_json(meta_dir / "manifest.json")
    split_records = load_json(meta_dir / f"full_box_split_{args.method}.json")
    manifest_by_slice = {get_slice_name(item): item for item in manifest}
    class_ids = read_class_ids(meta_dir, args.method)

    metric_columns = [
        "fg_ratio",
        "bg_ratio",
        "unknown_ratio",
        "coverage_non255",
        "certain_acc",
        "fg_precision",
        "fg_recall_strict",
        "dice_strict_255_as_bg",
        "iou_strict_255_as_bg",
        "dice_macro_strict",
        "iou_macro_strict",
        "false_fg_ratio",
        "wrong_class_ratio",
        "under_fg_ratio",
        "unknown_on_fg_ratio",
        "unknown_on_bg_ratio",
        "num_connected_components",
        "largest_component_ratio",
    ]

    rows: list[dict[str, Any]] = []
    valid_arrays: list[tuple[str, str, np.ndarray, np.ndarray]] = []

    for split_record in split_records:
        if args.only_box and split_record["label_mode"] != "box":
            continue

        slice_name = str(split_record["slice_name"])
        item = manifest_by_slice.get(slice_name)
        if item is None:
            raise KeyError(f"Missing manifest item for {slice_name}")
        if item.get("split") != "train":
            raise RuntimeError(f"Non-train leakage: {slice_name}")

        gt_key = "teacher_gt" if args.space == "teacher" else "student_gt"
        gt = load_mask(resolve_path(args.fold_root, item, gt_key))

        for method_name, tri_dir in (
            ("baseline_box_only", args.baseline_tri_dir),
            ("sac_medsam_final", args.sac_tri_dir),
        ):
            tri_path = tri_dir / slice_name
            base_row: dict[str, Any] = {
                "slice_name": slice_name,
                "label_mode": split_record["label_mode"],
                "method": method_name,
            }
            if not tri_path.exists():
                rows.append({**base_row, "missing": 1, "bad_label": 0})
                continue

            tri = load_mask(tri_path)
            if tri.shape != gt.shape:
                raise ValueError(
                    f"Shape mismatch for {method_name}/{slice_name}: "
                    f"tri={tri.shape}, gt={gt.shape}"
                )

            bad_label = validate_labels(tri, class_ids)
            row = {
                **base_row,
                "missing": 0,
                "bad_label": bad_label,
                **sample_metrics(tri, gt, class_ids),
            }
            rows.append(row)
            if bad_label == 0:
                valid_arrays.append((method_name, slice_name, tri, gt))

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    valid_df = df[(df["missing"] == 0) & (df["bad_label"] == 0)].copy()
    paired = paired_delta_summary(df, metric_columns)
    paired_path = args.out_csv.with_name(args.out_csv.stem + "_paired_delta.csv")
    paired.to_csv(paired_path, index=False)

    per_class = build_per_class_summary(valid_arrays, class_ids)
    per_class_path = args.out_csv.with_name(args.out_csv.stem + "_per_class.csv")
    per_class.to_csv(per_class_path, index=False)

    print(f"[OK] saved sample metrics: {args.out_csv}")
    print(f"[OK] saved paired deltas: {paired_path}")
    print(f"[OK] saved per-class metrics: {per_class_path}")
    print(f"class_ids = {class_ids}")
    if not df.empty:
        print("\nMissing counts:")
        print(df.groupby("method")["missing"].sum().to_string())
    if not valid_df.empty:
        print("\nMean metrics by method:")
        print(valid_df.groupby("method")[metric_columns].mean().T.to_string())
    if not paired.empty:
        print("\nPaired SAC - baseline summary:")
        print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
