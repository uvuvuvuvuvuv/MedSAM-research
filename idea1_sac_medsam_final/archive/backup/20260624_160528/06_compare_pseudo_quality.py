import argparse
import json
import os
import numpy as np
import pandas as pd


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mask(path):
    m = np.load(path)
    if m.ndim == 3:
        if m.shape[-1] == 1:
            m = m[..., 0]
        elif m.shape[0] == 1:
            m = m[0]
    return m.astype(np.uint8)


def binary_metrics(tri, gt, label_id=1):
    pred_fg = (tri == label_id)
    pred_bg = (tri == 0)
    pred_unk = (tri == 255)

    gt_fg = (gt == label_id)
    gt_bg = (gt != label_id)

    inter = np.logical_and(pred_fg, gt_fg).sum()
    pred_fg_sum = pred_fg.sum()
    gt_fg_sum = gt_fg.sum()
    union = np.logical_or(pred_fg, gt_fg).sum()

    certain = ~pred_unk
    certain_sum = certain.sum()
    correct_certain = ((tri == gt) & certain).sum()

    fg_precision = inter / max(pred_fg_sum, 1)
    fg_recall = inter / max(gt_fg_sum, 1)
    dice_strict = (2 * inter) / max(pred_fg_sum + gt_fg_sum, 1)
    iou_strict = inter / max(union, 1)

    false_fg_ratio = np.logical_and(pred_fg, gt_bg).sum() / max(pred_fg_sum, 1)
    under_fg_ratio = np.logical_and(pred_bg, gt_fg).sum() / max(gt_fg_sum, 1)
    unknown_on_fg_ratio = np.logical_and(pred_unk, gt_fg).sum() / max(gt_fg_sum, 1)
    unknown_on_bg_ratio = np.logical_and(pred_unk, gt_bg).sum() / max(gt_bg.sum(), 1)

    certain_acc = correct_certain / max(certain_sum, 1)
    coverage = certain.mean()

    return {
        "fg_ratio": float(pred_fg.mean()),
        "bg_ratio": float(pred_bg.mean()),
        "unknown_ratio": float(pred_unk.mean()),
        "coverage_non255": float(coverage),
        "certain_acc": float(certain_acc),
        "fg_precision": float(fg_precision),
        "fg_recall_strict": float(fg_recall),
        "dice_strict_255_as_bg": float(dice_strict),
        "iou_strict_255_as_bg": float(iou_strict),
        "false_fg_ratio": float(false_fg_ratio),
        "under_fg_ratio": float(under_fg_ratio),
        "unknown_on_fg_ratio": float(unknown_on_fg_ratio),
        "unknown_on_bg_ratio": float(unknown_on_bg_ratio),
    }


def count_components(tri, label_id=1):
    import cv2
    fg = (tri == label_id).astype(np.uint8)
    if fg.max() == 0:
        return 0, 0.0
    n, lab = cv2.connectedComponents(fg)
    if n <= 1:
        return 0, 0.0
    areas = [(lab == i).sum() for i in range(1, n)]
    total = float(sum(areas))
    largest = float(max(areas))
    return n - 1, largest / max(total, 1.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold_root", required=True)
    parser.add_argument("--baseline_tri_dir", required=True)
    parser.add_argument("--sac_tri_dir", required=True)
    parser.add_argument("--method", default="idea1_sac_medsam_final")
    parser.add_argument("--space", choices=["teacher", "student"], default="teacher")
    parser.add_argument("--only_box", action="store_true")
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    meta_dir = os.path.join(args.fold_root, "meta")
    manifest = load_json(os.path.join(meta_dir, "manifest.json"))
    split_records = load_json(os.path.join(meta_dir, f"full_box_split_{args.method}.json"))

    manifest_by_slice = {x["slice_name"]: x for x in manifest}
    split_by_slice = {x["slice_name"]: x for x in split_records}

    rows = []

    for slice_name, split_rec in split_by_slice.items():
        if args.only_box and split_rec["label_mode"] != "box":
            continue

        item = manifest_by_slice[slice_name]
        if item.get("split") != "train":
            raise RuntimeError(f"Non-train leakage: {slice_name}")

        if args.space == "teacher":
            gt_path = os.path.join(args.fold_root, item["teacher_gt"])
        else:
            gt_path = os.path.join(args.fold_root, item["student_gt"])

        gt = load_mask(gt_path)

        for method_name, tri_dir in [
            ("baseline_box_only", args.baseline_tri_dir),
            ("sac_medsam_final", args.sac_tri_dir),
        ]:
            tri_path = os.path.join(tri_dir, slice_name)
            if not os.path.exists(tri_path):
                rows.append({
                    "slice_name": slice_name,
                    "label_mode": split_rec["label_mode"],
                    "method": method_name,
                    "missing": 1,
                })
                continue

            tri = load_mask(tri_path)
            if tri.shape != gt.shape:
                raise ValueError(
                    f"shape mismatch: {method_name} {slice_name}, tri={tri.shape}, gt={gt.shape}"
                )

            u = set(np.unique(tri).tolist())
            bad_label = int(not u.issubset({0, 1, 255}))

            m = binary_metrics(tri, gt, label_id=1)
            cc_n, largest_ratio = count_components(tri, label_id=1)

            row = {
                "slice_name": slice_name,
                "label_mode": split_rec["label_mode"],
                "method": method_name,
                "missing": 0,
                "bad_label": bad_label,
                "num_connected_components": int(cc_n),
                "largest_component_ratio": float(largest_ratio),
            }
            row.update(m)
            rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    print("[OK] saved:", args.out_csv)
    print("rows:", len(df))
    print("missing:")
    print(df.groupby("method")["missing"].sum())

    ok = df[df["missing"] == 0].copy()

    print("\n=== summary by method ===")
    cols = [
        "fg_ratio", "unknown_ratio", "coverage_non255",
        "certain_acc",
        "fg_precision", "fg_recall_strict",
        "dice_strict_255_as_bg", "iou_strict_255_as_bg",
        "false_fg_ratio", "under_fg_ratio",
        "unknown_on_fg_ratio", "unknown_on_bg_ratio",
        "num_connected_components", "largest_component_ratio",
    ]
    print(ok.groupby("method")[cols].mean().T.to_string())

    # paired delta: SAC - baseline
    pivot = ok.pivot(index="slice_name", columns="method", values=cols)
    paired_rows = []
    for c in cols:
        if ("sac_medsam_final" in pivot[c].columns) and ("baseline_box_only" in pivot[c].columns):
            delta = pivot[c]["sac_medsam_final"] - pivot[c]["baseline_box_only"]
            paired_rows.append({
                "metric": c,
                "delta_mean_sac_minus_baseline": float(delta.mean()),
                "delta_median_sac_minus_baseline": float(delta.median()),
                "sac_better_count": int((delta > 0).sum()),
                "sac_worse_count": int((delta < 0).sum()),
                "tie_count": int((delta == 0).sum()),
            })

    paired = pd.DataFrame(paired_rows)
    delta_csv = args.out_csv.replace(".csv", "_paired_delta.csv")
    paired.to_csv(delta_csv, index=False)

    print("\n=== paired delta: SAC - baseline ===")
    print(paired.to_string(index=False))
    print("[OK] saved delta:", delta_csv)


if __name__ == "__main__":
    main()
