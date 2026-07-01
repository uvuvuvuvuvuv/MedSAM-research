import os
import sys
import json
import csv
import shutil
import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def parse_float_list(s):
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_dataset_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def get_records(split_obj):
    if isinstance(split_obj, list):
        return split_obj
    if isinstance(split_obj, dict):
        for key in ["records", "items", "data"]:
            if key in split_obj and isinstance(split_obj[key], list):
                return split_obj[key]
    raise TypeError("Unsupported full_box_split json format.")


def replace_records(split_obj, new_records):
    if isinstance(split_obj, list):
        return new_records
    out = dict(split_obj)
    for key in ["records", "items", "data"]:
        if key in out and isinstance(out[key], list):
            out[key] = new_records
            return out
    raise TypeError("Unsupported full_box_split json format.")


def get_slice_name(record):
    for key in ["slice_name", "geometry_key", "name", "filename"]:
        if key in record:
            return record[key]
    raise KeyError(f"Cannot find slice name in record: {record.keys()}")


def copy_if_exists(src, dst):
    if os.path.exists(src):
        ensure_dir(os.path.dirname(dst))
        shutil.copy2(src, dst)


def build_manifest_index(fold_root):
    manifest = load_json(os.path.join(fold_root, "meta", "manifest.json"))
    return {item["slice_name"]: item for item in manifest}


def resolve_path(fold_root, rel_or_abs):
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(fold_root, rel_or_abs)


def load_gt(path):
    arr = np.load(path)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


def count_components(mask_fg):
    mask_fg = mask_fg.astype(np.uint8)
    if mask_fg.max() == 0:
        return 0
    num, _ = cv2.connectedComponents(mask_fg)
    return max(0, int(num) - 1)


def clean_tri_by_components(tri, clean_ratio):
    """
    Remove small foreground components.
    Removed foreground pixels are set to 255, not 0.
    This avoids forcing uncertain regions to background.
    """
    tri = tri.astype(np.uint8).copy()
    if clean_ratio <= 0:
        return tri, {
            "removed_fg_pixels": 0,
            "original_fg_pixels": int(((tri > 0) & (tri != 255)).sum()),
            "kept_fg_pixels": int(((tri > 0) & (tri != 255)).sum()),
        }

    removed_total = 0
    original_total = int(((tri > 0) & (tri != 255)).sum())

    labels = [int(x) for x in np.unique(tri).tolist() if x not in [0, 255]]

    for label_id in labels:
        fg = (tri == label_id).astype(np.uint8)
        if fg.max() == 0:
            continue

        num, cc = cv2.connectedComponents(fg)
        if num <= 2:
            continue

        areas = []
        for i in range(1, num):
            areas.append((i, int((cc == i).sum())))

        if not areas:
            continue

        max_area = max(a for _, a in areas)
        min_keep_area = max(1, int(round(max_area * clean_ratio)))

        keep = np.zeros_like(fg, dtype=bool)
        for comp_id, area in areas:
            if area >= min_keep_area:
                keep |= (cc == comp_id)

        remove = (fg > 0) & (~keep)
        removed_total += int(remove.sum())
        tri[remove] = 255

    kept_total = int(((tri > 0) & (tri != 255)).sum())
    return tri, {
        "removed_fg_pixels": removed_total,
        "original_fg_pixels": original_total,
        "kept_fg_pixels": kept_total,
    }


def eval_one_pred(tri, gt):
    pred_fg = (tri > 0) & (tri != 255)
    unknown = tri == 255
    gt_fg = gt > 0

    tp = np.logical_and(pred_fg, gt_fg).sum()
    fp = np.logical_and(pred_fg, ~gt_fg).sum()
    fn = np.logical_and(~pred_fg, gt_fg).sum()

    pred_sum = pred_fg.sum()
    gt_sum = gt_fg.sum()
    union = np.logical_or(pred_fg, gt_fg).sum()

    dice = (2.0 * tp) / max(float(pred_sum + gt_sum), 1.0)
    iou = tp / max(float(union), 1.0)
    precision = tp / max(float(pred_sum), 1.0)
    recall = tp / max(float(gt_sum), 1.0)

    false_fg_ratio = fp / max(float(pred_sum), 1.0)
    unknown_on_fg_ratio = np.logical_and(unknown, gt_fg).sum() / max(float(gt_sum), 1.0)
    unknown_on_bg_ratio = np.logical_and(unknown, ~gt_fg).sum() / max(float((~gt_fg).sum()), 1.0)
    unknown_ratio = unknown.mean()

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "false_fg_ratio": float(false_fg_ratio),
        "unknown_on_fg_ratio": float(unknown_on_fg_ratio),
        "unknown_on_bg_ratio": float(unknown_on_bg_ratio),
        "unknown_ratio": float(unknown_ratio),
        "num_connected_components": float(count_components(pred_fg)),
    }


def mean_metrics(items):
    keys = items[0].keys()
    return {k: float(np.mean([x[k] for x in items])) for k in keys}


def prepare_temp_full_as_box_split(fold_root, source_method, tmp_method, full_records):
    meta_dir = os.path.join(fold_root, "meta")
    source_split_path = os.path.join(meta_dir, f"full_box_split_{source_method}.json")
    source_split = load_json(source_split_path)

    tmp_records = []
    for r in full_records:
        rr = dict(r)
        rr["original_label_mode"] = rr.get("label_mode", "")
        rr["label_mode"] = "box"
        tmp_records.append(rr)

    tmp_split = replace_records(source_split, tmp_records)
    save_json(tmp_split, os.path.join(meta_dir, f"full_box_split_{tmp_method}.json"))

    copy_if_exists(
        os.path.join(meta_dir, f"support_template_{source_method}.npz"),
        os.path.join(meta_dir, f"support_template_{tmp_method}.npz"),
    )
    copy_if_exists(
        os.path.join(meta_dir, f"support_template_stats_{source_method}.json"),
        os.path.join(meta_dir, f"support_template_stats_{tmp_method}.json"),
    )


def prepare_target_meta(fold_root, source_method, target_method):
    meta_dir = os.path.join(fold_root, "meta")
    for stem in [
        "full_box_split",
        "full_box_split_summary",
        "support_template_stats",
    ]:
        copy_if_exists(
            os.path.join(meta_dir, f"{stem}_{source_method}.json"),
            os.path.join(meta_dir, f"{stem}_{target_method}.json"),
        )

    copy_if_exists(
        os.path.join(meta_dir, f"support_template_{source_method}.npz"),
        os.path.join(meta_dir, f"support_template_{target_method}.npz"),
    )


def run_generate_for_full_calib(args, dataset, tmp_method):
    script = Path(__file__).resolve().with_name("04_generate_pseudo_sac.py")
    sac_ckpt = os.path.join(args.sac_checkpoint_root, dataset, args.fold, "medsam_sac_ema.pth")

    if not os.path.exists(sac_ckpt):
        raise FileNotFoundError(f"SAC checkpoint not found: {sac_ckpt}")

    cmd = [
        sys.executable,
        str(script),
        "--processed_root", args.processed_root,
        "--base_checkpoint", args.base_checkpoint,
        "--sac_checkpoint", sac_ckpt,
        "--datasets", dataset,
        "--fold", args.fold,
        "--method", tmp_method,
        "--device", args.device,
        "--max_samples", "0",
        "--fg_q_threshold", str(args.fg_q_threshold),
        "--q_threshold", str(args.q_threshold),
        "--log_every", str(args.log_every),
        "--overwrite",
    ]

    print("\n[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def evaluate_clean_candidates(fold_root, tmp_method, full_records, clean_ratios):
    manifest_index = build_manifest_index(fold_root)
    pred_dir = os.path.join(fold_root, "pseudo_teacher", f"tri_train_{tmp_method}")

    candidates = []

    for ratio in clean_ratios:
        rows = []

        for r in full_records:
            slice_name = get_slice_name(r)
            pred_path = os.path.join(pred_dir, slice_name)
            if not os.path.exists(pred_path):
                continue

            item = manifest_index[slice_name]
            gt_path = resolve_path(fold_root, item["teacher_gt"])

            tri = np.load(pred_path).astype(np.uint8)
            gt = load_gt(gt_path)

            tri_clean, _ = clean_tri_by_components(tri, ratio)

            if tri_clean.shape != gt.shape:
                raise ValueError(f"shape mismatch: {slice_name}, pred={tri_clean.shape}, gt={gt.shape}")

            rows.append(eval_one_pred(tri_clean, gt))

        if not rows:
            raise RuntimeError(f"No valid rows for clean_ratio={ratio}")

        m = mean_metrics(rows)
        m["clean_ratio"] = float(ratio)
        m["num_eval_full"] = int(len(rows))
        candidates.append(m)

    return candidates


def select_clean_ratio(candidates, dice_tol, recall_tol, min_false_fg_gain):
    # ratio=0 is default reference.
    ref = None
    for c in candidates:
        if abs(c["clean_ratio"]) < 1e-12:
            ref = c
            break
    if ref is None:
        raise RuntimeError("clean_ratio=0 must be included in clean_ratio_grid.")

    valid = []
    for c in candidates:
        if c["dice"] < ref["dice"] - dice_tol:
            continue
        if c["recall"] < ref["recall"] - recall_tol:
            continue
        valid.append(c)

    improved = [
        c for c in valid
        if c["false_fg_ratio"] <= ref["false_fg_ratio"] - min_false_fg_gain
    ]

    if improved:
        pool = improved
    else:
        pool = [ref]

    # Among valid and meaningfully improved candidates:
    # 1. lower false_fg
    # 2. fewer components
    # 3. higher dice
    # 4. higher recall
    selected = sorted(
        pool,
        key=lambda x: (
            x["false_fg_ratio"],
            x["num_connected_components"],
            -x["dice"],
            -x["recall"],
        )
    )[0]

    return selected, ref, valid, improved


def apply_clean_to_source_pseudo(fold_root, source_method, target_method, selected_ratio, split_records):
    teacher_src = os.path.join(fold_root, "pseudo_teacher", f"tri_train_{source_method}")
    student_src = os.path.join(fold_root, "pseudo_student", f"tri_train_{source_method}")

    teacher_dst = os.path.join(fold_root, "pseudo_teacher", f"tri_train_{target_method}")
    student_dst = os.path.join(fold_root, "pseudo_student", f"tri_train_{target_method}")
    ensure_dir(teacher_dst)
    ensure_dir(student_dst)

    stats = []

    for r in split_records:
        slice_name = get_slice_name(r)
        label_mode = r.get("label_mode", "")

        for space, src_dir, dst_dir in [
            ("teacher", teacher_src, teacher_dst),
            ("student", student_src, student_dst),
        ]:
            src_path = os.path.join(src_dir, slice_name)
            dst_path = os.path.join(dst_dir, slice_name)

            if not os.path.exists(src_path):
                raise FileNotFoundError(f"missing source pseudo: {src_path}")

            tri = np.load(src_path).astype(np.uint8)

            # full subset remains GT / unchanged.
            if label_mode == "full":
                tri_out = tri.copy()
                clean_info = {
                    "removed_fg_pixels": 0,
                    "original_fg_pixels": int(((tri > 0) & (tri != 255)).sum()),
                    "kept_fg_pixels": int(((tri > 0) & (tri != 255)).sum()),
                }
            else:
                tri_out, clean_info = clean_tri_by_components(tri, selected_ratio)

            np.save(dst_path, tri_out)

            stats.append({
                "slice_name": slice_name,
                "label_mode": label_mode,
                "space": space,
                "clean_ratio": selected_ratio,
                **clean_info,
                "num_components_after": count_components((tri_out > 0) & (tri_out != 255)),
            })

    return stats


def write_csv(rows, path):
    ensure_dir(os.path.dirname(path))
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def cleanup_tmp(fold_root, tmp_method):
    meta_dir = os.path.join(fold_root, "meta")
    paths = [
        os.path.join(meta_dir, f"full_box_split_{tmp_method}.json"),
        os.path.join(meta_dir, f"support_template_{tmp_method}.npz"),
        os.path.join(meta_dir, f"support_template_stats_{tmp_method}.json"),
        os.path.join(meta_dir, f"pseudo_quality_stats_{tmp_method}.csv"),
        os.path.join(fold_root, "pseudo_teacher", f"tri_train_{tmp_method}"),
        os.path.join(fold_root, "pseudo_student", f"tri_train_{tmp_method}"),
    ]

    for p in paths:
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        elif os.path.exists(p):
            os.remove(p)


def process_dataset(args, dataset):
    fold_root = os.path.join(args.processed_root, dataset, args.fold)
    meta_dir = os.path.join(fold_root, "meta")

    source_split_path = os.path.join(meta_dir, f"full_box_split_{args.source_method}.json")
    source_split = load_json(source_split_path)
    split_records = get_records(source_split)
    full_records = [r for r in split_records if r.get("label_mode") == "full"]

    if not full_records:
        raise RuntimeError(f"No full records for {dataset}")

    tmp_method = f"{args.target_method}_tmp_full_box_default"

    print("\n==============================")
    print(f"[CLEAN CALIBRATE] {dataset}")
    print(f"full records = {len(full_records)}")
    print("==============================")

    prepare_temp_full_as_box_split(
        fold_root=fold_root,
        source_method=args.source_method,
        tmp_method=tmp_method,
        full_records=full_records,
    )

    run_generate_for_full_calib(args, dataset, tmp_method)

    clean_ratios = parse_float_list(args.clean_ratio_grid)
    candidates = evaluate_clean_candidates(
        fold_root=fold_root,
        tmp_method=tmp_method,
        full_records=full_records,
        clean_ratios=clean_ratios,
    )

    selected, ref, valid, improved = select_clean_ratio(
        candidates=candidates,
        dice_tol=args.dice_tolerance,
        recall_tol=args.recall_tolerance,
        min_false_fg_gain=args.min_false_fg_gain,
    )

    print("\n[CANDIDATES]")
    for c in sorted(candidates, key=lambda x: x["clean_ratio"]):
        print(
            f"ratio={c['clean_ratio']:.3f} "
            f"dice={c['dice']:.6f} iou={c['iou']:.6f} "
            f"prec={c['precision']:.6f} rec={c['recall']:.6f} "
            f"false_fg={c['false_fg_ratio']:.6f} "
            f"unk_fg={c['unknown_on_fg_ratio']:.6f} "
            f"cc={c['num_connected_components']:.3f}"
        )

    prepare_target_meta(
        fold_root=fold_root,
        source_method=args.source_method,
        target_method=args.target_method,
    )

    apply_stats = apply_clean_to_source_pseudo(
        fold_root=fold_root,
        source_method=args.source_method,
        target_method=args.target_method,
        selected_ratio=selected["clean_ratio"],
        split_records=split_records,
    )

    calib = {
        "dataset": dataset,
        "fold": args.fold,
        "source_method": args.source_method,
        "target_method": args.target_method,
        "test_used": False,
        "calibration_subset": "train_full_subset_with_box_prompt",
        "default_thresholds": {
            "fg_q_threshold": args.fg_q_threshold,
            "q_threshold": args.q_threshold,
        },
        "clean_ratio_grid": clean_ratios,
        "selection_rule": {
            "description": (
                "First keep candidates with Dice >= default Dice - dice_tolerance "
                "and Recall >= default Recall - recall_tolerance. "
                "Among candidates that reduce false_fg_ratio by at least min_false_fg_gain, "
                "select the one with the lowest false_fg_ratio, then fewer components, then higher Dice."
            ),
            "dice_tolerance": args.dice_tolerance,
            "recall_tolerance": args.recall_tolerance,
            "min_false_fg_gain": args.min_false_fg_gain,
        },
        "reference_ratio0": ref,
        "selected": selected,
        "num_valid_candidates": len(valid),
        "num_improved_candidates": len(improved),
        "candidates": sorted(candidates, key=lambda x: x["clean_ratio"]),
    }

    out_json = os.path.join(meta_dir, f"clean_calibration_{args.target_method}.json")
    save_json(calib, out_json)

    out_csv = os.path.join(meta_dir, f"clean_apply_stats_{args.target_method}.csv")
    write_csv(apply_stats, out_csv)

    print("\n[SELECTED]")
    print(json.dumps(selected, indent=2, ensure_ascii=False))
    print("[OK] saved calibration:", out_json)
    print("[OK] saved apply stats:", out_csv)
    print(f"[OK] target pseudo: pseudo_student/tri_train_{args.target_method}")

    if args.cleanup_tmp:
        cleanup_tmp(fold_root, tmp_method)
        print("[OK] cleaned temporary full-box pseudo")

    return calib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", type=str, required=True)
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--sac_checkpoint_root", type=str, required=True)
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--fold", type=str, default="fold_0")
    parser.add_argument("--source_method", type=str, default="idea1_sac_medsam_final")
    parser.add_argument("--target_method", type=str, default="idea1_sac_medsam_final_calib_clean")

    parser.add_argument("--fg_q_threshold", type=float, default=0.45)
    parser.add_argument("--q_threshold", type=float, default=0.58)
    parser.add_argument("--clean_ratio_grid", type=str, default="0,0.01,0.03,0.05,0.10")

    parser.add_argument("--dice_tolerance", type=float, default=0.01)
    parser.add_argument("--recall_tolerance", type=float, default=0.03)
    parser.add_argument("--min_false_fg_gain", type=float, default=0.005)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--cleanup_tmp", action="store_true")
    args = parser.parse_args()

    datasets = parse_dataset_list(args.datasets)
    all_results = []

    for ds in datasets:
        all_results.append(process_dataset(args, ds))

    summary = {
        "target_method": args.target_method,
        "source_method": args.source_method,
        "test_used": False,
        "datasets": [
            {
                "dataset": r["dataset"],
                "selected": r["selected"],
                "reference_ratio0": r["reference_ratio0"],
                "num_valid_candidates": r["num_valid_candidates"],
                "num_improved_candidates": r["num_improved_candidates"],
            }
            for r in all_results
        ],
    }

    out_path = os.path.join(
        args.processed_root,
        f"summary_clean_calibration_{args.target_method}.json",
    )
    save_json(summary, out_path)
    print("\n[OK] saved summary:", out_path)


if __name__ == "__main__":
    main()
