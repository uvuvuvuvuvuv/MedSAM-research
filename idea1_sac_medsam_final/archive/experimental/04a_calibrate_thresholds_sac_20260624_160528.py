import os
import sys
import json
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
    raise KeyError(f"Cannot find slice_name-like key in record keys={list(record.keys())}")


def copy_if_exists(src, dst):
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def float_token(x):
    return f"{x:.3f}".replace(".", "p")


def make_tmp_method(target_method, fg_q, q):
    return f"{target_method}_tmp_fg{float_token(fg_q)}_q{float_token(q)}"


def build_manifest_index(fold_root):
    manifest_path = os.path.join(fold_root, "meta", "manifest.json")
    manifest = load_json(manifest_path)
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


def count_components(mask):
    mask = mask.astype(np.uint8)
    if mask.max() == 0:
        return 0
    num, _ = cv2.connectedComponents(mask)
    return max(0, int(num) - 1)


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
    under_fg_ratio = fn / max(float(gt_sum), 1.0)
    unknown_on_fg_ratio = np.logical_and(unknown, gt_fg).sum() / max(float(gt_sum), 1.0)
    unknown_on_bg_ratio = np.logical_and(unknown, ~gt_fg).sum() / max(float((~gt_fg).sum()), 1.0)
    unknown_ratio = unknown.mean()

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "false_fg_ratio": float(false_fg_ratio),
        "under_fg_ratio": float(under_fg_ratio),
        "unknown_on_fg_ratio": float(unknown_on_fg_ratio),
        "unknown_on_bg_ratio": float(unknown_on_bg_ratio),
        "unknown_ratio": float(unknown_ratio),
        "num_connected_components": float(count_components(pred_fg)),
    }


def mean_metrics(items):
    keys = items[0].keys()
    return {k: float(np.mean([x[k] for x in items])) for k in keys}


def prepare_temp_method_files(fold_root, source_method, tmp_method, full_records):
    meta_dir = os.path.join(fold_root, "meta")

    source_split_path = os.path.join(meta_dir, f"full_box_split_{source_method}.json")
    source_split = load_json(source_split_path)

    # 关键：只保留 D_full，但把 label_mode 改成 box。
    # 这样调用 04_generate_pseudo_sac.py 时不会直接复制 GT，而是使用 box prompt 生成 pseudo。
    tmp_records = []
    for r in full_records:
        rr = dict(r)
        rr["original_label_mode"] = rr.get("label_mode", "")
        rr["label_mode"] = "box"
        tmp_records.append(rr)

    tmp_split = replace_records(source_split, tmp_records)
    save_json(tmp_split, os.path.join(meta_dir, f"full_box_split_{tmp_method}.json"))

    # 04_generate_pseudo_sac.py 会按 method 名找 support template，因此复制一份。
    copy_if_exists(
        os.path.join(meta_dir, f"support_template_{source_method}.npz"),
        os.path.join(meta_dir, f"support_template_{tmp_method}.npz"),
    )
    copy_if_exists(
        os.path.join(meta_dir, f"support_template_stats_{source_method}.json"),
        os.path.join(meta_dir, f"support_template_stats_{tmp_method}.json"),
    )


def prepare_target_method_files(fold_root, source_method, target_method):
    meta_dir = os.path.join(fold_root, "meta")

    copy_if_exists(
        os.path.join(meta_dir, f"full_box_split_{source_method}.json"),
        os.path.join(meta_dir, f"full_box_split_{target_method}.json"),
    )
    copy_if_exists(
        os.path.join(meta_dir, f"full_box_split_summary_{source_method}.json"),
        os.path.join(meta_dir, f"full_box_split_summary_{target_method}.json"),
    )
    copy_if_exists(
        os.path.join(meta_dir, f"support_template_{source_method}.npz"),
        os.path.join(meta_dir, f"support_template_{target_method}.npz"),
    )
    copy_if_exists(
        os.path.join(meta_dir, f"support_template_stats_{source_method}.json"),
        os.path.join(meta_dir, f"support_template_stats_{target_method}.json"),
    )


def run_generate_script(args, dataset, tmp_method, fg_q, q):
    script = Path(__file__).resolve().with_name("04_generate_pseudo_sac.py")

    sac_ckpt = os.path.join(
        args.sac_checkpoint_root,
        dataset,
        args.fold,
        "medsam_sac_ema.pth",
    )
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
        "--fg_q_threshold", str(fg_q),
        "--q_threshold", str(q),
        "--log_every", str(args.log_every),
        "--overwrite",
    ]

    print("\n[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def evaluate_temp_method(fold_root, tmp_method, full_records):
    manifest_index = build_manifest_index(fold_root)
    pred_dir = os.path.join(fold_root, "pseudo_teacher", f"tri_train_{tmp_method}")

    metrics = []
    missing = []

    for r in full_records:
        slice_name = get_slice_name(r)
        pred_path = os.path.join(pred_dir, slice_name)
        if not os.path.exists(pred_path):
            missing.append(slice_name)
            continue

        item = manifest_index[slice_name]
        gt_path = resolve_path(fold_root, item["teacher_gt"])

        tri = np.load(pred_path).astype(np.uint8)
        gt = load_gt(gt_path)

        if tri.shape != gt.shape:
            raise ValueError(f"Shape mismatch for {slice_name}: pred={tri.shape}, gt={gt.shape}")

        metrics.append(eval_one_pred(tri, gt))

    if not metrics:
        raise RuntimeError(f"No valid prediction found for tmp_method={tmp_method}")

    out = mean_metrics(metrics)
    out["num_eval_full"] = int(len(metrics))
    out["num_missing"] = int(len(missing))
    return out


def cleanup_tmp_outputs(fold_root, tmp_methods):
    meta_dir = os.path.join(fold_root, "meta")

    for m in tmp_methods:
        for p in [
            os.path.join(meta_dir, f"full_box_split_{m}.json"),
            os.path.join(meta_dir, f"support_template_{m}.npz"),
            os.path.join(meta_dir, f"support_template_stats_{m}.json"),
            os.path.join(meta_dir, f"pseudo_quality_stats_{m}.csv"),
            os.path.join(fold_root, "pseudo_teacher", f"tri_train_{m}"),
            os.path.join(fold_root, "pseudo_student", f"tri_train_{m}"),
        ]:
            if os.path.isdir(p):
                shutil.rmtree(p, ignore_errors=True)
            elif os.path.exists(p):
                os.remove(p)


def calibrate_one_dataset(args, dataset):
    fold_root = os.path.join(args.processed_root, dataset, args.fold)
    meta_dir = os.path.join(fold_root, "meta")

    source_split_path = os.path.join(meta_dir, f"full_box_split_{args.source_method}.json")
    source_split = load_json(source_split_path)
    all_records = get_records(source_split)

    full_records = [r for r in all_records if r.get("label_mode") == "full"]
    if len(full_records) == 0:
        raise RuntimeError(f"No full records found for {dataset}")

    print(f"\n==============================")
    print(f"[CALIBRATE] {dataset}")
    print(f"full records = {len(full_records)}")
    print(f"==============================")

    fg_grid = parse_float_list(args.fg_q_grid)
    q_grid = parse_float_list(args.q_grid)

    candidates = []
    tmp_methods = []

    for fg_q in fg_grid:
        for q in q_grid:
            tmp_method = make_tmp_method(args.target_method, fg_q, q)
            tmp_methods.append(tmp_method)

            prepare_temp_method_files(
                fold_root=fold_root,
                source_method=args.source_method,
                tmp_method=tmp_method,
                full_records=full_records,
            )

            run_generate_script(args, dataset, tmp_method, fg_q, q)

            metrics = evaluate_temp_method(fold_root, tmp_method, full_records)

            # 统一选择准则：保证 Dice，同时惩罚 false foreground 和前景 unknown。
            score = (
                metrics["dice"]
                - args.false_fg_weight * metrics["false_fg_ratio"]
                - args.unknown_fg_weight * metrics["unknown_on_fg_ratio"]
            )

            row = {
                "fg_q_threshold": float(fg_q),
                "q_threshold": float(q),
                "score": float(score),
                **metrics,
            }
            candidates.append(row)

            print(
                f"[CAND] {dataset} fg_q={fg_q:.2f} q={q:.2f} "
                f"score={row['score']:.6f} dice={row['dice']:.6f} "
                f"prec={row['precision']:.6f} rec={row['recall']:.6f} "
                f"false_fg={row['false_fg_ratio']:.6f} unk_fg={row['unknown_on_fg_ratio']:.6f}",
                flush=True,
            )

    # score 最大；若接近，则 Dice 高优先；再 false_fg 低优先。
    candidates_sorted = sorted(
        candidates,
        key=lambda x: (
            x["score"],
            x["dice"],
            -x["false_fg_ratio"],
            -x["unknown_on_fg_ratio"],
        ),
        reverse=True,
    )
    selected = candidates_sorted[0]

    prepare_target_method_files(
        fold_root=fold_root,
        source_method=args.source_method,
        target_method=args.target_method,
    )

    calib = {
        "dataset": dataset,
        "fold": args.fold,
        "source_method": args.source_method,
        "target_method": args.target_method,
        "calibration_subset": "train_full_subset",
        "num_full_records": len(full_records),
        "test_used": False,
        "fg_q_grid": fg_grid,
        "q_grid": q_grid,
        "selection_rule": (
            f"score = dice - {args.false_fg_weight} * false_fg_ratio "
            f"- {args.unknown_fg_weight} * unknown_on_fg_ratio"
        ),
        "selected": {
            "fg_q_threshold": selected["fg_q_threshold"],
            "q_threshold": selected["q_threshold"],
            "score": selected["score"],
            "dice": selected["dice"],
            "iou": selected["iou"],
            "precision": selected["precision"],
            "recall": selected["recall"],
            "false_fg_ratio": selected["false_fg_ratio"],
            "unknown_on_fg_ratio": selected["unknown_on_fg_ratio"],
            "unknown_ratio": selected["unknown_ratio"],
            "num_connected_components": selected["num_connected_components"],
        },
        "candidates": candidates_sorted,
    }

    out_path = os.path.join(meta_dir, f"threshold_calibration_{args.target_method}.json")
    save_json(calib, out_path)

    print(f"\n[SELECTED] {dataset}")
    print(json.dumps(calib["selected"], indent=2, ensure_ascii=False))
    print("[OK] saved:", out_path)

    if args.cleanup_tmp:
        cleanup_tmp_outputs(fold_root, tmp_methods)
        print("[OK] cleaned temporary calibration outputs")

    return calib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", type=str, required=True)
    parser.add_argument("--base_checkpoint", type=str, required=True)
    parser.add_argument("--sac_checkpoint_root", type=str, required=True)
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--fold", type=str, default="fold_0")
    parser.add_argument("--source_method", type=str, default="idea1_sac_medsam_final")
    parser.add_argument("--target_method", type=str, default="idea1_sac_medsam_final_calib")
    parser.add_argument("--fg_q_grid", type=str, default="0.45,0.50,0.55")
    parser.add_argument("--q_grid", type=str, default="0.58,0.62,0.66")
    parser.add_argument("--false_fg_weight", type=float, default=0.5)
    parser.add_argument("--unknown_fg_weight", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--cleanup_tmp", action="store_true")
    args = parser.parse_args()

    datasets = parse_dataset_list(args.datasets)
    all_results = []

    for ds in datasets:
        all_results.append(calibrate_one_dataset(args, ds))

    summary = {
        "target_method": args.target_method,
        "source_method": args.source_method,
        "test_used": False,
        "datasets": [
            {
                "dataset": r["dataset"],
                "selected": r["selected"],
            }
            for r in all_results
        ],
    }

    out_path = os.path.join(
        args.processed_root,
        f"summary_threshold_calibration_{args.target_method}.json",
    )
    save_json(summary, out_path)
    print("\n[OK] saved summary:", out_path)


if __name__ == "__main__":
    main()
