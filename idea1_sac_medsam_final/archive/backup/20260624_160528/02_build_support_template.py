import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import os
import random
from collections import defaultdict

import cv2
import numpy as np
import torch


METHOD_DEFAULT = "idea1_sac_medsam_final"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_slice_name(item):
    if "slice_name" in item:
        return item["slice_name"]
    for key in ["teacher_img", "student_img", "teacher_gt", "student_gt"]:
        if key in item:
            return os.path.basename(item[key])
    raise KeyError(f"Cannot infer slice_name from keys={list(item.keys())}")


def resolve_path(fold_root, item, keys, fallback_dirs):
    for key in keys:
        if key in item and item[key]:
            p = item[key]
            if os.path.isabs(p) and os.path.exists(p):
                return p
            p2 = os.path.join(fold_root, p)
            if os.path.exists(p2):
                return p2

    slice_name = get_slice_name(item)
    for d in fallback_dirs:
        p = os.path.join(fold_root, d, slice_name)
        if os.path.exists(p):
            return p
        p_train = os.path.join(fold_root, d, "train", slice_name)
        if os.path.exists(p_train):
            return p_train
        p_test = os.path.join(fold_root, d, "test", slice_name)
        if os.path.exists(p_test):
            return p_test

    raise FileNotFoundError(
        f"Cannot resolve path for slice={slice_name}, keys={keys}, fallback_dirs={fallback_dirs}"
    )


def load_image_npy(path):
    arr = np.load(path)

    # CHW -> HWC
    if arr.ndim == 3 and arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected image shape {arr.shape} from {path}")

    arr = arr.astype(np.float32)

    # MedSAM 常见输入是 0-1，这里做 auto normalize
    if arr.max() > 2.0:
        arr = arr / 255.0

    arr = np.clip(arr, 0.0, 1.0)
    return arr


def load_mask_npy(path):
    m = np.load(path)
    if m.ndim == 3:
        if m.shape[-1] == 1:
            m = m[..., 0]
        elif m.shape[0] == 1:
            m = m[0]
        else:
            raise ValueError(f"Unexpected mask shape {m.shape} from {path}")
    return m.astype(np.int64)


def build_medsam(checkpoint, device):
    try:
        from segment_anything import sam_model_registry
    except Exception as e:
        raise ImportError(
            "Cannot import segment_anything. Please run in medsam310 env."
        ) from e

    model = sam_model_registry["vit_b"](checkpoint=checkpoint)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def extract_encoder_feature(model, img_hwc, device):
    """
    img_hwc: H,W,3 float32 in [0,1]
    return: feat_h, feat_w, C float32, L2-normalized
    """
    x = torch.from_numpy(img_hwc).permute(2, 0, 1).unsqueeze(0).float().to(device)

    feat = model.image_encoder(x)
    # usually [1, C, 64, 64]
    if isinstance(feat, (list, tuple)):
        feat = feat[0]
    feat = feat.detach().float().cpu().numpy()[0]  # C,H,W
    feat = np.transpose(feat, (1, 2, 0))  # H,W,C

    norm = np.linalg.norm(feat, axis=-1, keepdims=True) + 1e-6
    feat = feat / norm
    return feat.astype(np.float32)


def resize_mask_nearest(mask, size_hw):
    h, w = size_hw
    return cv2.resize(mask.astype(np.int32), (w, h), interpolation=cv2.INTER_NEAREST).astype(np.int64)


def resize_float(mask, size_hw):
    h, w = size_hw
    return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def bbox_from_binary(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def sample_rows(arr, max_n, rng):
    if arr.shape[0] <= max_n:
        return arr
    idx = rng.choice(arr.shape[0], size=max_n, replace=False)
    return arr[idx]


def simple_kmeans_l2norm(x, k, seed, num_iter=25):
    """
    x: N,C already L2-normalized
    return: k,C L2-normalized centers
    """
    if x.shape[0] == 0:
        raise ValueError("Empty feature array for kmeans")

    rng = np.random.default_rng(seed)
    n = x.shape[0]
    k = min(k, n)

    init_idx = rng.choice(n, size=k, replace=False)
    centers = x[init_idx].copy()

    for _ in range(num_iter):
        # cosine similarity because vectors are normalized
        sim = x @ centers.T
        labels = sim.argmax(axis=1)

        new_centers = []
        for j in range(k):
            part = x[labels == j]
            if part.shape[0] == 0:
                new_centers.append(x[rng.integers(0, n)])
            else:
                c = part.mean(axis=0)
                c = c / (np.linalg.norm(c) + 1e-6)
                new_centers.append(c.astype(np.float32))
        centers = np.stack(new_centers, axis=0)

    return centers.astype(np.float32)


def process_dataset(
    processed_root,
    checkpoint,
    dataset,
    fold,
    method,
    device,
    seed,
    k_fg,
    k_bg,
    max_fg_per_image,
    max_bg_per_image,
    max_total_features,
    overwrite,
):
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    fold_root = os.path.join(processed_root, dataset, fold)
    meta_dir = os.path.join(fold_root, "meta")

    manifest_path = os.path.join(meta_dir, "manifest.json")
    split_path = os.path.join(meta_dir, f"full_box_split_{method}.json")

    out_npz = os.path.join(meta_dir, f"support_template_{method}.npz")
    out_stats = os.path.join(meta_dir, f"support_template_stats_{method}.json")

    if os.path.exists(out_npz) and os.path.exists(out_stats) and not overwrite:
        print(f"[SKIP] exists: {out_npz}")
        return

    manifest = load_json(manifest_path)
    split_records = load_json(split_path)

    manifest_by_slice = {get_slice_name(x): x for x in manifest}

    full_records = [r for r in split_records if r.get("label_mode") == "full"]
    if len(full_records) == 0:
        raise RuntimeError(f"No full records found: {split_path}")

    # ------------------------------------------------------------------
    # Class ids.
    # For 2D binary datasets, class_ids is usually [1].
    # For 3D multi-class datasets such as ACDC/Synapse/BTCV/Prostate158,
    # split_records may be a plain list and each record may not contain
    # "class_ids". Therefore, reading only from full_records would wrongly
    # fall back to [1]. We first read label_meta.json, which records all
    # foreground labels in the processed dataset.
    # ------------------------------------------------------------------
    label_meta_path = os.path.join(meta_dir, "label_meta.json")
    summary_path = os.path.join(meta_dir, f"full_box_split_summary_{method}.json")

    class_ids = []

    if os.path.exists(label_meta_path):
        label_meta = load_json(label_meta_path)
        class_ids = sorted([
            int(c) for c in label_meta.get("unique_labels", [])
            if int(c) > 0
        ])

    if not class_ids and os.path.exists(summary_path):
        summary = load_json(summary_path)
        class_ids = sorted([
            int(c) for c in summary.get("class_ids", [])
            if int(c) > 0
        ])

    if not class_ids:
        class_ids = sorted({
            int(c)
            for r in full_records
            for c in r.get("class_ids", [1])
            if int(c) > 0
        })

    if not class_ids:
        class_ids = [1]

    print(f"[CLASS_IDS] {dataset}/{fold}: {class_ids}")

    model = build_medsam(checkpoint, device)

    fg_features = {c: [] for c in class_ids}
    bg_features = {c: [] for c in class_ids}
    shape_masks = {c: [] for c in class_ids}

    per_image_stats = []
    num_missing = 0

    for idx, rec in enumerate(full_records):
        slice_name = rec["slice_name"]
        item = manifest_by_slice.get(slice_name)
        if item is None:
            num_missing += 1
            print(f"[WARN] missing in manifest: {slice_name}")
            continue

        img_path = resolve_path(
            fold_root,
            item,
            keys=["teacher_img", "teacher_image", "img_teacher"],
            fallback_dirs=["teacher_npy/imgs"],
        )
        gt_path = resolve_path(
            fold_root,
            item,
            keys=["teacher_gt", "teacher_mask", "gt_teacher"],
            fallback_dirs=["teacher_npy/gts"],
        )

        img = load_image_npy(img_path)
        gt = load_mask_npy(gt_path)

        if img.shape[:2] != gt.shape[:2]:
            raise ValueError(f"Image/GT shape mismatch: {slice_name}, img={img.shape}, gt={gt.shape}")

        feat = extract_encoder_feature(model, img, device)
        fh, fw = feat.shape[:2]
        gt_feat = resize_mask_nearest(gt, (fh, fw))

        stat = {
            "slice_name": slice_name,
            "img_path": img_path,
            "gt_path": gt_path,
            "classes": {},
        }

        for c in class_ids:
            mask_c = (gt == c).astype(np.uint8)
            bbox = bbox_from_binary(mask_c)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                crop = mask_c[y1:y2, x1:x2]
                if crop.size > 0:
                    crop64 = resize_float(crop, (64, 64))
                    crop64 = np.clip(crop64, 0.0, 1.0)
                    shape_masks[c].append(crop64.astype(np.float32))

            fg_idx = np.where(gt_feat == c)
            bg_idx = np.where(gt_feat == 0)

            fg_arr = feat[fg_idx]
            bg_arr = feat[bg_idx]

            if fg_arr.shape[0] > 0:
                fg_arr = sample_rows(fg_arr, max_fg_per_image, rng)
                fg_features[c].append(fg_arr)

            if bg_arr.shape[0] > 0:
                bg_arr = sample_rows(bg_arr, max_bg_per_image, rng)
                bg_features[c].append(bg_arr)

            stat["classes"][str(c)] = {
                "gt_pixels_teacher": int(mask_c.sum()),
                "fg_feat_pixels": int(len(fg_idx[0])),
                "bg_feat_pixels": int(len(bg_idx[0])),
                "has_shape": bbox is not None,
            }

        per_image_stats.append(stat)

        if (idx + 1) % 20 == 0:
            print(f"[{dataset}] processed full templates {idx+1}/{len(full_records)}")

    npz_data = {}
    stats = {
        "dataset": dataset,
        "fold": fold,
        "method": method,
        "seed": seed,
        "checkpoint": checkpoint,
        "num_full_records": len(full_records),
        "num_missing_manifest": num_missing,
        "class_ids": class_ids,
        "k_fg": k_fg,
        "k_bg": k_bg,
        "max_fg_per_image": max_fg_per_image,
        "max_bg_per_image": max_bg_per_image,
        "max_total_features": max_total_features,
        "classes": {},
        "per_image_stats": per_image_stats[:50],  # 避免 json 过大，只留前 50 个样本明细
    }

    for c in class_ids:
        if fg_features[c]:
            fg = np.concatenate(fg_features[c], axis=0).astype(np.float32)
        else:
            fg = np.zeros((0, 256), dtype=np.float32)

        if bg_features[c]:
            bg = np.concatenate(bg_features[c], axis=0).astype(np.float32)
        else:
            bg = np.zeros((0, 256), dtype=np.float32)

        fg = sample_rows(fg, max_total_features, rng) if fg.shape[0] > 0 else fg
        bg = sample_rows(bg, max_total_features, rng) if bg.shape[0] > 0 else bg

        if fg.shape[0] == 0:
            raise RuntimeError(f"No foreground features collected for class {c}")
        if bg.shape[0] == 0:
            raise RuntimeError(f"No background features collected for class {c}")

        proto_fg = simple_kmeans_l2norm(fg, k_fg, seed=seed + c * 11)
        proto_bg = simple_kmeans_l2norm(bg, k_bg, seed=seed + c * 17)

        if shape_masks[c]:
            A = np.stack(shape_masks[c], axis=0).mean(axis=0).astype(np.float32)
        else:
            A = np.zeros((64, 64), dtype=np.float32)

        R = (1.0 - 4.0 * A * (1.0 - A)).astype(np.float32)
        R = np.clip(R, 0.0, 1.0)

        npz_data[f"proto_fg_c{c}"] = proto_fg
        npz_data[f"proto_bg_c{c}"] = proto_bg
        npz_data[f"shape_A_c{c}"] = A
        npz_data[f"shape_R_c{c}"] = R

        stats["classes"][str(c)] = {
            "num_fg_features": int(fg.shape[0]),
            "num_bg_features": int(bg.shape[0]),
            "num_shape_masks": int(len(shape_masks[c])),
            "proto_fg_shape": list(proto_fg.shape),
            "proto_bg_shape": list(proto_bg.shape),
            "shape_A_mean": float(A.mean()),
            "shape_A_max": float(A.max()),
            "shape_R_mean": float(R.mean()),
            "shape_R_min": float(R.min()),
            "shape_R_max": float(R.max()),
        }

    npz_data["class_ids"] = np.array(class_ids, dtype=np.int64)
    np.savez_compressed(out_npz, **npz_data)
    save_json(stats, out_stats)

    print(f"[OK] support template built: {dataset}/{fold}")
    print(f"     out_npz   = {out_npz}")
    print(f"     out_stats = {out_stats}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--k_fg", type=int, default=3)
    parser.add_argument("--k_bg", type=int, default=5)
    parser.add_argument("--max_fg_per_image", type=int, default=512)
    parser.add_argument("--max_bg_per_image", type=int, default=512)
    parser.add_argument("--max_total_features", type=int, default=200000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda unavailable, fallback to cpu")
        args.device = "cpu"

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for ds in datasets:
        process_dataset(
            processed_root=args.processed_root,
            checkpoint=args.checkpoint,
            dataset=ds,
            fold=args.fold,
            method=args.method,
            device=args.device,
            seed=args.seed,
            k_fg=args.k_fg,
            k_bg=args.k_bg,
            max_fg_per_image=args.max_fg_per_image,
            max_bg_per_image=args.max_bg_per_image,
            max_total_features=args.max_total_features,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
