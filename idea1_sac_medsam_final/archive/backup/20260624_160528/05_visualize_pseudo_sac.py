import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_uint8_rgb(img):
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=-1)
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    img = img.astype(np.float32)
    if img.max() <= 1.5:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def load_mask(path):
    m = np.load(path)
    if m.ndim == 3:
        if m.shape[-1] == 1:
            m = m[..., 0]
        elif m.shape[0] == 1:
            m = m[0]
    return m.astype(np.uint8)


def overlay_gt(img, gt):
    out = img.copy()
    color = np.zeros_like(out)
    color[:, :, 1] = (gt > 0).astype(np.uint8) * 255
    return cv2.addWeighted(out, 0.72, color, 0.28, 0)


def overlay_tri(img, tri):
    out = img.copy()
    color = np.zeros_like(out)

    fg = (tri > 0) & (tri != 255)
    unk = tri == 255

    # fg red, unknown yellow
    color[:, :, 2] = fg.astype(np.uint8) * 255
    color[:, :, 1] = unk.astype(np.uint8) * 220
    color[:, :, 2] = np.maximum(color[:, :, 2], unk.astype(np.uint8) * 220)

    return cv2.addWeighted(out, 0.72, color, 0.28, 0)


def draw_instances(img, prompt_meta):
    out = img.copy()
    for ins in prompt_meta.get("instances", []):
        box = ins.get("bbox_teacher", ins.get("bbox"))
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        lid = int(ins.get("label_id", 1))
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            out, f"L{lid}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            out, f"L{lid}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def put_title(img, title):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        out, title,
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


def make_canvas(img, gt, tri, prompt_meta, row):
    p0 = draw_instances(img, prompt_meta)
    p1 = overlay_gt(img, gt)
    p2 = overlay_tri(img, tri)

    stat = (
        f"{row['slice_name']} | {row['label_mode']} | "
        f"fg={row['box_fg_ratio']:.3f} unk={row['box_unknown_ratio']:.3f} | "
        f"cc={int(row['num_connected_components'])} "
        f"largest={row['largest_component_ratio']:.3f}"
    )

    p0 = put_title(p0, "Image + box")
    p1 = put_title(p1, "GT overlay")
    p2 = put_title(p2, "SAC tri pseudo: red=FG, yellow=255")

    canvas = np.hstack([p0, p1, p2])
    canvas = put_title(canvas, stat)
    return canvas


def select_rows(df, n):
    box = df[df["label_mode"] == "box"].copy()
    full = df[df["label_mode"] == "full"].copy()

    groups = []

    groups.append(("worst_fragmentation", box.sort_values("largest_component_ratio", ascending=True).head(n)))
    groups.append(("high_unknown", box.sort_values("box_unknown_ratio", ascending=False).head(n)))

    normal = box[
        (box["box_fg_ratio"] >= 0.60)
        & (box["box_fg_ratio"] <= 0.85)
        & (box["box_unknown_ratio"] <= 0.35)
        & (box["largest_component_ratio"] >= 0.95)
    ].copy()
    groups.append(("normal", normal.head(n)))

    groups.append(("full_subset", full.head(n)))

    return groups


def process_dataset(args, dataset):
    fold_root = os.path.join(args.processed_root, dataset, args.fold)
    meta_dir = os.path.join(fold_root, "meta")

    manifest = load_json(os.path.join(meta_dir, "manifest.json"))
    prompts = load_json(os.path.join(fold_root, "prompts", "prompts_train.json"))
    csv_path = os.path.join(meta_dir, f"pseudo_quality_stats_{args.method}.csv")

    df = pd.read_csv(csv_path)

    manifest_by_slice = {x["slice_name"]: x for x in manifest}

    tri_dir = os.path.join(fold_root, "pseudo_teacher", f"tri_train_{args.method}")
    out_root = os.path.join(fold_root, "pseudo_teacher", f"vis_train_{args.method}")
    ensure_dir(out_root)

    selected = select_rows(df, args.n_per_group)

    saved = []
    for group_name, rows in selected:
        group_dir = os.path.join(out_root, group_name)
        ensure_dir(group_dir)

        for _, row in rows.iterrows():
            slice_name = row["slice_name"]
            item = manifest_by_slice[slice_name]

            img_path = os.path.join(fold_root, item["teacher_img"])
            gt_path = os.path.join(fold_root, item["teacher_gt"])
            tri_path = os.path.join(tri_dir, slice_name)

            img = to_uint8_rgb(np.load(img_path))
            gt = load_mask(gt_path)
            tri = load_mask(tri_path)
            prompt_meta = prompts[slice_name]

            canvas = make_canvas(img, gt, tri, prompt_meta, row)
            out_path = os.path.join(group_dir, slice_name.replace(".npy", ".jpg"))
            cv2.imwrite(out_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            saved.append(out_path)

    print(f"[OK] saved {len(saved)} visualizations")
    print(f"     out = {out_root}")
    for p in saved[:20]:
        print("     ", p)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--method", default="idea1_sac_medsam_final")
    parser.add_argument("--n_per_group", type=int, default=8)
    args = parser.parse_args()

    for ds in [x.strip() for x in args.datasets.split(",") if x.strip()]:
        process_dataset(args, ds)


if __name__ == "__main__":
    main()
