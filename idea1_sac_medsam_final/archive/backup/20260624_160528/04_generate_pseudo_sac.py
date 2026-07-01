import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import csv
import json
import os
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F


METHOD_DEFAULT = "idea1_sac_medsam_final"
MODEL_TYPE = "vit_b"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_slice_name(item):
    if "slice_name" in item:
        return item["slice_name"]
    for key in ["teacher_img", "student_img", "teacher_gt", "student_gt"]:
        if key in item:
            return os.path.basename(item[key])
    raise KeyError(f"Cannot infer slice_name from keys={list(item.keys())}")


def resolve_path(fold_root, item, key):
    p = item[key]
    if os.path.isabs(p):
        return p
    return os.path.join(fold_root, p)


def load_image_tensor(path, device):
    img = np.load(path)

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    img = img.astype(np.float32)
    if img.max() > 2:
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return x, img


def load_mask(path):
    m = np.load(path)
    if m.ndim == 3:
        if m.shape[-1] == 1:
            m = m[..., 0]
        elif m.shape[0] == 1:
            m = m[0]
        else:
            raise ValueError(f"Unexpected mask shape: {m.shape}")
    return m


def build_medsam(base_checkpoint, sac_checkpoint, device):
    from segment_anything import sam_model_registry

    model = sam_model_registry[MODEL_TYPE](checkpoint=base_checkpoint)
    payload = torch.load(sac_checkpoint, map_location="cpu")

    if isinstance(payload, dict) and "model" in payload:
        state = payload["model"]
    else:
        state = payload

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load SAC] missing={len(missing)} unexpected={len(unexpected)}")

    model.to(device)
    model.eval()
    return model


def make_box_tensor(box, device):
    box = np.array(box, dtype=np.float32).reshape(1, 4)
    return torch.from_numpy(box).float().to(device)


def make_box_mask_np(box, h, w):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(w - 1, int(np.floor(x1))))
    y1 = max(0, min(h - 1, int(np.floor(y1))))
    x2 = max(0, min(w, int(np.ceil(x2))))
    y2 = max(0, min(h, int(np.ceil(y2))))

    m = np.zeros((h, w), dtype=bool)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = True
    return m


@torch.no_grad()
def forward_medsam(model, image_tensor, box_tensor):
    image_embedding = model.image_encoder(image_tensor)

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=box_tensor,
        masks=None,
    )

    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )

    logits_1024 = F.interpolate(
        low_res_logits,
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    return low_res_logits, logits_1024, image_embedding


def compute_qf_from_embedding(image_embedding, proto_fg, proto_bg, out_size_hw, temperature=0.07):
    feat = image_embedding.detach()
    feat = F.normalize(feat, dim=1)

    b, c, h, w = feat.shape
    feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, c)

    pf = torch.from_numpy(proto_fg).float().to(feat.device)
    pb = torch.from_numpy(proto_bg).float().to(feat.device)
    pf = F.normalize(pf, dim=1)
    pb = F.normalize(pb, dim=1)

    sim_fg = feat_flat @ pf.t()
    sim_bg = feat_flat @ pb.t()

    max_fg = sim_fg.max(dim=1).values
    max_bg = sim_bg.max(dim=1).values

    q = torch.exp(max_fg / temperature) / (
        torch.exp(max_fg / temperature) + torch.exp(max_bg / temperature) + 1e-6
    )
    q = q.reshape(1, 1, h, w)
    q = F.interpolate(q, size=out_size_hw, mode="bilinear", align_corners=False)
    return q.clamp(0, 1)


def build_shape_map_np(shape_A, box, out_h, out_w):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(out_w - 1, int(np.floor(x1))))
    y1 = max(0, min(out_h - 1, int(np.floor(y1))))
    x2 = max(0, min(out_w, int(np.ceil(x2))))
    y2 = max(0, min(out_h, int(np.ceil(y2))))

    out = np.zeros((out_h, out_w), dtype=np.float32)

    if x2 <= x1 or y2 <= y1:
        return out

    a = cv2.resize(shape_A.astype(np.float32), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
    out[y1:y2, x1:x2] = np.clip(a, 0.0, 1.0)
    return out


def restore_teacher_to_native(mask_teacher, geom, native_shape=None):
    if all(k in geom for k in ["new_h", "new_w", "orig_h", "orig_w"]):
        new_h = int(geom["new_h"])
        new_w = int(geom["new_w"])
        orig_h = int(geom["orig_h"])
        orig_w = int(geom["orig_w"])
        cropped = mask_teacher[:new_h, :new_w]
        return cv2.resize(cropped, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    if native_shape is not None:
        return cv2.resize(mask_teacher, (native_shape[1], native_shape[0]), interpolation=cv2.INTER_NEAREST)

    raise KeyError("geometry_meta lacks new_h/new_w/orig_h/orig_w and native_shape is None")


def restore_native_mask_to_student(mask_native, geom):
    """
    Map native-space mask to fixed student canvas using geometry_meta["native_to_student"].
    This follows the frozen baseline contract: resize to new_h/new_w, then paste into
    target_h/target_w canvas using offset_x/offset_y.
    """
    if "native_to_student" not in geom:
        raise KeyError("geometry meta missing native_to_student")

    g = geom["native_to_student"]
    target_h = int(g["target_h"])
    target_w = int(g["target_w"])
    new_h = int(g["new_h"])
    new_w = int(g["new_w"])
    offset_x = int(g.get("offset_x", 0))
    offset_y = int(g.get("offset_y", 0))

    resized = cv2.resize(
        mask_native.astype(np.uint8),
        (new_w, new_h),
        interpolation=cv2.INTER_NEAREST,
    )

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized
    return canvas


def count_components(mask_fg):
    mask_fg = mask_fg.astype(np.uint8)
    if mask_fg.max() == 0:
        return 0, 0.0
    n, lab = cv2.connectedComponents(mask_fg)
    if n <= 1:
        return 0, 0.0
    areas = [(lab == i).sum() for i in range(1, n)]
    total = float(sum(areas))
    largest = float(max(areas)) if areas else 0.0
    return n - 1, largest / max(total, 1.0)


def dice_iou(pred, gt, label_id):
    p = pred == label_id
    g = gt == label_id
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    d = (2 * inter + 1e-6) / (p.sum() + g.sum() + 1e-6)
    i = (inter + 1e-6) / (union + 1e-6)
    return float(d), float(i)


def merge_instance_into_tri(tri, fg_mask, unknown_mask, label_id):
    existing_fg = (tri > 0) & (tri != 255)

    conflict = fg_mask & existing_fg & (tri != label_id)
    tri[conflict] = 255

    write_fg = fg_mask & (~conflict)
    tri[write_fg] = np.uint8(label_id)

    write_unknown = unknown_mask & (tri == 0)
    tri[write_unknown] = 255

    return tri


def generate_one_slice(
    model,
    image_tensor,
    prompt_meta,
    label_mode,
    teacher_gt,
    template,
    device,
    fg_q_threshold,
    q_threshold,
    save_prob_stats,
):
    h, w = teacher_gt.shape
    tri = np.zeros((h, w), dtype=np.uint8)

    # full subset 直接写 GT
    if label_mode == "full":
        tri = teacher_gt.astype(np.uint8).copy()
        return tri, {
            "mean_P_inside_box": 0.0,
            "mean_qF_inside_box": 0.0,
            "mean_A_inside_box": 0.0,
            "weak_activation_ratio_0.30_0.40": 0.0,
            "weak_activation_ratio_0.40_0.50": 0.0,
            "box_fg_ratio": 0.0,
            "box_unknown_ratio": 0.0,
        }

    instances = prompt_meta.get("instances", [])
    if len(instances) == 0:
        return tri, {}

    all_box = np.zeros((h, w), dtype=bool)
    all_p_vals = []
    all_q_vals = []
    all_a_vals = []
    weak_030_040 = []
    weak_040_050 = []

    for ins in instances:
        box = ins.get("bbox_teacher", ins.get("bbox"))
        label_id = int(ins.get("label_id", 1))

        if f"proto_fg_c{label_id}" not in template:
            label_id = 1

        box_tensor = make_box_tensor(box, device)
        low_logits, logits_1024, image_embedding = forward_medsam(model, image_tensor, box_tensor)

        p = torch.sigmoid(logits_1024)
        q = compute_qf_from_embedding(
            image_embedding,
            template[f"proto_fg_c{label_id}"],
            template[f"proto_bg_c{label_id}"],
            out_size_hw=(h, w),
        )

        p_np = p[0, 0].detach().cpu().numpy().astype(np.float32)
        q_np = q[0, 0].detach().cpu().numpy().astype(np.float32)
        a_np = build_shape_map_np(template[f"shape_A_c{label_id}"], box, h, w)

        Q = 0.65 * p_np + 0.35 * q_np
        box_mask = make_box_mask_np(box, h, w)
        all_box |= box_mask

        fg = (
            box_mask
            & (p_np >= 0.50)
            & (Q >= 0.58)
            & (q_np >= fg_q_threshold)
            & (a_np >= 0.10)
        )

        bg = box_mask & (p_np <= 0.15) & (q_np <= 0.25)
        unknown = box_mask & (~fg) & (~bg)

        tri = merge_instance_into_tri(tri, fg, unknown, label_id)

        all_p_vals.append(p_np[box_mask])
        all_q_vals.append(q_np[box_mask])
        all_a_vals.append(a_np[box_mask])
        weak_030_040.append(((p_np > 0.30) & (p_np < 0.40) & box_mask).sum() / max(box_mask.sum(), 1))
        weak_040_050.append(((p_np >= 0.40) & (p_np < 0.50) & box_mask).sum() / max(box_mask.sum(), 1))

    if all_p_vals:
        p_cat = np.concatenate(all_p_vals)
        q_cat = np.concatenate(all_q_vals)
        a_cat = np.concatenate(all_a_vals)
    else:
        p_cat = np.array([0.0])
        q_cat = np.array([0.0])
        a_cat = np.array([0.0])

    stats = {
        "mean_P_inside_box": float(p_cat.mean()),
        "mean_qF_inside_box": float(q_cat.mean()),
        "mean_A_inside_box": float(a_cat.mean()),
        "weak_activation_ratio_0.30_0.40": float(np.mean(weak_030_040)) if weak_030_040 else 0.0,
        "weak_activation_ratio_0.40_0.50": float(np.mean(weak_040_050)) if weak_040_050 else 0.0,
        "box_fg_ratio": float(((tri > 0) & (tri != 255) & all_box).sum() / max(all_box.sum(), 1)),
        "box_unknown_ratio": float(((tri == 255) & all_box).sum() / max(all_box.sum(), 1)),
    }

    return tri, stats


def process_dataset(args, dataset):
    device = torch.device(args.device)

    fold_root = os.path.join(args.processed_root, dataset, args.fold)
    meta_dir = os.path.join(fold_root, "meta")

    manifest = load_json(os.path.join(meta_dir, "manifest.json"))
    prompts = load_json(os.path.join(fold_root, "prompts", "prompts_train.json"))
    split_records = load_json(os.path.join(meta_dir, f"full_box_split_{args.method}.json"))
    geometry_meta = load_json(os.path.join(meta_dir, "geometry_meta.json"))

    split_by_slice = {r["slice_name"]: r for r in split_records}
    manifest_by_slice = {get_slice_name(x): x for x in manifest}

    template = np.load(os.path.join(meta_dir, f"support_template_{args.method}.npz"))

    model = build_medsam(args.base_checkpoint, args.sac_checkpoint, device)

    out_teacher = os.path.join(fold_root, "pseudo_teacher", f"tri_train_{args.method}")
    out_student = os.path.join(fold_root, "pseudo_student", f"tri_train_{args.method}")
    ensure_dir(out_teacher)
    ensure_dir(out_student)

    if args.overwrite:
        for d in [out_teacher, out_student]:
            for fn in os.listdir(d):
                if fn.endswith(".npy"):
                    os.remove(os.path.join(d, fn))

    stats_path = os.path.join(meta_dir, f"pseudo_quality_stats_{args.method}.csv")
    fieldnames = [
        "slice_name", "label_mode",
        "fg_ratio", "bg_ratio", "unknown_ratio",
        "box_fg_ratio", "box_unknown_ratio",
        "weak_activation_ratio_0.30_0.40",
        "weak_activation_ratio_0.40_0.50",
        "mean_P_inside_box", "mean_qF_inside_box", "mean_A_inside_box",
        "num_connected_components", "largest_component_ratio",
        "pseudo_dice_on_full_subset", "pseudo_iou_on_full_subset",
        "false_activation_ratio_on_full_subset",
        "under_activation_ratio_on_full_subset",
    ]

    with open(stats_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        count = 0
        for slice_name, split_rec in split_by_slice.items():
            if args.max_samples > 0 and count >= args.max_samples:
                break

            if slice_name not in manifest_by_slice:
                continue
            if slice_name not in prompts:
                continue

            item = manifest_by_slice[slice_name]
            if item.get("split") != "train":
                raise RuntimeError(f"Leakage: non-train item in split file: {slice_name}")

            image_tensor, _ = load_image_tensor(resolve_path(fold_root, item, "teacher_img"), device)
            teacher_gt = load_mask(resolve_path(fold_root, item, "teacher_gt")).astype(np.uint8)
            native_gt = load_mask(resolve_path(fold_root, item, "native_gt")).astype(np.uint8)
            student_gt = load_mask(resolve_path(fold_root, item, "student_gt")).astype(np.uint8)

            tri_teacher, aux = generate_one_slice(
                model=model,
                image_tensor=image_tensor,
                prompt_meta=prompts[slice_name],
                label_mode=split_rec["label_mode"],
                teacher_gt=teacher_gt,
                template=template,
                device=device,
                fg_q_threshold=args.fg_q_threshold,
                q_threshold=args.q_threshold,
                save_prob_stats=True,
            )

            geom = geometry_meta.get(slice_name, geometry_meta.get(item.get("geometry_key", slice_name), {}))
            tri_native = restore_teacher_to_native(tri_teacher, geom, native_shape=native_gt.shape)
            tri_student = restore_native_mask_to_student(tri_native, geom)

            np.save(os.path.join(out_teacher, slice_name), tri_teacher.astype(np.uint8))
            np.save(os.path.join(out_student, slice_name), tri_student.astype(np.uint8))

            fg = (tri_teacher > 0) & (tri_teacher != 255)
            bg = tri_teacher == 0
            unk = tri_teacher == 255
            cc_n, largest_ratio = count_components(fg)

            dice_full = 0.0
            iou_full = 0.0
            false_ratio = 0.0
            under_ratio = 0.0

            if split_rec["label_mode"] == "full":
                label_id = 1
                dice_full, iou_full = dice_iou(tri_teacher, teacher_gt, label_id)
                gt_fg = teacher_gt == label_id
                pred_fg = tri_teacher == label_id
                false_ratio = float((pred_fg & (~gt_fg)).sum() / max(pred_fg.sum(), 1))
                under_ratio = float((gt_fg & (~pred_fg)).sum() / max(gt_fg.sum(), 1))

            row = {
                "slice_name": slice_name,
                "label_mode": split_rec["label_mode"],
                "fg_ratio": float(fg.mean()),
                "bg_ratio": float(bg.mean()),
                "unknown_ratio": float(unk.mean()),
                "box_fg_ratio": float(aux.get("box_fg_ratio", 0.0)),
                "box_unknown_ratio": float(aux.get("box_unknown_ratio", 0.0)),
                "weak_activation_ratio_0.30_0.40": float(aux.get("weak_activation_ratio_0.30_0.40", 0.0)),
                "weak_activation_ratio_0.40_0.50": float(aux.get("weak_activation_ratio_0.40_0.50", 0.0)),
                "mean_P_inside_box": float(aux.get("mean_P_inside_box", 0.0)),
                "mean_qF_inside_box": float(aux.get("mean_qF_inside_box", 0.0)),
                "mean_A_inside_box": float(aux.get("mean_A_inside_box", 0.0)),
                "num_connected_components": int(cc_n),
                "largest_component_ratio": float(largest_ratio),
                "pseudo_dice_on_full_subset": float(dice_full),
                "pseudo_iou_on_full_subset": float(iou_full),
                "false_activation_ratio_on_full_subset": float(false_ratio),
                "under_activation_ratio_on_full_subset": float(under_ratio),
            }
            writer.writerow(row)

            count += 1
            if count % args.log_every == 0:
                print(f"[{dataset}] generated {count}/{len(split_by_slice)}")

    print(f"[OK] pseudo generated: {dataset}")
    print(f"     teacher = {out_teacher}")
    print(f"     student = {out_student}")
    print(f"     stats   = {stats_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--sac_checkpoint", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--fg_q_threshold", type=float, default=0.45)
    parser.add_argument("--q_threshold", type=float, default=0.58)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda unavailable, fallback to cpu")
        args.device = "cpu"

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for ds in datasets:
        process_dataset(args, ds)


if __name__ == "__main__":
    main()
