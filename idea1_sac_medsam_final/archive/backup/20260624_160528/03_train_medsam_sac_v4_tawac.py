import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import copy
import csv
import json
import os
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


METHOD_DEFAULT = "idea1_sac_medsam_final"
MODEL_TYPE = "vit_b"


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    return x


def load_gt_tensor(path, device, label_id):
    gt = np.load(path)
    if gt.ndim == 3:
        if gt.shape[-1] == 1:
            gt = gt[..., 0]
        elif gt.shape[0] == 1:
            gt = gt[0]
        else:
            raise ValueError(f"Unexpected GT shape: {gt.shape}")

    y = (gt.astype(np.int64) == int(label_id)).astype(np.float32)
    y = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float().to(device)
    return y


def build_medsam(checkpoint, device):
    from segment_anything import sam_model_registry

    model = sam_model_registry[MODEL_TYPE](checkpoint=checkpoint)
    model.to(device)
    return model


def freeze_for_mask_decoder_only(model):
    for p in model.parameters():
        p.requires_grad = False

    for p in model.mask_decoder.parameters():
        p.requires_grad = True

    model.prompt_encoder.eval()
    model.image_encoder.eval()
    model.mask_decoder.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    return trainable


def make_box_tensor(box, device):
    box = np.array(box, dtype=np.float32).reshape(1, 4)
    return torch.from_numpy(box).float().to(device)


def make_box_mask(box, h, w, device):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(w - 1, int(np.floor(x1))))
    y1 = max(0, min(h - 1, int(np.floor(y1))))
    x2 = max(0, min(w, int(np.ceil(x2))))
    y2 = max(0, min(h, int(np.ceil(y2))))

    m = torch.zeros((1, 1, h, w), dtype=torch.float32, device=device)
    if x2 > x1 and y2 > y1:
        m[:, :, y1:y2, x1:x2] = 1.0
    return m


def downsample_mask(mask, size_hw):
    return F.interpolate(mask.float(), size=size_hw, mode="nearest")


def dice_loss(prob, target, eps=1e-6):
    prob = prob.float()
    target = target.float()
    inter = (prob * target).sum(dim=(1, 2, 3))
    denom = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    dice = (2 * inter + eps) / denom
    return 1.0 - dice.mean()


def bce_mean(prob, target, weight=None, eps=1e-6):
    prob = prob.clamp(eps, 1.0 - eps)
    loss = F.binary_cross_entropy(prob, target.float(), reduction="none")
    if weight is not None:
        loss = loss * weight.float()
        denom = weight.float().sum().clamp_min(1.0)
        return loss.sum() / denom
    return loss.mean()


def tv_smooth_loss(prob):
    dy = torch.abs(prob[:, :, 1:, :] - prob[:, :, :-1, :]).mean()
    dx = torch.abs(prob[:, :, :, 1:] - prob[:, :, :, :-1]).mean()
    return dx + dy


def forward_medsam(model, image_tensor, box_tensor):
    """
    image_tensor: [1,3,1024,1024]
    box_tensor: [1,4], teacher-space bbox
    """
    image_embedding = model.image_encoder(image_tensor)

    sparse_embeddings, dense_embeddings = model.prompt_encoder(
        points=None,
        boxes=box_tensor,
        masks=None,
    )

    low_res_logits, iou_pred = model.mask_decoder(
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


@torch.no_grad()
def forward_medsam_ema(model, image_tensor, box_tensor):
    model.eval()
    low_res_logits, logits_1024, image_embedding = forward_medsam(model, image_tensor, box_tensor)
    return low_res_logits.detach(), logits_1024.detach(), image_embedding.detach()


def compute_qf_from_embedding(image_embedding, proto_fg, proto_bg, out_size_hw):
    """
    image_embedding: [1,C,64,64]
    proto_fg: [Kf,C]
    proto_bg: [Kb,C]
    return qF: [1,1,out_h,out_w]
    """
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

    # temperature already used in support construction; here use sigmoid-style normalization
    q = torch.exp(max_fg / 0.07) / (torch.exp(max_fg / 0.07) + torch.exp(max_bg / 0.07) + 1e-6)
    q = q.reshape(1, 1, h, w)

    q = F.interpolate(q, size=out_size_hw, mode="bilinear", align_corners=False)
    return q.clamp(0, 1).detach()



def compute_proto_scores_from_embedding(image_embedding, proto_fg, proto_bg, out_size_hw):
    """
    Return:
      qF/qB: normalized foreground/background posterior scores.
      sim_fg/sim_bg: raw max cosine similarity to foreground/background prototypes.

    Compared with qB = 1 - qF, sim_bg - sim_fg provides a less saturated
    background-prototype margin for box-internal false activation suppression.
    """
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

    qf = torch.exp(max_fg / 0.07) / (
        torch.exp(max_fg / 0.07) + torch.exp(max_bg / 0.07) + 1e-6
    )
    qb = 1.0 - qf

    qf = qf.reshape(1, 1, h, w)
    qb = qb.reshape(1, 1, h, w)
    max_fg = max_fg.reshape(1, 1, h, w)
    max_bg = max_bg.reshape(1, 1, h, w)

    qf = F.interpolate(qf, size=out_size_hw, mode="bilinear", align_corners=False).clamp(0, 1)
    qb = F.interpolate(qb, size=out_size_hw, mode="bilinear", align_corners=False).clamp(0, 1)
    max_fg = F.interpolate(max_fg, size=out_size_hw, mode="bilinear", align_corners=False)
    max_bg = F.interpolate(max_bg, size=out_size_hw, mode="bilinear", align_corners=False)

    return qf.detach(), qb.detach(), max_fg.detach(), max_bg.detach()


def build_shape_map(shape_A, box, out_h, out_w, device):
    """
    shape_A: [64,64]
    map to full image by resizing into bbox.
    """
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = max(0, min(out_w - 1, int(np.floor(x1))))
    y1 = max(0, min(out_h - 1, int(np.floor(y1))))
    x2 = max(0, min(out_w, int(np.ceil(x2))))
    y2 = max(0, min(out_h, int(np.ceil(y2))))

    out = torch.zeros((1, 1, out_h, out_w), dtype=torch.float32, device=device)

    if x2 <= x1 or y2 <= y1:
        return out

    a = torch.from_numpy(shape_A).float().to(device).view(1, 1, 64, 64)
    a = F.interpolate(a, size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)
    out[:, :, y1:y2, x1:x2] = a
    return out.clamp(0, 1)


def parse_train_entries(fold_root, manifest, prompts, split_records):
    manifest_by_slice = {get_slice_name(x): x for x in manifest}
    split_by_slice = {r["slice_name"]: r for r in split_records}

    entries = []
    for slice_name, rec in split_by_slice.items():
        item = manifest_by_slice.get(slice_name)
        if item is None:
            continue

        prompt_meta = prompts.get(slice_name)
        if prompt_meta is None:
            continue

        instances = prompt_meta.get("instances", [])
        if not instances:
            continue

        for ins in instances:
            bbox = ins.get("bbox_teacher", ins.get("bbox"))
            label_id = int(ins.get("label_id", 1))

            entries.append({
                "slice_name": slice_name,
                "label_mode": rec["label_mode"],
                "case_id": rec.get("case_id", ""),
                "label_id": label_id,
                "bbox": bbox,
                "teacher_img": resolve_path(fold_root, item, "teacher_img"),
                "teacher_gt": resolve_path(fold_root, item, "teacher_gt"),
            })

    return entries


def update_ema(model, ema_model, decay):
    with torch.no_grad():
        msd = model.state_dict()
        esd = ema_model.state_dict()
        for k in esd.keys():
            if k in msd:
                esd[k].copy_(esd[k] * decay + msd[k].detach() * (1.0 - decay))


def save_checkpoint(path, model, extra=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model": model.state_dict(),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def train_one_dataset(args, dataset):
    device = torch.device(args.device)

    fold_root = os.path.join(args.processed_root, dataset, args.fold)
    meta_dir = os.path.join(fold_root, "meta")

    manifest = load_json(os.path.join(meta_dir, "manifest.json"))
    prompts = load_json(os.path.join(fold_root, "prompts", "prompts_train.json"))
    split_records = load_json(os.path.join(meta_dir, f"full_box_split_{args.method}.json"))

    template_npz = os.path.join(meta_dir, f"support_template_{args.method}.npz")
    template = np.load(template_npz)
    class_ids = [int(x) for x in template["class_ids"].tolist()]

    entries = parse_train_entries(fold_root, manifest, prompts, split_records)

    full_entries = [e for e in entries if e["label_mode"] == "full"]
    box_entries = [e for e in entries if e["label_mode"] == "box"]

    if len(full_entries) == 0:
        raise RuntimeError(f"No full entries for {dataset}")
    if len(box_entries) == 0:
        raise RuntimeError(f"No box entries for {dataset}")

    print(f"[{dataset}] entries: full={len(full_entries)} box={len(box_entries)} total={len(entries)}")

    model = build_medsam(args.checkpoint, device)
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()

    trainable_params = freeze_for_mask_decoder_only(model)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    out_dir = os.path.join(args.out_root, dataset, args.fold)
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, f"medsam_ft_log_{args.method}.csv")
    last_ckpt = os.path.join(out_dir, "medsam_sac_last.pth")
    ema_ckpt = os.path.join(out_dir, "medsam_sac_ema.pth")

    if args.overwrite and os.path.exists(log_path):
        os.remove(log_path)

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch", "step", "dataset", "slice_name", "label_mode",
                "loss", "loss_full", "loss_out", "loss_seed",
                "loss_wac", "loss_proto", "loss_smooth", "loss_fas",
                "mean_p", "mean_qf", "mean_qb", "mean_sim_fg", "mean_sim_bg", "fas_weight_sum", "fas_warm",
                "weak1_sum", "weak2_sum", "strong_sum", "gate1_sum", "gate2_sum",
                "box_area_low", "seed_sum", "unc_sum",
            ],
        )
        writer.writeheader()

    global_step = 0

    for epoch in range(args.epochs):
        random.shuffle(full_entries)
        random.shuffle(box_entries)

        # 1:3 full/box 混合。先构造一个循环任务表。
        schedule = []
        n = max(len(full_entries), len(box_entries))
        for i in range(n):
            if i < len(full_entries):
                schedule.append(full_entries[i])
            for j in range(3):
                idx = i * 3 + j
                if idx < len(box_entries):
                    schedule.append(box_entries[idx])

        for entry in schedule:
            if args.max_steps > 0 and global_step >= args.max_steps:
                break

            model.train()
            model.prompt_encoder.eval()
            model.image_encoder.eval()

            label_id = int(entry["label_id"])
            bbox = entry["bbox"]

            image = load_image_tensor(entry["teacher_img"], device)
            target = load_gt_tensor(entry["teacher_gt"], device, label_id)
            box_tensor = make_box_tensor(bbox, device)

            optimizer.zero_grad(set_to_none=True)

            # image encoder frozen，但 mask decoder 需要梯度
            with torch.no_grad():
                image_embedding = model.image_encoder(image)

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
                size=image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            p_1024 = torch.sigmoid(logits_1024)
            p_low = torch.sigmoid(low_res_logits)

            with torch.no_grad():
                ema_low_logits, ema_logits_1024, ema_embedding = forward_medsam_ema(
                    ema_model, image, box_tensor
                )
                p_ema_low = torch.sigmoid(ema_low_logits)
                p_ema_1024 = torch.sigmoid(ema_logits_1024)

            low_hw = p_low.shape[-2:]
            full_hw = p_1024.shape[-2:]

            target_low = F.interpolate(target, size=low_hw, mode="nearest")
            box_mask_1024 = make_box_mask(bbox, full_hw[0], full_hw[1], device)
            box_mask_low = F.interpolate(box_mask_1024, size=low_hw, mode="nearest")

            proto_fg = template[f"proto_fg_c{label_id}"]
            proto_bg = template[f"proto_bg_c{label_id}"]
            shape_A = template[f"shape_A_c{label_id}"]

            qf_low, qb_low, sim_fg_low, sim_bg_low = compute_proto_scores_from_embedding(
                image_embedding, proto_fg, proto_bg, low_hw
            )
            shape_low = build_shape_map(shape_A, bbox, low_hw[0], low_hw[1], device)

            loss_full = torch.tensor(0.0, device=device)
            loss_out = torch.tensor(0.0, device=device)
            loss_seed = torch.tensor(0.0, device=device)
            loss_wac = torch.tensor(0.0, device=device)
            loss_proto = torch.tensor(0.0, device=device)
            loss_smooth = torch.tensor(0.0, device=device)
            loss_fas = torch.tensor(0.0, device=device)
            fas_weight_sum = torch.tensor(0.0, device=device)
            fas_warm = torch.tensor(0.0, device=device)

            weak1_sum = 0.0
            weak2_sum = 0.0
            strong_sum = 0.0
            gate1_sum = 0.0
            gate2_sum = 0.0
            box_area_low = float(box_mask_low.sum().detach().cpu())
            seed_sum = 0.0
            unc_sum = 0.0

            if entry["label_mode"] == "full":
                loss_full = dice_loss(p_1024, target) + bce_mean(p_1024, target)
                loss = loss_full

            else:
                outside = 1.0 - box_mask_1024
                loss_out = bce_mean(p_1024, torch.zeros_like(p_1024), outside)

                # seed loss at low resolution
                seed = ((p_ema_low > 0.65) & (qf_low > 0.55) & (box_mask_low > 0.5)).float()
                seed_sum = float(seed.sum().detach().cpu())
                if seed.sum() > 0:
                    loss_seed = bce_mean(p_low, torch.ones_like(p_low), seed)

                # prototype consistency in uncertain region
                unc = ((p_low > 0.15) & (p_low < 0.85) & (box_mask_low > 0.5)).float()
                unc_sum = float(unc.sum().detach().cpu())
                if unc.sum() > 0:
                    loss_proto = bce_mean(p_low, qf_low, unc)

                # TA-WAC: teacher-aligned weak activation compensation
                # This version follows the teacher's idea more explicitly:
                # weak activations are not simply pushed to foreground.
                # Instead, they are fitted to a soft target derived from
                # neighboring reliable strong activations, foreground-prototype
                # confidence, and full-mask-derived shape prior.
                weak1 = ((p_low > 0.30) & (p_low < 0.40) & (box_mask_low > 0.5)).float()
                weak2 = ((p_low >= 0.40) & (p_low < 0.50) & (box_mask_low > 0.5)).float()

                strong = (
                    (p_ema_low >= args.tawac_strong_thresh)
                    & (qf_low > args.tawac_qf_thresh)
                    & (box_mask_low > 0.5)
                ).float()

                k = int(args.tawac_kernel)
                if k % 2 == 0:
                    k += 1
                pad = k // 2

                # Neighboring reliable strong activation.
                local_strong_prob = F.max_pool2d(p_ema_low * strong, kernel_size=k, stride=1, padding=pad)
                local_presence = F.max_pool2d(strong, kernel_size=k, stride=1, padding=pad)

                # Full-mask-derived shape reliability.
                shape_reliability = (1.0 - 4.0 * shape_low * (1.0 - shape_low)).clamp(0, 1)
                shape_gate = (0.50 + 0.50 * shape_reliability).clamp(0.50, 1.0)

                # Foreground prototype confidence.
                qf_gate = (0.25 + 0.75 * qf_low).clamp(0.25, 1.0)

                # Neighbor similarity / reliability proxy.
                # It encodes: nearby reliable strong activation + target-like appearance + shape prior.
                neighbor_quality = (local_strong_prob.clamp(0, 1) * qf_gate * shape_gate).clamp(0, 1)

                # Soft ideal activation values. We do not push weak pixels directly to 1.
                # This is safer for boundary/ambiguous medical structures.
                target1 = (
                    args.tawac_target1_base
                    + args.tawac_target1_scale * neighbor_quality
                ).clamp(0.55, 0.82).detach()

                target2 = (
                    args.tawac_target2_base
                    + args.tawac_target2_scale * neighbor_quality
                ).clamp(0.60, 0.88).detach()

                gate_base = (
                    (local_presence > args.tawac_presence_thresh).float()
                    * shape_gate
                    * qf_gate
                    * (0.50 + 0.50 * local_strong_prob.clamp(0, 1))
                )

                gate1 = weak1 * gate_base
                gate2 = weak2 * gate_base

                weak1_sum = float(weak1.sum().detach().cpu())
                weak2_sum = float(weak2.sum().detach().cpu())
                strong_sum = float(strong.sum().detach().cpu())
                gate1_sum = float(gate1.sum().detach().cpu())
                gate2_sum = float(gate2.sum().detach().cpu())

                p_safe = p_low.clamp(1e-4, 1.0 - 1e-4)

                if gate1.sum() > 0:
                    wac_map1 = F.binary_cross_entropy(
                        p_safe,
                        target1,
                        reduction="none",
                    )
                    loss_wac = loss_wac + (wac_map1 * gate1).sum() / gate1.sum().clamp_min(1.0)

                if gate2.sum() > 0:
                    wac_map2 = F.binary_cross_entropy(
                        p_safe,
                        target2,
                        reduction="none",
                    )
                    loss_wac = loss_wac + 0.5 * (wac_map2 * gate2).sum() / gate2.sum().clamp_min(1.0)

                # ------------------------------------------------------------
                # FAS: box-internal False Activation Suppression.
                # Motivation:
                #   WAC compensates weak object activation, while FAS suppresses
                #   high-probability pixels that are more background-like according
                #   to support prototypes and shape prior.
                # The loss is intentionally soft: p^2 is used instead of BCE(p, 0)
                # to avoid aggressively suppressing ambiguous true foreground.
                # ------------------------------------------------------------
                if args.lambda_fas > 0:
                    if args.fas_warmup_steps > 0:
                        fas_warm_value = max(
                            0.0,
                            min(1.0, (float(global_step) - float(args.fas_warmup_steps)) / float(max(args.fas_warmup_steps, 1))),
                        )
                    else:
                        fas_warm_value = 1.0
                    fas_warm = torch.tensor(fas_warm_value, device=device)

                    p_gate = torch.sigmoid((p_low - args.fas_p_thresh) / args.fas_temp)
                    bg_post_gate = torch.sigmoid((qb_low - qf_low - args.fas_bg_margin) / args.fas_temp)
                    bg_raw_gate = torch.sigmoid((sim_bg_low - sim_fg_low - args.fas_sim_margin) / args.fas_sim_temp)
                    bg_gate = 0.5 * bg_post_gate + 0.5 * bg_raw_gate
                    shape_bg_gate = (1.0 - shape_low).clamp(0.0, 1.0)

                    protect_qf = (qf_low > args.fas_fg_protect).float()
                    protect_shape = (shape_low > args.fas_shape_protect).float()
                    protect_local = (local_presence > 0.50).float()
                    protected_fg = (protect_qf + protect_shape + protect_local).clamp(0.0, 1.0)

                    fas_weight = (
                        box_mask_low
                        * p_gate
                        * bg_gate
                        * shape_bg_gate
                        * (1.0 - protected_fg)
                    ).detach()

                    fas_weight_sum = fas_weight.sum()
                    if float(fas_weight_sum.detach().cpu()) > 0:
                        loss_fas = (fas_weight * p_low.pow(2)).sum() / fas_weight_sum.clamp_min(1.0)

                loss_smooth = tv_smooth_loss(p_low)

                loss = (
                    args.lambda_out * loss_out
                    + args.lambda_seed * loss_seed
                    + args.lambda_wac * loss_wac
                    + args.lambda_proto * loss_proto
                    + args.lambda_smooth * loss_smooth
                    + args.lambda_fas * fas_warm * loss_fas
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            update_ema(model, ema_model, args.ema_decay)

            global_step += 1

            row = {
                "epoch": epoch,
                "step": global_step,
                "dataset": dataset,
                "slice_name": entry["slice_name"],
                "label_mode": entry["label_mode"],
                "loss": float(loss.detach().cpu()),
                "loss_full": float(loss_full.detach().cpu()),
                "loss_out": float(loss_out.detach().cpu()),
                "loss_seed": float(loss_seed.detach().cpu()),
                "loss_wac": float(loss_wac.detach().cpu()),
                "loss_proto": float(loss_proto.detach().cpu()),
                "loss_smooth": float(loss_smooth.detach().cpu()),
                "loss_fas": float(loss_fas.detach().cpu()),
                "mean_p": float(p_low.detach().mean().cpu()),
                "mean_qf": float(qf_low.detach().mean().cpu()),
                "mean_qb": float(qb_low.detach().mean().cpu()),
                "mean_sim_fg": float(sim_fg_low.detach().mean().cpu()),
                "mean_sim_bg": float(sim_bg_low.detach().mean().cpu()),
                "fas_weight_sum": float(fas_weight_sum.detach().cpu()),
                "fas_warm": float(fas_warm.detach().cpu()),
                "weak1_sum": weak1_sum,
                "weak2_sum": weak2_sum,
                "strong_sum": strong_sum,
                "gate1_sum": gate1_sum,
                "gate2_sum": gate2_sum,
                "box_area_low": box_area_low,
                "seed_sum": seed_sum,
                "unc_sum": unc_sum,
            }

            with open(log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writerow(row)

            if global_step % args.log_every == 0:
                print(
                    f"[{dataset}] epoch={epoch} step={global_step} "
                    f"mode={entry['label_mode']} loss={row['loss']:.4f} "
                    f"full={row['loss_full']:.4f} out={row['loss_out']:.4f} "
                    f"seed={row['loss_seed']:.4f} wac={row['loss_wac']:.4f} "
                    f"proto={row['loss_proto']:.4f} fas={row['loss_fas']:.4f} "
                    f"p={row['mean_p']:.4f} qf={row['mean_qf']:.4f} qb={row['mean_qb']:.4f} "
                    f"fas_w={row['fas_weight_sum']:.1f}"
                )

        save_checkpoint(last_ckpt, model, extra={"epoch": epoch, "global_step": global_step, "method": args.method})
        save_checkpoint(ema_ckpt, ema_model, extra={"epoch": epoch, "global_step": global_step, "method": args.method})

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    summary = {
        "dataset": dataset,
        "method": args.method,
        "epochs": args.epochs,
        "global_step": global_step,
        "checkpoint_last": last_ckpt,
        "checkpoint_ema": ema_ckpt,
        "log_path": log_path,
        "trainable": "mask_decoder_only",
        "lambda_out": args.lambda_out,
        "lambda_seed": args.lambda_seed,
        "lambda_wac": args.lambda_wac,
        "lambda_proto": args.lambda_proto,
        "lambda_smooth": args.lambda_smooth,
        "lambda_fas": args.lambda_fas,
        "fas_warmup_steps": args.fas_warmup_steps,
        "fas_p_thresh": args.fas_p_thresh,
        "fas_bg_margin": args.fas_bg_margin,
        "fas_fg_protect": args.fas_fg_protect,
        "fas_shape_protect": args.fas_shape_protect,
        "fas_temp": args.fas_temp,
        "fas_sim_margin": args.fas_sim_margin,
        "fas_sim_temp": args.fas_sim_temp,
    }
    save_json(summary, os.path.join(out_dir, f"medsam_ft_summary_{args.method}.json"))

    print(f"[OK] train done: {dataset}")
    print(f"     last = {last_ckpt}")
    print(f"     ema  = {ema_ckpt}")
    print(f"     log  = {log_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--ema_decay", type=float, default=0.99)

    parser.add_argument("--lambda_out", type=float, default=1.0)
    parser.add_argument("--lambda_seed", type=float, default=0.5)
    parser.add_argument("--lambda_wac", type=float, default=0.5)
    parser.add_argument("--lambda_proto", type=float, default=0.3)
    parser.add_argument("--lambda_smooth", type=float, default=0.03)

    # TA-WAC parameters: teacher-aligned weak activation compensation.
    parser.add_argument("--tawac_kernel", type=int, default=31)
    parser.add_argument("--tawac_presence_thresh", type=float, default=0.50)
    parser.add_argument("--tawac_strong_thresh", type=float, default=0.50)
    parser.add_argument("--tawac_qf_thresh", type=float, default=0.55)
    parser.add_argument("--tawac_target1_base", type=float, default=0.52)
    parser.add_argument("--tawac_target1_scale", type=float, default=0.30)
    parser.add_argument("--tawac_target2_base", type=float, default=0.58)
    parser.add_argument("--tawac_target2_scale", type=float, default=0.30)

    # FAS is disabled by default, so previous/default experiments are unchanged
    # unless --lambda_fas is explicitly set.
    parser.add_argument("--lambda_fas", type=float, default=0.0)
    parser.add_argument("--fas_warmup_steps", type=int, default=200)
    parser.add_argument("--fas_p_thresh", type=float, default=0.45)
    parser.add_argument("--fas_bg_margin", type=float, default=0.03)
    parser.add_argument("--fas_fg_protect", type=float, default=0.55)
    parser.add_argument("--fas_shape_protect", type=float, default=0.60)
    parser.add_argument("--fas_temp", type=float, default=0.05)
    parser.add_argument("--fas_sim_margin", type=float, default=0.02)
    parser.add_argument("--fas_sim_temp", type=float, default=0.05)

    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda unavailable, fallback to cpu")
        args.device = "cpu"

    set_seed(args.seed)

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for ds in datasets:
        train_one_dataset(args, ds)


if __name__ == "__main__":
    main()
