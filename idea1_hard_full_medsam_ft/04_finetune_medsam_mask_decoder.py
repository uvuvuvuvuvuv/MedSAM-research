#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from idea1_common import METHOD_DEFAULT, RoundPaths, atomic_save_json, load_json, load_rgb_uint8

MODEL_TYPE = "vit_b"


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_medsam(checkpoint: Path, repo_root: Path, device: torch.device):
    sys.path.insert(0, str(repo_root.resolve()))
    from segment_anything import sam_model_registry

    model = sam_model_registry[MODEL_TYPE](checkpoint=str(checkpoint))
    model.to(device)
    return model


def freeze_mask_decoder_only(model) -> list[torch.nn.Parameter]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.mask_decoder.parameters():
        parameter.requires_grad = True
    model.image_encoder.eval()
    model.prompt_encoder.eval()
    model.mask_decoder.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable mask-decoder parameters")
    return params


def load_image_for_sam(path: Path, model, device: torch.device) -> torch.Tensor:
    image = load_rgb_uint8(path)
    tensor = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    # Match SamPredictor/model.forward preprocessing: pixel mean/std normalization and padding.
    return model.preprocess(tensor)


def load_target(path: Path, device: torch.device) -> torch.Tensor:
    target = np.load(path)
    target = np.asarray(target)
    target = np.squeeze(target)
    if target.ndim != 2:
        raise ValueError(f"Target must be 2D: {path}, shape={target.shape}")
    return torch.from_numpy((target > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)


def make_box(box: list[float], device: torch.device) -> torch.Tensor:
    return torch.tensor(np.asarray(box, dtype=np.float32).reshape(1, 4), device=device)


def forward_logits(model, image: torch.Tensor, box: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    with torch.no_grad():
        image_embedding = model.image_encoder(image)
        sparse, dense = model.prompt_encoder(points=None, boxes=box, masks=None)
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=image_embedding,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return F.interpolate(low_res_logits, size=target_hw, mode="bilinear", align_corners=False)


def dice_loss_from_logits(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = torch.sigmoid(logits.float())
    target = target.float()
    intersection = (probability * target).sum(dim=(1, 2, 3))
    denom = probability.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def load_pairs(path: Path) -> list[dict[str, Any]]:
    obj = load_json(path)
    if not isinstance(obj, list) or not obj:
        raise ValueError(f"pairs.json must be a non-empty list: {path}")
    required = {"pair_id", "teacher_img", "target_mask", "bbox_teacher"}
    for row in obj:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Pair misses {missing}: {row}")
    return [dict(x) for x in obj]


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-GT-only MedSAM mask-decoder fine-tuning.")
    parser.add_argument("--repo_root", type=Path, required=True)
    parser.add_argument("--fold_root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--pairs", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--batch_size", type=int, default=2, help="Effective pairs per optimizer step; physical SAM microbatch is 1.")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--dice_weight", type=float, default=1.0)
    parser.add_argument("--bce_weight", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp_init_scale", type=float, default=1024.0)
    parser.add_argument("--max_consecutive_amp_skips", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_steps <= 0 or args.batch_size <= 0:
        raise ValueError("--max_steps and --batch_size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA unavailable; using CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    set_seed(args.seed + args.round)

    paths = RoundPaths(args.fold_root, args.method, args.round)
    pairs_path = args.pairs or (paths.pair_dir / "pairs.json")
    pairs = load_pairs(pairs_path)
    output_ckpt = paths.teacher_dir / "medsam_ft.pth"
    log_path = paths.teacher_dir / "train_log.csv"
    summary_path = paths.teacher_dir / "train_summary.json"
    if output_ckpt.exists() and not args.overwrite:
        raise FileExistsError(output_ckpt)
    paths.teacher_dir.mkdir(parents=True, exist_ok=True)

    model = build_medsam(args.checkpoint, args.repo_root, device)
    trainable = freeze_mask_decoder_only(model)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(
        enabled=amp_enabled,
        init_scale=args.amp_init_scale,
    )

    fields = [
        "optimizer_step", "micro_step", "pair_id", "slice_name", "case_id", "label_id",
        "loss", "dice_loss", "bce_loss", "lr", "grad_norm", "amp_scale", "step_skipped",
    ]
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    start = time.perf_counter()
    optimizer_step = 0
    micro_step = 0
    order = list(range(len(pairs)))
    rng = random.Random(args.seed + args.round)
    optimizer.zero_grad(set_to_none=True)
    running = []
    consecutive_amp_skips = 0
    amp_skipped_steps = 0

    while optimizer_step < args.max_steps:
        rng.shuffle(order)
        for index in order:
            row = pairs[index]
            image = load_image_for_sam(Path(row["teacher_img"]), model, device)
            target = load_target(Path(row["target_mask"]), device)
            box = make_box(row["bbox_teacher"], device)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = forward_logits(model, image, box, target.shape[-2:])
                dice = dice_loss_from_logits(logits, target)
                bce = F.binary_cross_entropy_with_logits(logits.float(), target.float())
                loss = args.dice_weight * dice + args.bce_weight * bce
                scaled_loss = loss / float(args.batch_size)
            if not torch.isfinite(loss.detach()).all():
                raise RuntimeError(
                    f"Non-finite loss for pair={row['pair_id']} "
                    f"at optimizer_step={optimizer_step}, micro_step={micro_step}"
                )
            scaler.scale(scaled_loss).backward()
            micro_step += 1
            do_step = (micro_step % args.batch_size == 0)
            grad_norm_value = float("nan")
            amp_scale_value = float(scaler.get_scale()) if amp_enabled else 1.0
            step_skipped = False
            if do_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                grad_norm_value = float(torch.as_tensor(grad_norm).detach().cpu())
                finite_grad = math.isfinite(grad_norm_value)
                if amp_enabled:
                    old_scale = float(scaler.get_scale())
                    # GradScaler inspects the unscaled gradients and skips optimizer.step()
                    # automatically when inf/nan is present.
                    scaler.step(optimizer)
                    scaler.update()
                    new_scale = float(scaler.get_scale())
                    amp_scale_value = new_scale
                    step_skipped = (not finite_grad) or (new_scale < old_scale)
                    if step_skipped:
                        amp_skipped_steps += 1
                        consecutive_amp_skips += 1
                        print(
                            f"[WARN][{args.dataset}][round={args.round}] AMP overflow: "
                            f"optimizer_step={optimizer_step}, grad_norm={grad_norm_value}, "
                            f"scale={old_scale}->{new_scale}; step skipped",
                            flush=True,
                        )
                        if consecutive_amp_skips > args.max_consecutive_amp_skips:
                            raise RuntimeError(
                                f"Exceeded max consecutive AMP skips "
                                f"({args.max_consecutive_amp_skips}) at optimizer_step={optimizer_step}"
                            )
                    else:
                        optimizer_step += 1
                        consecutive_amp_skips = 0
                else:
                    if not finite_grad:
                        raise RuntimeError(
                            f"Non-finite FP32 gradient norm at optimizer_step={optimizer_step}, "
                            f"pair={row['pair_id']}"
                        )
                    optimizer.step()
                    optimizer_step += 1
                optimizer.zero_grad(set_to_none=True)

            log_row = {
                "optimizer_step": optimizer_step,
                "micro_step": micro_step,
                "pair_id": row["pair_id"],
                "slice_name": row.get("slice_name", ""),
                "case_id": row.get("case_id", ""),
                "label_id": row.get("label_id", 1),
                "loss": float(loss.detach().cpu()),
                "dice_loss": float(dice.detach().cpu()),
                "bce_loss": float(bce.detach().cpu()),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": grad_norm_value,
                "amp_scale": amp_scale_value,
                "step_skipped": int(step_skipped),
            }
            with log_path.open("a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(log_row)
            running.append(log_row["loss"])
            if do_step and (optimizer_step == 1 or optimizer_step % args.log_every == 0):
                mean_loss = float(np.mean(running[-args.log_every * args.batch_size :]))
                print(f"[{args.dataset}][round={args.round}] step={optimizer_step}/{args.max_steps} loss={mean_loss:.6f}")
            if optimizer_step >= args.max_steps:
                break

    # Standard SAM state_dict only, so the baseline generator can load it directly.
    torch.save(model.state_dict(), output_ckpt)
    elapsed = time.perf_counter() - start
    summary = {
        "dataset": args.dataset,
        "round": args.round,
        "method": args.method,
        "input_checkpoint": str(args.checkpoint),
        "output_checkpoint": str(output_ckpt),
        "pairs": str(pairs_path),
        "num_pairs": len(pairs),
        "optimizer_steps": optimizer_step,
        "micro_steps": micro_step,
        "effective_batch_size": args.batch_size,
        "physical_microbatch": 1,
        "trainable": "mask_decoder_only",
        "loss": "1.0*Dice + 1.0*BCEWithLogits",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "amp": amp_enabled,
        "amp_init_scale": args.amp_init_scale,
        "amp_skipped_steps": amp_skipped_steps,
        "elapsed_seconds": elapsed,
    }
    atomic_save_json(summary, summary_path)
    atomic_save_json(
        {"stage": "medsam_finetune", "elapsed_seconds": elapsed, "num_outputs": 1},
        paths.teacher_dir / "stage_time_finetune.json",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
