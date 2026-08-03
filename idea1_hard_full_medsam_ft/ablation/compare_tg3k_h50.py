#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def read(path: Path) -> dict:
    x = json.loads(path.read_text(encoding="utf-8"))
    return {
        "summary_path": str(path),
        "n": int(x["num_eval_slices"]),
        "dice": float(x["dice_macro"]["mean"]),
        "iou": float(x["iou_macro"]["mean"]),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a_summary", type=Path, required=True)
    p.add_argument("--h_summary", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    args = p.parse_args()
    a, h = read(args.a_summary), read(args.h_summary)
    if a["n"] != h["n"]:
        raise RuntimeError("A50 and H50 evaluated different sample counts")
    rows = [
        {"variant": "A50_current_idea1", **a},
        {"variant": "H50_hard20_fullonly", **h},
    ]
    for r in rows:
        r["delta_dice_vs_H50"] = r["dice"] - h["dice"]
        r["delta_iou_vs_H50"] = r["iou"] - h["iou"]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with (out / "tg3k_h50_ablation.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    result = {
        "a50_minus_h50_dice": a["dice"] - h["dice"],
        "a50_minus_h50_iou": a["iou"] - h["iou"],
        "rows": rows,
    }
    (out / "tg3k_h50_ablation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    text = (
        f"A50 Dice={a['dice']:.9f}, IoU={a['iou']:.9f}\n"
        f"H50 Dice={h['dice']:.9f}, IoU={h['iou']:.9f}\n"
        f"A50-H50 Dice={a['dice']-h['dice']:+.9f}\n"
        f"A50-H50 IoU ={a['iou']-h['iou']:+.9f}\n"
    )
    (out / "tg3k_h50_ablation.txt").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
