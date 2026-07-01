#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare two directories of tri-state pseudo labels pixel by pixel."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--old-dir", type=Path, required=True)
    p.add_argument("--new-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tag", type=str, default="comparison")
    p.add_argument("--require-exact", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    old_files = {p.name: p for p in args.old_dir.glob("*.npy")}
    new_files = {p.name: p for p in args.new_dir.glob("*.npy")}

    old_names = set(old_files)
    new_names = set(new_files)
    common = sorted(old_names & new_names)
    missing_in_old = sorted(new_names - old_names)
    missing_in_new = sorted(old_names - new_names)

    if not common:
        raise RuntimeError("No common .npy files found.")

    rows: List[Dict[str, object]] = []
    transition = Counter()
    exact_count = 0
    total_pixels = 0
    total_diff = 0
    invalid_files = []

    for name in common:
        old = np.load(old_files[name])
        new = np.load(new_files[name])

        shape_equal = old.shape == new.shape
        dtype_equal = old.dtype == new.dtype

        old_values = sorted(np.unique(old).tolist())
        new_values = sorted(np.unique(new).tolist())
        allowed = {0, 255} | {int(v) for v in old_values + new_values if int(v) > 0 and int(v) != 255}

        if not shape_equal:
            rows.append({
                "slice_name": name,
                "shape_equal": False,
                "dtype_equal": dtype_equal,
                "old_shape": str(tuple(old.shape)),
                "new_shape": str(tuple(new.shape)),
                "old_dtype": str(old.dtype),
                "new_dtype": str(new.dtype),
                "old_values": str(old_values),
                "new_values": str(new_values),
                "exact_equal": False,
                "diff_pixels": "",
                "diff_ratio": "",
            })
            continue

        exact = np.array_equal(old, new)
        diff = old != new
        diff_pixels = int(diff.sum())
        pixels = int(old.size)
        diff_ratio = diff_pixels / max(pixels, 1)

        exact_count += int(exact)
        total_pixels += pixels
        total_diff += diff_pixels

        if diff_pixels:
            pairs = np.stack([old[diff].astype(np.int64), new[diff].astype(np.int64)], axis=1)
            uniq, counts = np.unique(pairs, axis=0, return_counts=True)
            for (a, b), c in zip(uniq.tolist(), counts.tolist()):
                transition[(int(a), int(b))] += int(c)

        if not set(old_values).issubset(allowed) or not set(new_values).issubset(allowed):
            invalid_files.append(name)

        rows.append({
            "slice_name": name,
            "shape_equal": shape_equal,
            "dtype_equal": dtype_equal,
            "old_shape": str(tuple(old.shape)),
            "new_shape": str(tuple(new.shape)),
            "old_dtype": str(old.dtype),
            "new_dtype": str(new.dtype),
            "old_values": str(old_values),
            "new_values": str(new_values),
            "exact_equal": exact,
            "diff_pixels": diff_pixels,
            "diff_ratio": diff_ratio,
        })

    rows.sort(key=lambda r: (float(r["diff_ratio"]) if r["diff_ratio"] != "" else 1.0), reverse=True)

    csv_path = args.out_dir / f"{args.tag}_per_file.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    transition_rows = [
        {"old_label": a, "new_label": b, "pixels": c}
        for (a, b), c in sorted(transition.items())
    ]
    transition_path = args.out_dir / f"{args.tag}_transitions.csv"
    with transition_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["old_label", "new_label", "pixels"])
        writer.writeheader()
        writer.writerows(transition_rows)

    summary = {
        "tag": args.tag,
        "old_dir": str(args.old_dir),
        "new_dir": str(args.new_dir),
        "old_file_count": len(old_files),
        "new_file_count": len(new_files),
        "common_file_count": len(common),
        "missing_in_old_count": len(missing_in_old),
        "missing_in_new_count": len(missing_in_new),
        "missing_in_old_examples": missing_in_old[:20],
        "missing_in_new_examples": missing_in_new[:20],
        "shape_equal_count": sum(bool(r["shape_equal"]) for r in rows),
        "dtype_equal_count": sum(bool(r["dtype_equal"]) for r in rows),
        "exact_equal_count": exact_count,
        "non_exact_count": len(common) - exact_count,
        "exact_equal_fraction": exact_count / len(common),
        "total_pixels_compared": total_pixels,
        "total_diff_pixels": total_diff,
        "overall_diff_ratio": total_diff / max(total_pixels, 1),
        "max_file_diff_ratio": max(
            (float(r["diff_ratio"]) for r in rows if r["diff_ratio"] != ""),
            default=0.0,
        ),
        "invalid_files": invalid_files,
        "transition_matrix": {
            f"{a}->{b}": c for (a, b), c in sorted(transition.items())
        },
        "per_file_csv": str(csv_path),
        "transition_csv": str(transition_path),
    }

    summary_path = args.out_dir / f"{args.tag}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.require_exact:
        assert not missing_in_old, missing_in_old[:20]
        assert not missing_in_new, missing_in_new[:20]
        assert summary["shape_equal_count"] == len(common)
        assert summary["dtype_equal_count"] == len(common)
        assert exact_count == len(common), (
            f"{len(common) - exact_count}/{len(common)} files differ; "
            f"overall_diff_ratio={summary['overall_diff_ratio']:.8f}"
        )
        print("[OK] exact pixel-wise equivalence verified")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
