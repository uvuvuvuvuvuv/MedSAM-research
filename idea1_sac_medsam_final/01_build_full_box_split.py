from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


METHOD_DEFAULT = "idea1_sac_medsam_final"
KNOWN_3D_DATASETS = {"btcv", "synapse", "acdc", "prostate158"}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def get_slice_name(item: dict[str, Any]) -> str:
    if item.get("slice_name"):
        return str(item["slice_name"])
    for key in ("student_img", "teacher_img", "student_gt", "teacher_gt"):
        if item.get(key):
            return Path(str(item[key])).name
    raise KeyError(f"Cannot infer slice_name from keys={list(item.keys())}")


def get_case_id(item: dict[str, Any]) -> str:
    return str(item.get("case_id", "unknown_case"))


def parse_class_ids(label_meta: Any) -> list[int]:
    if not isinstance(label_meta, dict):
        return [1]

    candidates: list[int] = []

    for key in ("unique_labels", "class_ids"):
        values = label_meta.get(key)
        if isinstance(values, list):
            candidates.extend(int(v) for v in values if int(v) > 0)

    labels = label_meta.get("labels")
    if isinstance(labels, dict):
        for value in labels.keys():
            try:
                class_id = int(value)
            except (TypeError, ValueError):
                continue
            if class_id > 0:
                candidates.append(class_id)

    if not candidates and "num_classes" in label_meta:
        num_classes = int(label_meta["num_classes"])
        # Most segmentation metadata includes background in num_classes.
        candidates.extend(range(1, max(num_classes, 2)))

    class_ids = sorted(set(candidates))
    return class_ids or [1]


def infer_is_3d(dataset: str, split_meta: dict[str, Any]) -> bool:
    if dataset.lower() in KNOWN_3D_DATASETS:
        return True
    for key in ("is_3d", "volume_level_split", "has_volume"):
        if key in split_meta:
            return bool(split_meta[key])
    return False


def choose_full_2d(
    train_items: list[dict[str, Any]],
    full_ratio: float,
    seed: int,
) -> set[str]:
    items = list(train_items)
    random.Random(seed).shuffle(items)
    n_full = max(1, int(round(len(items) * full_ratio)))
    return {get_slice_name(item) for item in items[:n_full]}


def choose_full_3d_by_case(
    train_items: list[dict[str, Any]],
    full_ratio: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    case_to_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in train_items:
        case_to_items[get_case_id(item)].append(item)

    cases = sorted(case_to_items)
    random.Random(seed).shuffle(cases)
    n_full_cases = max(1, int(round(len(cases) * full_ratio)))
    full_cases = set(cases[:n_full_cases])

    full_slices = {
        get_slice_name(item)
        for case_id in full_cases
        for item in case_to_items[case_id]
    }
    return full_slices, full_cases


def process_dataset(
    processed_root: Path,
    dataset: str,
    fold: str,
    method: str,
    full_ratio: float,
    seed: int,
    overwrite: bool,
) -> None:
    if not 0.0 < full_ratio < 1.0:
        raise ValueError(f"full_ratio must be in (0, 1), got {full_ratio}")

    fold_root = processed_root / dataset / fold
    meta_dir = fold_root / "meta"
    manifest_path = meta_dir / "manifest.json"
    split_meta_path = meta_dir / "split_meta.json"
    label_meta_path = meta_dir / "label_meta.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    out_path = meta_dir / f"full_box_split_{method}.json"
    summary_path = meta_dir / f"full_box_split_summary_{method}.json"
    if out_path.exists() and summary_path.exists() and not overwrite:
        print(f"[SKIP] exists: {out_path}")
        return

    manifest = load_json(manifest_path)
    split_meta = load_json(split_meta_path) if split_meta_path.exists() else {}
    label_meta = load_json(label_meta_path) if label_meta_path.exists() else {}

    train_items = [item for item in manifest if item.get("split") == "train"]
    test_items = [item for item in manifest if item.get("split") == "test"]
    if not train_items:
        raise RuntimeError(f"No train items found in {manifest_path}")

    class_ids = parse_class_ids(label_meta)
    is_3d = infer_is_3d(dataset, split_meta)

    if is_3d:
        full_slices, full_cases = choose_full_3d_by_case(
            train_items, full_ratio, seed
        )
    else:
        full_slices = choose_full_2d(train_items, full_ratio, seed)
        full_cases = set()

    records: list[dict[str, Any]] = []
    for item in train_items:
        slice_name = get_slice_name(item)
        records.append(
            {
                "slice_name": slice_name,
                "split": "train",
                "label_mode": "full" if slice_name in full_slices else "box",
                "case_id": get_case_id(item),
                "class_ids": class_ids,
                "box_source": "gt_tight_box",
                "full_ratio": float(full_ratio),
                "seed": int(seed),
                "dataset": dataset,
                "fold": fold,
            }
        )

    if any(record["split"] != "train" for record in records):
        raise RuntimeError("Split leakage detected in full/box records")

    n_full = sum(record["label_mode"] == "full" for record in records)
    n_box = len(records) - n_full

    summary = {
        "dataset": dataset,
        "fold": fold,
        "method": method,
        "is_3d": is_3d,
        "full_ratio": float(full_ratio),
        "seed": int(seed),
        "num_manifest_items": len(manifest),
        "num_train_items": len(train_items),
        "num_test_items_seen_but_not_used": len(test_items),
        "num_records": len(records),
        "num_full": n_full,
        "num_box": n_box,
        "num_full_cases": len(full_cases) if is_3d else None,
        "full_case_ids": sorted(full_cases) if is_3d else None,
        "class_ids": class_ids,
        "box_source": "gt_tight_box",
        "test_leakage": 0,
    }

    save_json(records, out_path)
    save_json(summary, summary_path)

    print(f"[OK] {dataset}/{fold}")
    print(
        f"     train={len(train_items)} full={n_full} box={n_box} "
        f"test_seen_not_used={len(test_items)}"
    )
    if is_3d:
        print(f"     full_cases={len(full_cases)}")
    print(f"     out={out_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a deterministic 10% full / 90% box supervision split."
    )
    parser.add_argument("--processed_root", type=Path, required=True)
    parser.add_argument("--datasets", required=True, help="Comma-separated datasets")
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--method", default=METHOD_DEFAULT)
    parser.add_argument("--full_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    if not datasets:
        raise ValueError("--datasets is empty")

    for dataset in datasets:
        process_dataset(
            processed_root=args.processed_root,
            dataset=dataset,
            fold=args.fold,
            method=args.method,
            full_ratio=args.full_ratio,
            seed=args.seed,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
