import argparse
import json
import os
import random
from collections import defaultdict


METHOD = "idea1_sac_medsam_final"


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
    # 兜底：从路径取文件名
    for key in ["student_img", "teacher_img", "student_gt", "teacher_gt"]:
        if key in item:
            return os.path.basename(item[key])
    raise KeyError(f"Cannot infer slice_name from manifest item keys={list(item.keys())}")


def get_case_id(item):
    return str(item.get("case_id", "unknown_case"))


def get_class_ids_from_label_meta(label_meta):
    # 兼容不同 label_meta 格式；没有就默认二分类前景 1
    if not isinstance(label_meta, dict):
        return [1]

    if "class_ids" in label_meta and isinstance(label_meta["class_ids"], list):
        ids = [int(x) for x in label_meta["class_ids"] if int(x) > 0]
        return ids or [1]

    if "labels" in label_meta:
        labels = label_meta["labels"]
        if isinstance(labels, dict):
            ids = []
            for k in labels.keys():
                try:
                    kk = int(k)
                    if kk > 0:
                        ids.append(kk)
                except Exception:
                    pass
            return sorted(ids) or [1]

    if "num_classes" in label_meta:
        n = int(label_meta["num_classes"])
        if n <= 1:
            return [1]
        return list(range(1, n))

    return [1]


def choose_full_2d(train_items, full_ratio, seed):
    rng = random.Random(seed)
    items = train_items[:]
    rng.shuffle(items)
    n_full = max(1, int(round(len(items) * full_ratio)))
    return set(get_slice_name(x) for x in items[:n_full])


def choose_full_3d_by_case(train_items, full_ratio, seed):
    rng = random.Random(seed)
    case_to_items = defaultdict(list)
    for item in train_items:
        case_to_items[get_case_id(item)].append(item)

    cases = sorted(case_to_items.keys())
    rng.shuffle(cases)

    n_full_cases = max(1, int(round(len(cases) * full_ratio)))
    full_cases = set(cases[:n_full_cases])

    full_slices = set()
    for c in full_cases:
        for item in case_to_items[c]:
            full_slices.add(get_slice_name(item))
    return full_slices, full_cases


def infer_is_3d(dataset_name, split_meta):
    ds = dataset_name.lower()
    if ds in {"btcv", "synapse", "acdc", "prostate158"}:
        return True
    # 兜底：split_meta 里如果写了维度，也尊重
    for k in ["is_3d", "volume_level_split", "has_volume"]:
        if k in split_meta:
            return bool(split_meta[k])
    return False


def process_dataset(processed_root, dataset, fold, full_ratio, seed, overwrite):
    fold_root = os.path.join(processed_root, dataset, fold)
    meta_dir = os.path.join(fold_root, "meta")

    manifest_path = os.path.join(meta_dir, "manifest.json")
    split_meta_path = os.path.join(meta_dir, "split_meta.json")
    label_meta_path = os.path.join(meta_dir, "label_meta.json")

    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"manifest.json not found: {manifest_path}")

    out_path = os.path.join(meta_dir, f"full_box_split_{METHOD}.json")
    if os.path.exists(out_path) and not overwrite:
        print(f"[SKIP] exists: {out_path}")
        return

    manifest = load_json(manifest_path)
    split_meta = load_json(split_meta_path) if os.path.exists(split_meta_path) else {}
    label_meta = load_json(label_meta_path) if os.path.exists(label_meta_path) else {}

    class_ids = get_class_ids_from_label_meta(label_meta)

    train_items = [x for x in manifest if x.get("split") == "train"]
    test_items = [x for x in manifest if x.get("split") == "test"]

    if len(train_items) == 0:
        raise RuntimeError(f"No train items found in manifest: {manifest_path}")

    is_3d = infer_is_3d(dataset, split_meta)

    if is_3d:
        full_slices, full_cases = choose_full_3d_by_case(train_items, full_ratio, seed)
    else:
        full_slices = choose_full_2d(train_items, full_ratio, seed)
        full_cases = set()

    records = []
    for item in train_items:
        slice_name = get_slice_name(item)
        case_id = get_case_id(item)
        label_mode = "full" if slice_name in full_slices else "box"

        records.append({
            "slice_name": slice_name,
            "split": "train",
            "label_mode": label_mode,
            "case_id": case_id,
            "class_ids": class_ids,
            "box_source": "gt_tight_box",
            "full_ratio": float(full_ratio),
            "seed": int(seed),
            "dataset": dataset,
            "fold": fold,
        })

    # 安全审计：禁止 test 泄漏
    if any(r["split"] != "train" for r in records):
        raise RuntimeError("Split leakage detected: non-train record in full_box_split")

    n_full = sum(r["label_mode"] == "full" for r in records)
    n_box = sum(r["label_mode"] == "box" for r in records)

    summary = {
        "dataset": dataset,
        "fold": fold,
        "method": METHOD,
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
        "class_ids": class_ids,
        "test_leakage": 0,
    }

    save_json(records, out_path)
    save_json(summary, os.path.join(meta_dir, f"full_box_split_summary_{METHOD}.json"))

    print(f"[OK] {dataset}/{fold}")
    print(f"     train={len(train_items)} full={n_full} box={n_box} test_seen_not_used={len(test_items)}")
    print(f"     out={out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_root", required=True)
    parser.add_argument("--datasets", required=True, help="comma-separated datasets")
    parser.add_argument("--fold", default="fold_0")
    parser.add_argument("--full_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    for ds in datasets:
        process_dataset(
            processed_root=args.processed_root,
            dataset=ds,
            fold=args.fold,
            full_ratio=args.full_ratio,
            seed=args.seed,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
