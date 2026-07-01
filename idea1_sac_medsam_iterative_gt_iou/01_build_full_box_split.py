from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


METHOD_DEFAULT = "idea1_sac_medsam_iterative_gt_iou"
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

    raise KeyError(
        f"Cannot infer slice_name from keys={list(item.keys())}"
    )


def get_case_id(
    item: dict[str, Any],
    *,
    required: bool = False,
) -> str:
    value = item.get("case_id")

    if value is None or not str(value).strip():
        if required:
            raise KeyError(
                "3D dataset item does not contain a valid case_id: "
                f"slice_name={get_slice_name(item)}"
            )
        return "unknown_case"

    return str(value)


def parse_class_ids(label_meta: Any) -> list[int]:
    if not isinstance(label_meta, dict):
        return [1]

    candidates: list[int] = []

    for key in ("unique_labels", "class_ids"):
        values = label_meta.get(key)
        if not isinstance(values, list):
            continue

        for value in values:
            try:
                class_id = int(value)
            except (TypeError, ValueError):
                continue

            if class_id > 0:
                candidates.append(class_id)

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


def infer_is_3d(
    dataset: str,
    split_meta: dict[str, Any],
) -> bool:
    if dataset.lower() in KNOWN_3D_DATASETS:
        return True

    for key in ("is_3d", "volume_level_split", "has_volume"):
        if key in split_meta:
            return bool(split_meta[key])

    return False


def validate_unique_slice_names(
    train_items: list[dict[str, Any]],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for item in train_items:
        slice_name = get_slice_name(item)

        if slice_name in seen:
            duplicates.add(slice_name)

        seen.add(slice_name)

    if duplicates:
        examples = sorted(duplicates)[:10]
        raise RuntimeError(
            "Duplicate slice_name values were found in the training manifest. "
            "The iterative split uses slice_name as the unique sample key. "
            f"Examples: {examples}"
        )


def build_case_to_items(
    train_items: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    case_to_items: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for item in train_items:
        case_id = get_case_id(item, required=True)
        case_to_items[case_id].append(item)

    return dict(case_to_items)


def choose_random_full_2d(
    train_items: list[dict[str, Any]],
    full_count: int,
    seed: int,
) -> tuple[set[str], list[str]]:
    ordered_names = sorted(
        get_slice_name(item)
        for item in train_items
    )

    random_order = list(ordered_names)
    random.Random(seed).shuffle(random_order)

    if not 1 <= full_count <= len(random_order):
        raise ValueError(
            "For a 2D dataset, full_count must satisfy "
            f"1 <= full_count <= {len(random_order)}, got {full_count}"
        )

    full_slices = set(random_order[:full_count])
    return full_slices, random_order


def choose_random_full_3d(
    train_items: list[dict[str, Any]],
    full_count: int,
    seed: int,
) -> tuple[set[str], set[str], list[str]]:
    case_to_items = build_case_to_items(train_items)

    random_order = sorted(case_to_items)
    random.Random(seed).shuffle(random_order)

    if not 1 <= full_count <= len(random_order):
        raise ValueError(
            "For a 3D dataset, full_count represents the number of cases "
            f"and must satisfy 1 <= full_count <= {len(random_order)}, "
            f"got {full_count}"
        )

    full_cases = set(random_order[:full_count])

    full_slices = {
        get_slice_name(item)
        for case_id in full_cases
        for item in case_to_items[case_id]
    }

    return full_slices, full_cases, random_order


def get_string_list(
    payload: dict[str, Any],
    candidate_keys: tuple[str, ...],
) -> list[str] | None:
    for key in candidate_keys:
        if key not in payload:
            continue

        values = payload[key]

        if not isinstance(values, list):
            raise TypeError(
                f"{key} in selection file must be a list, "
                f"got {type(values).__name__}"
            )

        return [str(value) for value in values]

    return None


def validate_selection_metadata(
    payload: dict[str, Any],
    dataset: str,
    fold: str,
    selection_path: Path,
) -> None:
    selected_dataset = payload.get("dataset")
    selected_fold = payload.get("fold")

    if selected_dataset is not None and str(selected_dataset) != dataset:
        raise ValueError(
            f"Dataset mismatch in {selection_path}: "
            f"expected={dataset}, found={selected_dataset}"
        )

    if selected_fold is not None and str(selected_fold) != fold:
        raise ValueError(
            f"Fold mismatch in {selection_path}: "
            f"expected={fold}, found={selected_fold}"
        )


def load_external_full_selection_2d(
    selection_path: Path,
    train_items: list[dict[str, Any]],
    dataset: str,
    fold: str,
) -> set[str]:
    payload = load_json(selection_path)

    if not isinstance(payload, dict):
        raise TypeError(
            "The external selection file must contain a JSON object. "
            f"Got {type(payload).__name__}: {selection_path}"
        )

    validate_selection_metadata(
        payload,
        dataset,
        fold,
        selection_path,
    )

    full_slice_names = get_string_list(
        payload,
        (
            "cumulative_full_slice_names",
            "full_slice_names",
        ),
    )

    if full_slice_names is None:
        raise KeyError(
            "2D external selection file must contain one of: "
            "'cumulative_full_slice_names' or 'full_slice_names'. "
            f"File: {selection_path}"
        )

    train_slice_names = {
        get_slice_name(item)
        for item in train_items
    }

    full_slices = set(full_slice_names)
    unknown_slices = full_slices - train_slice_names

    if unknown_slices:
        examples = sorted(unknown_slices)[:10]
        raise ValueError(
            "The external selection contains samples that do not belong "
            f"to the current training set. Examples: {examples}"
        )

    if not full_slices:
        raise ValueError(
            f"External Full selection is empty: {selection_path}"
        )

    return full_slices


def load_external_full_selection_3d(
    selection_path: Path,
    train_items: list[dict[str, Any]],
    dataset: str,
    fold: str,
) -> tuple[set[str], set[str]]:
    payload = load_json(selection_path)

    if not isinstance(payload, dict):
        raise TypeError(
            "The external selection file must contain a JSON object. "
            f"Got {type(payload).__name__}: {selection_path}"
        )

    validate_selection_metadata(
        payload,
        dataset,
        fold,
        selection_path,
    )

    case_to_items = build_case_to_items(train_items)
    valid_cases = set(case_to_items)

    full_case_ids = get_string_list(
        payload,
        (
            "cumulative_full_case_ids",
            "full_case_ids",
        ),
    )

    if full_case_ids is not None:
        full_cases = set(full_case_ids)

        unknown_cases = full_cases - valid_cases
        if unknown_cases:
            examples = sorted(unknown_cases)[:10]
            raise ValueError(
                "The external selection contains case IDs that do not "
                f"belong to the current training set. Examples: {examples}"
            )

        if not full_cases:
            raise ValueError(
                f"External Full case selection is empty: {selection_path}"
            )

        full_slices = {
            get_slice_name(item)
            for case_id in full_cases
            for item in case_to_items[case_id]
        }

        return full_slices, full_cases

    # Compatibility fallback:
    # infer selected cases from a cumulative list of slice names, but only
    # accept it when every selected case is completely included.
    full_slice_names = get_string_list(
        payload,
        (
            "cumulative_full_slice_names",
            "full_slice_names",
        ),
    )

    if full_slice_names is None:
        raise KeyError(
            "3D external selection file must contain "
            "'cumulative_full_case_ids'. A complete cumulative slice list "
            "is also accepted for compatibility. "
            f"File: {selection_path}"
        )

    train_slice_to_case: dict[str, str] = {}

    for case_id, items in case_to_items.items():
        for item in items:
            train_slice_to_case[get_slice_name(item)] = case_id

    selected_slices = set(full_slice_names)
    unknown_slices = selected_slices - set(train_slice_to_case)

    if unknown_slices:
        examples = sorted(unknown_slices)[:10]
        raise ValueError(
            "The external selection contains slices that do not belong "
            f"to the current training set. Examples: {examples}"
        )

    full_cases = {
        train_slice_to_case[slice_name]
        for slice_name in selected_slices
    }

    expected_slices = {
        get_slice_name(item)
        for case_id in full_cases
        for item in case_to_items[case_id]
    }

    missing_case_slices = expected_slices - selected_slices

    if missing_case_slices:
        examples = sorted(missing_case_slices)[:10]
        raise ValueError(
            "Partial-case Full selection detected in a 3D dataset. "
            "All slices of a selected case must be included together. "
            f"Missing examples: {examples}"
        )

    return expected_slices, full_cases


def resolve_selection_path(
    selection_file: str | None,
    dataset: str,
    fold: str,
) -> Path | None:
    if selection_file is None:
        return None

    # Allows paths such as:
    # /path/to/{dataset}/{fold}/full_selection_round_1.json
    resolved = selection_file.format(
        dataset=dataset,
        fold=fold,
    )

    return Path(resolved)


def process_dataset(
    processed_root: Path,
    dataset: str,
    fold: str,
    method: str,
    round_id: int,
    sampling_mode: str,
    full_count: int | None,
    selection_path: Path | None,
    seed: int,
    overwrite: bool,
) -> None:
    fold_root = processed_root / dataset / fold
    meta_dir = fold_root / "meta"

    manifest_path = meta_dir / "manifest.json"
    split_meta_path = meta_dir / "split_meta.json"
    label_meta_path = meta_dir / "label_meta.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"manifest.json not found: {manifest_path}"
        )

    out_path = meta_dir / f"full_box_split_{method}.json"
    summary_path = meta_dir / f"full_box_split_summary_{method}.json"
    selection_state_path = meta_dir / f"full_selection_{method}.json"

    output_paths = [
        out_path,
        summary_path,
        selection_state_path,
    ]
    existing_paths = [
        path
        for path in output_paths
        if path.exists()
    ]

    if len(existing_paths) == len(output_paths) and not overwrite:
        print(f"[SKIP] outputs already exist for {dataset}/{fold}: {method}")
        return

    if existing_paths and not overwrite:
        raise FileExistsError(
            "Partial outputs already exist. Use --overwrite to regenerate "
            f"them consistently. Existing: {existing_paths}"
        )

    manifest = load_json(manifest_path)

    if not isinstance(manifest, list):
        raise TypeError(
            f"manifest.json must contain a list: {manifest_path}"
        )

    split_meta = (
        load_json(split_meta_path)
        if split_meta_path.exists()
        else {}
    )
    label_meta = (
        load_json(label_meta_path)
        if label_meta_path.exists()
        else {}
    )

    train_items = [
        item
        for item in manifest
        if item.get("split") == "train"
    ]
    test_items = [
        item
        for item in manifest
        if item.get("split") == "test"
    ]

    if not train_items:
        raise RuntimeError(
            f"No train items found in {manifest_path}"
        )

    validate_unique_slice_names(train_items)

    class_ids = parse_class_ids(label_meta)
    is_3d = infer_is_3d(dataset, split_meta)

    full_cases: set[str] = set()
    random_order_slice_names: list[str] | None = None
    random_order_case_ids: list[str] | None = None

    if sampling_mode == "random":
        if full_count is None:
            raise ValueError(
                "--full_count is required when --sampling_mode=random"
            )

        if selection_path is not None:
            raise ValueError(
                "--selection_file must not be provided when "
                "--sampling_mode=random"
            )

        if is_3d:
            (
                full_slices,
                full_cases,
                random_order_case_ids,
            ) = choose_random_full_3d(
                train_items=train_items,
                full_count=full_count,
                seed=seed,
            )
        else:
            (
                full_slices,
                random_order_slice_names,
            ) = choose_random_full_2d(
                train_items=train_items,
                full_count=full_count,
                seed=seed,
            )

        selection_source = f"random_seed_{seed}"

    elif sampling_mode == "external":
        if selection_path is None:
            raise ValueError(
                "--selection_file is required when "
                "--sampling_mode=external"
            )

        if not selection_path.exists():
            raise FileNotFoundError(
                f"Selection file not found: {selection_path}"
            )

        if is_3d:
            (
                full_slices,
                full_cases,
            ) = load_external_full_selection_3d(
                selection_path=selection_path,
                train_items=train_items,
                dataset=dataset,
                fold=fold,
            )
        else:
            full_slices = load_external_full_selection_2d(
                selection_path=selection_path,
                train_items=train_items,
                dataset=dataset,
                fold=fold,
            )

        selection_source = str(selection_path)

    else:
        raise ValueError(
            f"Unsupported sampling_mode: {sampling_mode}"
        )

    full_unit_count = (
        len(full_cases)
        if is_3d
        else len(full_slices)
    )

    # In external mode, full_count is optional and acts as an expected
    # cumulative count check.
    if full_count is not None and full_unit_count != full_count:
        unit_name = "cases" if is_3d else "images"
        raise ValueError(
            f"Full count mismatch: expected {full_count} {unit_name}, "
            f"but selection contains {full_unit_count}"
        )

    records: list[dict[str, Any]] = []

    for item in train_items:
        slice_name = get_slice_name(item)

        records.append(
            {
                "slice_name": slice_name,
                "split": "train",
                "label_mode": (
                    "full"
                    if slice_name in full_slices
                    else "box"
                ),
                "case_id": get_case_id(
                    item,
                    required=is_3d,
                ),
                "class_ids": class_ids,
                "box_source": "gt_tight_box",
                "round_id": int(round_id),
                "sampling_mode": sampling_mode,
                "selection_source": selection_source,
                "seed": int(seed),
                "dataset": dataset,
                "fold": fold,
            }
        )

    if any(record["split"] != "train" for record in records):
        raise RuntimeError(
            "Split leakage detected in full/box records"
        )

    n_full = sum(
        record["label_mode"] == "full"
        for record in records
    )
    n_box = len(records) - n_full
    actual_full_ratio = n_full / len(records)

    # Keep the full_ratio field for compatibility with existing downstream
    # scripts, but it now records the actual ratio instead of controlling
    # sample selection.
    for record in records:
        record["full_ratio"] = float(actual_full_ratio)

    summary = {
        "schema_version": 2,
        "dataset": dataset,
        "fold": fold,
        "method": method,
        "round_id": int(round_id),
        "sampling_mode": sampling_mode,
        "sampling_unit": "case" if is_3d else "image",
        "selection_source": selection_source,
        "is_3d": is_3d,
        "requested_full_count": full_count,
        "actual_full_unit_count": full_unit_count,
        "actual_full_ratio": float(actual_full_ratio),
        "seed": int(seed),
        "num_manifest_items": len(manifest),
        "num_train_items": len(train_items),
        "num_test_items_seen_but_not_used": len(test_items),
        "num_records": len(records),
        "num_full": n_full,
        "num_box": n_box,
        "num_full_cases": (
            len(full_cases)
            if is_3d
            else None
        ),
        "full_case_ids": (
            sorted(full_cases)
            if is_3d
            else None
        ),
        "class_ids": class_ids,
        "box_source": "gt_tight_box",
        "test_leakage": 0,
    }

    selection_state = {
        "schema_version": 1,
        "dataset": dataset,
        "fold": fold,
        "method": method,
        "round_id": int(round_id),
        "sampling_mode": sampling_mode,
        "sampling_unit": "case" if is_3d else "image",
        "selection_source": selection_source,
        "seed": int(seed),
        "requested_full_count": full_count,
        "cumulative_full_count": full_unit_count,
        "cumulative_full_slice_names": sorted(full_slices),
        "cumulative_full_case_ids": (
            sorted(full_cases)
            if is_3d
            else None
        ),
        "random_order_slice_names": random_order_slice_names,
        "random_order_case_ids": random_order_case_ids,
    }

    save_json(records, out_path)
    save_json(summary, summary_path)
    save_json(selection_state, selection_state_path)

    print(f"[OK] {dataset}/{fold}")
    print(
        f"     round={round_id} mode={sampling_mode} "
        f"is_3d={is_3d}"
    )
    print(
        f"     train={len(train_items)} full={n_full} box={n_box} "
        f"actual_full_ratio={actual_full_ratio:.6f}"
    )

    if is_3d:
        print(
            f"     full_cases={len(full_cases)} "
            f"case_ids={sorted(full_cases)}"
        )

    print(f"     split={out_path}")
    print(f"     summary={summary_path}")
    print(f"     selection={selection_state_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an iterative Full/Box supervision split. "
            "Round 0 uses a deterministic fixed-count random selection. "
            "Later rounds read a cumulative Full selection JSON file."
        )
    )

    parser.add_argument(
        "--processed_root",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated dataset names",
    )
    parser.add_argument(
        "--fold",
        default="fold_0",
    )
    parser.add_argument(
        "--method",
        default=METHOD_DEFAULT,
    )
    parser.add_argument(
        "--round_id",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--sampling_mode",
        choices=("random", "external"),
        default="random",
        help=(
            "random: deterministic fixed-count selection, normally Round 0; "
            "external: read cumulative Full samples from --selection_file"
        ),
    )
    parser.add_argument(
        "--full_count",
        type=int,
        default=None,
        help=(
            "2D: number of Full images. "
            "3D: number of Full cases. "
            "Required for random mode. In external mode it is optional "
            "and is used to validate the cumulative selection count."
        ),
    )
    parser.add_argument(
        "--selection_file",
        default=None,
        help=(
            "Cumulative Full selection JSON used in external mode. "
            "The path may contain {dataset} and {fold} placeholders."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.round_id < 0:
        raise ValueError(
            f"--round_id must be >= 0, got {args.round_id}"
        )

    if args.full_count is not None and args.full_count <= 0:
        raise ValueError(
            f"--full_count must be positive, got {args.full_count}"
        )

    if args.sampling_mode == "random":
        if args.full_count is None:
            raise ValueError(
                "--full_count is required in random mode"
            )

        if args.selection_file is not None:
            raise ValueError(
                "--selection_file cannot be used in random mode"
            )

    if args.sampling_mode == "external":
        if args.selection_file is None:
            raise ValueError(
                "--selection_file is required in external mode"
            )

    datasets = [
        name.strip()
        for name in args.datasets.split(",")
        if name.strip()
    ]

    if not datasets:
        raise ValueError("--datasets is empty")

    for dataset in datasets:
        selection_path = resolve_selection_path(
            selection_file=args.selection_file,
            dataset=dataset,
            fold=args.fold,
        )

        process_dataset(
            processed_root=args.processed_root,
            dataset=dataset,
            fold=args.fold,
            method=args.method,
            round_id=args.round_id,
            sampling_mode=args.sampling_mode,
            full_count=args.full_count,
            selection_path=selection_path,
            seed=args.seed,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()