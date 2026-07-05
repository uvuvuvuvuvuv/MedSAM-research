from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import py_compile
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Import from shared modules
# ---------------------------------------------------------------------------
try:
    from .pipeline_common import (
        METHOD_DEFAULT,
        build_fold_paths,
        is_3d_dataset,
        save_json_atomic,
    )
except ImportError:
    from pipeline_common import (  # type: ignore[no-redef]
        METHOD_DEFAULT,
        build_fold_paths,
        is_3d_dataset,
        save_json_atomic,
    )


DEFAULT_PROCESSED_ROOT = Path(
    "/storage/baiyuting/data/out_data_idea1/"
    "MedSAM-main/data/processed"
)
DEFAULT_BASE_CHECKPOINT = Path(
    "/storage/baiyuting/data/MedSAM-main/"
    "work_dir/MedSAM/medsam_vit_b.pth"
)
DEFAULT_MEDSAM_FT_ROOT = Path(
    "/storage/baiyuting/data/out_data_idea1/"
    "MedSAM-main/work_dir/MedSAM_ft"
)
DEFAULT_RUNNER_ROOT = Path(
    "/storage/baiyuting/data/out_data_idea1/"
    "MedSAM-main/work_dir/iterative_sampling_runs"
)

PRESET_SCHEDULES: dict[str, list[int]] = {
    "formal_2d": [5, 10, 15, 20],
    "formal_3d": [1, 2, 3, 4, 5],
}

def _formal_schedule(is_3d: bool) -> list[int]:
    """Full formal schedule independent of --preset or --end_round."""
    return list(PRESET_SCHEDULES["formal_3d" if is_3d else "formal_2d"])


def is_formal_final_round(round_id: int, is_3d: bool) -> bool:
    """True only when round_id is the last round of the full formal schedule."""
    return round_id == len(_formal_schedule(is_3d)) - 1


SCRIPT_NAMES = {
    "split": "01_build_full_box_split.py",
    "template": "02_build_support_template.py",
    "train": "03_train_medsam_sac.py",
    "pseudo": "04_generate_pseudo_sac.py",
    "select": "05_select_hard_by_gt_iou.py",
}

STAGE_ORDER = (
    "split",
    "template",
    "train",
    "pseudo",
    "select",
)

STOP_AFTER_CHOICES = (
    "split",
    "template",
    "train",
    "pseudo",
    "select",
    "round",
    "all",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def append_jsonl(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
        f.write("\n")


def parse_csv_strings(value: str) -> list[str]:
    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def parse_schedule(value: str) -> list[int]:
    values = parse_csv_strings(value)
    if not values:
        raise ValueError("--schedule is empty")

    schedule: list[int] = []
    for raw in values:
        try:
            number = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"Invalid schedule value: {raw!r}"
            ) from exc

        if number <= 0:
            raise ValueError(
                f"Schedule values must be positive, got {number}"
            )
        schedule.append(number)

    validate_schedule(schedule)
    return schedule


def validate_schedule(schedule: list[int]) -> None:
    if not schedule:
        raise ValueError("The Full schedule is empty")

    for previous, current in zip(schedule, schedule[1:]):
        if current <= previous:
            raise ValueError(
                "Full schedule must be strictly increasing: "
                f"{schedule}"
            )


def sanitize_name(value: str) -> str:
    allowed = set(
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_-."
    )
    result = "".join(
        char if char in allowed else "_"
        for char in value.strip()
    )
    result = result.strip("._-")
    return result or "run"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def count_npy_files(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(
        1
        for path in directory.iterdir()
        if path.is_file() and path.suffix == ".npy"
    )


def format_command(command: list[str]) -> str:
    return shlex.join(command)


def infer_is_3d(
    processed_root: Path,
    dataset: str,
    fold: str,
) -> bool:
    if is_3d_dataset(dataset):
        return True

    split_meta_path = (
        processed_root
        / dataset
        / fold
        / "meta"
        / "split_meta.json"
    )

    if split_meta_path.exists():
        split_meta = load_json(split_meta_path)
        if isinstance(split_meta, dict):
            for key in (
                "is_3d",
                "volume_level_split",
                "has_volume",
            ):
                if key in split_meta:
                    return bool(split_meta[key])

    return False


def get_slice_name(record: dict[str, Any]) -> str:
    if record.get("slice_name"):
        return str(record["slice_name"])

    for key in (
        "teacher_img",
        "student_img",
        "teacher_gt",
        "student_gt",
    ):
        value = record.get(key)
        if value:
            return Path(str(value)).name

    raise KeyError(
        "Cannot infer slice_name from record keys: "
        f"{list(record.keys())}"
    )


def get_case_id(record: dict[str, Any]) -> str:
    value = record.get("case_id")
    if value is None or str(value).strip() == "":
        raise KeyError(
            "A 3D Full/Box record is missing case_id: "
            f"{get_slice_name(record)}"
        )
    return str(value)


def build_round_tags(
    schedule: list[int],
    is_3d: bool,
) -> list[str]:
    unit = "case" if is_3d else "full"
    return [
        f"r{idx:02d}_{unit}{count}"
        for idx, count in enumerate(schedule)
    ]


def determine_schedule(
    args: argparse.Namespace,
) -> list[int]:
    if args.schedule:
        return parse_schedule(args.schedule)

    if args.preset == "custom":
        raise ValueError(
            "--preset custom requires --schedule, "
            "for example --schedule 5,7"
        )

    schedule = list(PRESET_SCHEDULES[args.preset])
    validate_schedule(schedule)
    return schedule


def default_run_name(
    datasets: list[str],
    schedule: list[int],
    method: str,
) -> str:
    dataset_token = "-".join(datasets)
    schedule_token = "-".join(str(x) for x in schedule)
    return sanitize_name(
        f"{dataset_token}_{schedule_token}_{method}"
    )


def get_script_paths(
    code_root: Path,
) -> dict[str, Path]:
    return {
        stage: code_root / filename
        for stage, filename in SCRIPT_NAMES.items()
    }


def compile_scripts(
    script_paths: dict[str, Path],
) -> None:
    for stage, path in script_paths.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {stage} script: {path}"
            )

        py_compile.compile(
            str(path),
            doraise=True,
        )


def validate_dataset_roots(
    args: argparse.Namespace,
    datasets: list[str],
    dimensions: dict[str, bool],
) -> None:
    if not args.base_checkpoint.is_file():
        raise FileNotFoundError(
            "Base MedSAM checkpoint does not exist: "
            f"{args.base_checkpoint}"
        )

    for dataset in datasets:
        fold_root = (
            args.processed_root
            / dataset
            / args.fold
        )

        manifest_path = (
            fold_root
            / "meta"
            / "manifest.json"
        )

        prompts_path = (
            fold_root
            / "prompts"
            / "prompts_train.json"
        )

        geometry_path = (
            fold_root
            / "meta"
            / "geometry_meta.json"
        )

        for required in (
            manifest_path,
            prompts_path,
            geometry_path,
        ):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Required dataset input is missing: {required}"
                )

        if dimensions[dataset]:
            manifest = load_json(manifest_path)
            train_records = [
                record
                for record in manifest
                if isinstance(record, dict)
                and record.get("split") == "train"
            ]
            missing_case_ids = [
                get_slice_name(record)
                for record in train_records
                if not record.get("case_id")
            ]
            if missing_case_ids:
                raise RuntimeError(
                    f"{dataset}: 3D train records without case_id; "
                    f"examples={missing_case_ids[:10]}"
                )


def validate_dimension_group(
    args: argparse.Namespace,
    datasets: list[str],
    dimensions: dict[str, bool],
) -> None:
    unique_dimensions = {
        dimensions[dataset]
        for dataset in datasets
    }

    if len(unique_dimensions) != 1:
        raise ValueError(
            "Do not mix 2D and 3D datasets in one runner call. "
            f"Resolved dimensions={dimensions}"
        )

    is_3d = next(iter(unique_dimensions))

    if args.preset == "formal_2d" and is_3d:
        raise ValueError(
            f"Preset {args.preset!r} is for 2D datasets, "
            f"but datasets are 3D: {datasets}"
        )

    if args.preset == "formal_3d" and not is_3d:
        raise ValueError(
            "Preset 'formal_3d' requires 3D datasets, "
            f"but datasets are 2D: {datasets}"
        )


def round_output_paths(
    args: argparse.Namespace,
    dataset: str,
    method: str,
    round_tag: str,
) -> dict[str, Path]:
    p = build_fold_paths(
        processed_root=args.processed_root,
        dataset=dataset,
        fold=args.fold,
        method=method,
        round_tag=round_tag,
        medsam_ft_root=args.medsam_ft_root,
    )
    run_id = p["run_id"]
    meta_dir = p["meta_dir"]
    train_root = p["train_root"]
    fold_root = p["fold_root"]

    return {
        "fold_root": fold_root,
        "meta_dir": meta_dir,
        "run_id": run_id,
        "split": p["split_path"],
        "split_summary": p["split_summary_path"],
        "template": p["support_path"],
        "template_stats": p["support_stats_path"],
        "feature_audit": (
            meta_dir
            / f"feature_extraction_audit_{run_id}.json"
        ),
        "train_root": train_root,
        "train_last": p["train_last_path"],
        "train_ema": p["train_ema_path"],
        "train_log": p["train_log_path"],
        "train_summary": p["train_summary_path"],
        "pseudo_teacher": p["pseudo_teacher_dir"],
        "pseudo_student": p["pseudo_student_dir"],
        "pseudo_audit": (
            meta_dir
            / f"pseudo_generation_output_audit_{run_id}.json"
        ),
        "pseudo_config": (
            meta_dir
            / f"pseudo_generation_config_{run_id}.json"
        ),
        "pseudo_stats": (
            meta_dir
            / f"pseudo_quality_stats_{run_id}.csv"
        ),
        "round_summary": p["round_summary_path"],
        "hard_selection": p["hard_selection_path"],
        "hard_summary": p["hard_summary_path"],
        "hard_ranking_2d": (
            meta_dir
            / f"hard_ranking_{run_id}.csv"
        ),
        "hard_slice_ranking_3d": (
            meta_dir
            / f"hard_slice_ranking_{run_id}.csv"
        ),
        "hard_case_ranking_3d": (
            meta_dir
            / f"hard_case_ranking_{run_id}.csv"
        ),
    }


def validate_split_stage(
    args: argparse.Namespace,
    dataset: str,
    method: str,
    round_tag: str,
    expected_full_units: int,
    is_3d: bool,
) -> tuple[bool, str]:
    paths = round_output_paths(
        args,
        dataset,
        method,
        round_tag,
    )

    required = (
        paths["split"],
        paths["split_summary"],
    )

    missing = [
        str(path)
        for path in required
        if not path.is_file()
    ]
    if missing:
        return False, (
            "missing split outputs: "
            + ", ".join(missing)
        )

    try:
        records = load_json(paths["split"])
    except Exception as exc:
        return False, f"cannot read split JSON: {exc}"

    if not isinstance(records, list):
        return False, "split JSON is not a list"

    full_records = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("label_mode") == "full"
    ]
    box_records = [
        record
        for record in records
        if isinstance(record, dict)
        and record.get("label_mode") == "box"
    ]

    if not full_records:
        return False, "split has no Full records"
    if not box_records:
        return False, "split has no Box records"

    try:
        if is_3d:
            actual_units = len(
                {
                    get_case_id(record)
                    for record in full_records
                }
            )
        else:
            actual_units = len(full_records)
    except Exception as exc:
        return False, str(exc)

    if actual_units != expected_full_units:
        return False, (
            "Full-unit count mismatch: "
            f"expected={expected_full_units}, "
            f"actual={actual_units}"
        )

    try:
        summary = load_json(paths["split_summary"])
    except Exception as exc:
        return False, f"cannot read split summary: {exc}"

    if isinstance(summary, dict):
        test_leakage = summary.get("test_leakage")
        if test_leakage not in (None, 0, False):
            return False, (
                f"split summary reports test leakage: "
                f"{test_leakage}"
            )

    return True, (
        f"Full units={actual_units}, "
        f"Full slices={len(full_records)}, "
        f"Box slices={len(box_records)}"
    )


def validate_npz_template(
    path: Path,
) -> tuple[bool, str]:
    if not path.is_file():
        return False, f"missing template: {path}"

    try:
        with zipfile.ZipFile(path, "r") as archive:
            names = set(archive.namelist())
    except Exception as exc:
        return False, f"invalid NPZ archive: {exc}"

    if "class_ids.npy" not in names:
        return False, "template has no class_ids.npy"

    has_fg = any(
        name.startswith("bank_fg_c")
        and name.endswith(".npy")
        for name in names
    )
    has_bg = any(
        name.startswith("bank_bg_c")
        and name.endswith(".npy")
        for name in names
    )
    has_shape = any(
        name.startswith("shape_templates_c")
        and name.endswith(".npy")
        for name in names
    )

    if not (has_fg and has_bg and has_shape):
        return False, (
            "template is missing foreground/background "
            "bank features or shape_templates"
        )

    return True, (
        f"NPZ entries={len(names)}"
    )


def validate_template_stage(
    args: argparse.Namespace,
    dataset: str,
    method: str,
    round_tag: str,
) -> tuple[bool, str]:
    paths = round_output_paths(
        args,
        dataset,
        method,
        round_tag,
    )

    if not paths["template"].is_file():
        return False, f"missing template: {paths['template']}"

    ok, reason = validate_npz_template(paths["template"])
    if not ok:
        return False, reason

    if not paths["template_stats"].is_file():
        return False, f"missing template_stats: {paths['template_stats']}"

    try:
        stats = load_json(paths["template_stats"])
    except Exception as exc:
        return False, f"cannot read template stats: {exc}"

    if not isinstance(stats, dict):
        return False, "template stats is not an object"

    if stats.get("method") != method:
        return False, (
            "template stats method mismatch: "
            f"{stats.get('method')!r}"
        )

    if stats.get("dataset") != dataset:
        return False, (
            "template stats dataset mismatch: "
            f"{stats.get('dataset')!r}"
        )

    processed_count = int(stats.get("num_full_records", 0))
    if processed_count <= 0:
        return False, (
            f"template stats: num_full_records={processed_count} "
            f"is not positive"
        )

    skipped = int(stats.get("skipped_or_invalid_count", -1))
    if skipped != 0:
        return False, (
            f"template stats reports skipped/ invalid records: "
            f"{skipped}"
        )

    return True, (
        f"{reason}; processed Full slices={processed_count}"
    )


def validate_train_stage(
    args: argparse.Namespace,
    dataset: str,
    method: str,
    round_tag: str,
) -> tuple[bool, str]:
    paths = round_output_paths(
        args,
        dataset,
        method,
        round_tag,
    )

    for key in (
        "train_last",
        "train_ema",
        "train_log",
        "train_summary",
    ):
        path = paths[key]
        if not path.is_file():
            return False, f"missing {key}: {path}"
        if path.stat().st_size <= 0:
            return False, f"empty {key}: {path}"

    try:
        summary = load_json(paths["train_summary"])
    except Exception as exc:
        return False, f"cannot read train summary: {exc}"

    if not isinstance(summary, dict):
        return False, "train summary is not an object"

    if summary.get("method") != method:
        return False, (
            "train summary method mismatch: "
            f"{summary.get('method')!r}"
        )

    global_step = int(
        summary.get("global_step", 0)
    )
    if global_step <= 0:
        return False, (
            f"global_step is not positive: {global_step}"
        )

    if summary.get("trainable") != "mask_decoder_only":
        return False, (
            "trainable module mismatch: "
            f"{summary.get('trainable')!r}"
        )

    return True, (
        f"global_step={global_step}, "
        f"EMA={paths['train_ema']}"
    )


def validate_pseudo_stage(
    args: argparse.Namespace,
    dataset: str,
    method: str,
    round_tag: str,
) -> tuple[bool, str]:
    paths = round_output_paths(
        args,
        dataset,
        method,
        round_tag,
    )

    for key in (
        "pseudo_audit",
        "pseudo_config",
        "pseudo_stats",
    ):
        if not paths[key].is_file():
            return False, f"missing {key}: {paths[key]}"

    try:
        audit = load_json(paths["pseudo_audit"])
    except Exception as exc:
        return False, f"cannot read pseudo audit: {exc}"

    if not isinstance(audit, dict):
        return False, "pseudo audit is not an object"

    if audit.get("method") != method:
        return False, (
            "pseudo audit method mismatch: "
            f"{audit.get('method')!r}"
        )

    if audit.get("output_complete") is not True:
        return False, "output_complete is not true"

    expected_count = int(
        audit.get("expected_count", -1)
    )
    processed_count = int(
        audit.get("processed_count", -1)
    )
    teacher_count = count_npy_files(
        paths["pseudo_teacher"]
    )
    student_count = count_npy_files(
        paths["pseudo_student"]
    )

    if expected_count <= 0:
        return False, (
            f"invalid expected_count={expected_count}"
        )

    if processed_count != expected_count:
        return False, (
            "pseudo processed-count mismatch: "
            f"expected={expected_count}, "
            f"processed={processed_count}"
        )

    if teacher_count != expected_count:
        return False, (
            "Teacher pseudo count mismatch: "
            f"expected={expected_count}, "
            f"actual={teacher_count}"
        )

    if student_count != expected_count:
        return False, (
            "Student pseudo count mismatch: "
            f"expected={expected_count}, "
            f"actual={student_count}"
        )

    return True, (
        f"expected={expected_count}, "
        f"teacher={teacher_count}, "
        f"student={student_count}"
    )


def validate_selection_stage(
    args: argparse.Namespace,
    dataset: str,
    method: str,
    round_tag: str,
    previous_full_units: int,
    next_full_units: int,
    is_3d: bool,
) -> tuple[bool, str]:
    paths = round_output_paths(
        args,
        dataset,
        method,
        round_tag,
    )

    required_keys = [
        "hard_selection",
        "hard_summary",
    ]

    if is_3d:
        required_keys.extend(
            [
                "hard_slice_ranking_3d",
                "hard_case_ranking_3d",
            ]
        )
    else:
        required_keys.append(
            "hard_ranking_2d"
        )

    for key in required_keys:
        if not paths[key].is_file():
            return False, f"missing {key}: {paths[key]}"

    try:
        summary = load_json(paths["hard_summary"])
        hard = load_json(paths["hard_selection"])
    except Exception as exc:
        return False, f"cannot read selection outputs: {exc}"

    if not isinstance(summary, dict):
        return False, "selection summary is not an object"
    if not isinstance(hard, dict):
        return False, "hard selection is not an object"

    if summary.get("selection_complete") is not True:
        return False, "selection_complete is not true"

    actual_previous = int(
        summary.get(
            "previous_full_unit_count",
            -1,
        )
    )
    actual_next = int(
        summary.get(
            "cumulative_full_unit_count",
            -1,
        )
    )

    if actual_previous != previous_full_units:
        return False, (
            "previous Full-unit count mismatch: "
            f"expected={previous_full_units}, "
            f"actual={actual_previous}"
        )

    if actual_next != next_full_units:
        return False, (
            "next Full-unit count mismatch: "
            f"expected={next_full_units}, "
            f"actual={actual_next}"
        )

    if int(summary.get("test_samples_used", -1)) != 0:
        return False, (
            "selection summary reports test_samples_used != 0"
        )

    if int(hard.get("cumulative_full_count", -1)) != next_full_units:
        return False, (
            "hard selection cumulative count mismatch"
        )

    return True, (
        f"Full units {actual_previous} -> {actual_next}"
    )


def canonical_selection_payload(
    hard_payload: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "schema_version",
        "dataset",
        "fold",
        "source_method",
        "source_round_id",
        "round_id",
        "selection_strategy",
        "sampling_unit",
        "score_space",
        "score_name",
        "class_ids",
        "previous_full_count",
        "new_selected_count",
        "cumulative_full_count",
        "new_selected_slice_names",
        "cumulative_full_slice_names",
        "new_selected_case_ids",
        "cumulative_full_case_ids",
    )

    payload = {
        key: hard_payload.get(key)
        for key in keys
    }

    required = (
        "dataset",
        "fold",
        "source_method",
        "round_id",
        "previous_full_count",
        "new_selected_count",
        "cumulative_full_count",
        "cumulative_full_slice_names",
    )

    missing = [
        key
        for key in required
        if payload.get(key) is None
    ]
    if missing:
        raise RuntimeError(
            "Cannot reconstruct Full selection snapshot; "
            f"missing keys={missing}"
        )

    return payload


def selection_snapshot_path(
    run_dir: Path,
    dataset: str,
    source_round_id: int,
    source_method: str,
) -> Path:
    return (
        run_dir
        / "selection_snapshots"
        / dataset
        / (
            f"after_round_{source_round_id:02d}_"
            f"{source_method}.json"
        )
    )


def create_or_validate_selection_snapshot(
    args: argparse.Namespace,
    run_dir: Path,
    dataset: str,
    method: str,
    round_tag: str,
    round_id: int,
    next_round_id: int,
    expected_next_units: int,
) -> Path:
    snapshot = selection_snapshot_path(
        run_dir,
        dataset,
        round_id,
        method,
    )

    if snapshot.is_file():
        payload = load_json(snapshot)
        if (
            isinstance(payload, dict)
            and payload.get("source_method") == method
            and int(payload.get("round_id", -1))
            == next_round_id
            and int(
                payload.get(
                    "cumulative_full_count",
                    -1,
                )
            )
            == expected_next_units
        ):
            return snapshot

    paths = round_output_paths(
        args,
        dataset,
        method,
        round_tag,
    )

    hard_payload = load_json(
        paths["hard_selection"]
    )
    if not isinstance(hard_payload, dict):
        raise TypeError(
            f"Hard selection is not an object: "
            f"{paths['hard_selection']}"
        )

    payload = canonical_selection_payload(
        hard_payload
    )

    if payload.get("source_method") != method:
        raise RuntimeError(
            "Hard-selection source method mismatch: "
            f"expected={method}, "
            f"actual={payload.get('source_method')}"
        )

    if int(payload["round_id"]) != next_round_id:
        raise RuntimeError(
            "Hard-selection next round mismatch: "
            f"expected={next_round_id}, "
            f"actual={payload['round_id']}"
        )

    if (
        int(payload["cumulative_full_count"])
        != expected_next_units
    ):
        raise RuntimeError(
            "Hard-selection cumulative count mismatch: "
            f"expected={expected_next_units}, "
            f"actual={payload['cumulative_full_count']}"
        )

    save_json_atomic(payload, snapshot)
    return snapshot


def build_split_command(
    args: argparse.Namespace,
    scripts: dict[str, Path],
    dataset: str,
    method: str,
    round_tag: str,
    round_id: int,
    full_count: int,
    selection_file: Path | None,
    stage_overwrite: bool,
) -> list[str]:
    command = [
        str(args.python),
        str(scripts["split"]),
        "--processed_root",
        str(args.processed_root),
        "--datasets",
        dataset,
        "--fold",
        args.fold,
        "--method",
        method,
        "--round_tag",
        round_tag,
        "--round_id",
        str(round_id),
        "--sampling_mode",
        "random" if round_id == 0 else "external",
        "--full_count",
        str(full_count),
        "--seed",
        str(args.seed),
    ]

    if round_id > 0:
        if selection_file is None:
            raise ValueError(
                "External split requires a selection file"
            )
        command.extend(
            [
                "--selection_file",
                str(selection_file),
            ]
        )

    if stage_overwrite:
        command.append("--overwrite")

    return command


def build_template_command(
    args: argparse.Namespace,
    scripts: dict[str, Path],
    dataset: str,
    method: str,
    round_tag: str,
    stage_overwrite: bool,
) -> list[str]:
    command = [
        str(args.python),
        str(scripts["template"]),
        "--processed_root",
        str(args.processed_root),
        "--checkpoint",
        str(args.base_checkpoint),
        "--datasets",
        dataset,
        "--fold",
        args.fold,
        "--method",
        method,
        "--round_tag",
        round_tag,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--kmax_shape",
        str(args.kmax_shape),
        "--shape_size",
        str(args.shape_size),
        "--cluster_max_iter",
        str(args.cluster_max_iter),
        "--ring_expand_ratio",
        str(args.ring_expand_ratio),
    ]

    if stage_overwrite:
        command.append("--overwrite")

    return command


def build_train_command(
    args: argparse.Namespace,
    scripts: dict[str, Path],
    dataset: str,
    method: str,
    round_tag: str,
    stage_overwrite: bool,
) -> list[str]:
    command = [
        str(args.python),
        str(scripts["train"]),
        "--processed_root",
        str(args.processed_root),
        "--checkpoint",
        str(args.base_checkpoint),
        "--datasets",
        dataset,
        "--fold",
        args.fold,
        "--method",
        method,
        "--round_tag",
        round_tag,
        "--medsam_ft_root",
        str(args.medsam_ft_root),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--epochs",
        str(args.epochs),
        "--max_steps",
        str(args.max_steps),
        "--lr",
        str(args.lr),
        "--weight_decay",
        str(args.weight_decay),
        "--ema_decay",
        str(args.ema_decay),
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--log_every",
        str(args.train_log_every),
    ]

    if stage_overwrite:
        command.append("--overwrite")

    return command


def build_pseudo_command(
    args: argparse.Namespace,
    scripts: dict[str, Path],
    dataset: str,
    method: str,
    round_tag: str,
    ema_checkpoint: Path,
    stage_overwrite: bool,
) -> list[str]:
    command = [
        str(args.python),
        str(scripts["pseudo"]),
        "--processed_root",
        str(args.processed_root),
        "--base_checkpoint",
        str(args.base_checkpoint),
        "--ema_checkpoint",
        str(ema_checkpoint),
        "--datasets",
        dataset,
        "--fold",
        args.fold,
        "--method",
        method,
        "--round_tag",
        round_tag,
        "--device",
        args.device,
        "--max_samples",
        str(args.pseudo_max_samples),
        "--alpha",
        str(args.alpha),
        "--beta",
        str(args.beta),
        "--gamma",
        str(args.gamma),
        "--tau_low",
        str(args.tau_low),
        "--tau_high",
        str(args.tau_high),
        "--temperature",
        str(args.temperature),
        "--top_k",
        str(args.top_k),
        "--log_every",
        str(args.pseudo_log_every),
    ]

    if stage_overwrite:
        command.append("--overwrite")

    return command


def build_select_command(
    args: argparse.Namespace,
    scripts: dict[str, Path],
    dataset: str,
    method: str,
    round_tag: str,
    round_id: int,
    select_count: int,
    stage_overwrite: bool,
) -> list[str]:
    command = [
        str(args.python),
        str(scripts["select"]),
        "--processed_root",
        str(args.processed_root),
        "--datasets",
        dataset,
        "--fold",
        args.fold,
        "--method",
        method,
        "--round_tag",
        round_tag,
        "--round_id",
        str(round_id),
        "--select_count",
        str(select_count),
    ]

    # 05 writes both method-specific outputs and a shared hand-off file:
    # meta/full_selection_round_<next_round>.json. That shared filename can
    # legitimately remain from another experiment (for example, a smoke run).
    # In that case, allow 05 to replace it. Under --resume/--overwrite this
    # also safely refreshes incomplete method-specific selection outputs.
    shared_next_selection = (
        args.processed_root
        / dataset
        / args.fold
        / "meta"
        / f"full_selection_round_{round_id + 1}.json"
    )
    if stage_overwrite or shared_next_selection.exists():
        command.append("--overwrite")

    return command


def load_or_initialize_state(
    run_dir: Path,
    config: dict[str, Any],
    resume: bool,
    overwrite: bool,
) -> tuple[dict[str, Any], Path]:
    state_path = run_dir / "run_state.json"

    if state_path.exists() and resume:
        state = load_json(state_path)
        if not isinstance(state, dict):
            raise TypeError(
                f"Invalid state file: {state_path}"
            )
        return state, state_path

    if state_path.exists() and not overwrite:
        raise FileExistsError(
            f"Runner state already exists: {state_path}. "
            "Use --resume to continue or --overwrite to rerun."
        )

    state = {
        "schema_version": 1,
        "run_name": config["run_name"],
        "status": "initialized",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "config_path": str(
            run_dir / "run_config.json"
        ),
        "datasets": {},
        "last_error": None,
    }
    save_json_atomic(state, state_path)
    return state, state_path


def update_stage_state(
    state: dict[str, Any],
    state_path: Path,
    dataset: str,
    round_id: int,
    method: str,
    full_count: int,
    stage: str,
    status: str,
    detail: str = "",
    log_path: Path | None = None,
) -> None:
    datasets = state.setdefault(
        "datasets",
        {},
    )
    dataset_state = datasets.setdefault(
        dataset,
        {
            "rounds": {},
        },
    )
    rounds = dataset_state.setdefault(
        "rounds",
        {},
    )
    round_state = rounds.setdefault(
        str(round_id),
        {
            "method": method,
            "full_count": full_count,
            "stages": {},
        },
    )

    round_state["method"] = method
    round_state["full_count"] = full_count

    stage_state = {
        "status": status,
        "updated_at": utc_now(),
        "detail": detail,
    }
    if log_path is not None:
        stage_state["log_path"] = str(log_path)

    round_state.setdefault(
        "stages",
        {},
    )[stage] = stage_state

    state["updated_at"] = utc_now()
    save_json_atomic(state, state_path)


def run_command(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    commands_jsonl: Path,
    env: dict[str, str],
    dataset: str,
    round_id: int,
    method: str,
    stage: str,
    dry_run: bool,
) -> None:
    command_text = format_command(command)
    started_at = utc_now()

    print(
        f"\n[RUN] dataset={dataset} "
        f"round={round_id} stage={stage}"
    )
    print(f"      method={method}")
    print(f"      command={command_text}")

    command_record: dict[str, Any] = {
        "started_at": started_at,
        "dataset": dataset,
        "round_id": round_id,
        "method": method,
        "stage": stage,
        "cwd": str(cwd),
        "command": command,
        "command_text": command_text,
        "dry_run": dry_run,
    }

    if dry_run:
        command_record.update(
            {
                "finished_at": utc_now(),
                "return_code": None,
                "status": "dry_run",
            }
        )
        append_jsonl(
            command_record,
            commands_jsonl,
        )
        return

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_file:
        log_file.write(
            f"[started_at] {started_at}\n"
        )
        log_file.write(
            f"[cwd] {cwd}\n"
        )
        log_file.write(
            f"[command] {command_text}\n\n"
        )
        log_file.flush()

        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
                log_file.flush()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise

        return_code = process.wait()

        finished_at = utc_now()
        log_file.write(
            f"\n[finished_at] {finished_at}\n"
        )
        log_file.write(
            f"[return_code] {return_code}\n"
        )

    command_record.update(
        {
            "finished_at": finished_at,
            "return_code": return_code,
            "status": (
                "completed"
                if return_code == 0
                else "failed"
            ),
        }
    )
    append_jsonl(
        command_record,
        commands_jsonl,
    )

    if return_code != 0:
        raise subprocess.CalledProcessError(
            return_code,
            command,
        )


def execute_or_skip_stage(
    *,
    args: argparse.Namespace,
    state: dict[str, Any],
    state_path: Path,
    run_dir: Path,
    commands_jsonl: Path,
    env: dict[str, str],
    dataset: str,
    round_id: int,
    method: str,
    full_count: int,
    stage: str,
    command: list[str],
    validator,
) -> None:
    valid, reason = validator()

    if args.resume and valid:
        print(
            f"[SKIP] dataset={dataset} "
            f"round={round_id} stage={stage}: "
            f"{reason}"
        )
        update_stage_state(
            state,
            state_path,
            dataset,
            round_id,
            method,
            full_count,
            stage,
            "validated_existing",
            reason,
        )
        return

    log_path = (
        run_dir
        / "logs"
        / dataset
        / f"round_{round_id:02d}"
        / f"{stage}.log"
    )

    update_stage_state(
        state,
        state_path,
        dataset,
        round_id,
        method,
        full_count,
        stage,
        "running" if not args.dry_run else "dry_run",
        "command prepared",
        log_path,
    )

    run_command(
        command,
        cwd=args.code_root,
        log_path=log_path,
        commands_jsonl=commands_jsonl,
        env=env,
        dataset=dataset,
        round_id=round_id,
        method=method,
        stage=stage,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        update_stage_state(
            state,
            state_path,
            dataset,
            round_id,
            method,
            full_count,
            stage,
            "dry_run",
            "command not executed",
            log_path,
        )
        return

    valid, reason = validator()
    if not valid:
        update_stage_state(
            state,
            state_path,
            dataset,
            round_id,
            method,
            full_count,
            stage,
            "validation_failed",
            reason,
            log_path,
        )
        raise RuntimeError(
            f"Stage {stage} completed but validation failed: "
            f"dataset={dataset}, round={round_id}, "
            f"method={method}, reason={reason}"
        )

    update_stage_state(
        state,
        state_path,
        dataset,
        round_id,
        method,
        full_count,
        stage,
        "completed",
        reason,
        log_path,
    )


def should_stop(
    stop_after: str,
    stage: str,
) -> bool:
    return stop_after == stage


def build_config(
    args: argparse.Namespace,
    datasets: list[str],
    dimensions: dict[str, bool],
    schedule: list[int],
    round_tags: list[str],
    scripts: dict[str, Path],
    run_name: str,
) -> dict[str, Any]:
    code_hashes = {
        path.name: sha256_file(path)
        for path in scripts.values()
    }

    return {
        "schema_version": 1,
        "run_name": run_name,
        "created_at": utc_now(),
        "python": str(args.python),
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": str(args.code_root),
        "cuda_visible_devices": (
            args.cuda_visible_devices
        ),
        "datasets": datasets,
        "dimensions": {
            dataset: (
                "3d"
                if dimensions[dataset]
                else "2d"
            )
            for dataset in datasets
        },
        "fold": args.fold,
        "preset": args.preset,
        "schedule": schedule,
        "method": args.method,
        "round_tags": round_tags,
        "seed": args.seed,
        "paths": {
            "code_root": str(args.code_root),
            "processed_root": str(
                args.processed_root
            ),
            "base_checkpoint": str(
                args.base_checkpoint
            ),
            "medsam_ft_root": str(
                args.medsam_ft_root
            ),
            "runner_root": str(
                args.runner_root
            ),
        },
        "protocol": {
            "round_zero_sampling": "random",
            "later_round_sampling": (
                "external cumulative selection"
            ),
            "hard_selection_signal": (
                "Teacher-space train GT IoU "
                "used only by 05 selector"
            ),
            "training_initialization_each_round": (
                "same base MedSAM checkpoint"
            ),
            "pseudo_checkpoint_each_round": (
                "medsam_sac_ema.pth"
            ),
            "test_samples_used_for_selection": 0,
            "final_round_runs_selector": False,
        },
        "support": {
            "kmax_shape": args.kmax_shape,
            "shape_size": args.shape_size,
            "cluster_max_iter": args.cluster_max_iter,
            "ring_expand_ratio": args.ring_expand_ratio,
        },
        "training": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "ema_decay": args.ema_decay,
            "max_grad_norm": args.max_grad_norm,
            "log_every": args.train_log_every,
        },
        "pseudo": {
            "max_samples": (
                args.pseudo_max_samples
            ),
            "alpha": args.alpha,
            "beta": args.beta,
            "gamma": args.gamma,
            "tau_low": args.tau_low,
            "tau_high": args.tau_high,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "log_every": args.pseudo_log_every,
        },
        "execution": {
            "resume": args.resume,
            "overwrite": args.overwrite,
            "dry_run": args.dry_run,
            "start_round": args.start_round,
            "end_round": args.end_round,
            "stop_after": args.stop_after,
        },
        "script_hashes": code_hashes,
    }


def run_pipeline(
    args: argparse.Namespace,
) -> None:
    datasets = parse_csv_strings(
        args.datasets
    )
    if not datasets:
        raise ValueError("--datasets is empty")

    if len(datasets) != len(set(datasets)):
        raise ValueError(
            f"Duplicate datasets: {datasets}"
        )

    schedule = determine_schedule(args)

    if args.start_round < 0:
        raise ValueError(
            "--start_round must be >= 0"
        )

    final_round = len(schedule) - 1
    end_round = (
        final_round
        if args.end_round is None
        else args.end_round
    )

    if end_round < args.start_round:
        raise ValueError(
            "--end_round must be >= --start_round"
        )

    if end_round > final_round:
        raise ValueError(
            f"--end_round={end_round} exceeds "
            f"final round={final_round}"
        )

    scripts = get_script_paths(
        args.code_root
    )
    compile_scripts(scripts)

    dimensions = {
        dataset: infer_is_3d(
            args.processed_root,
            dataset,
            args.fold,
        )
        for dataset in datasets
    }

    validate_dimension_group(
        args,
        datasets,
        dimensions,
    )
    validate_dataset_roots(
        args,
        datasets,
        dimensions,
    )

    is_3d = dimensions[datasets[0]]
    round_tags = build_round_tags(schedule, is_3d)

    run_name = (
        sanitize_name(args.run_name)
        if args.run_name
        else default_run_name(
            datasets,
            schedule,
            args.method,
        )
    )

    run_dir = (
        args.runner_root
        / run_name
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    config = build_config(
        args,
        datasets,
        dimensions,
        schedule,
        round_tags,
        scripts,
        run_name,
    )
    config_path = (
        run_dir
        / "run_config.json"
    )

    if (
        config_path.exists()
        and args.resume
    ):
        old_config = load_json(config_path)
        if isinstance(old_config, dict):
            comparable_keys = (
                "datasets",
                "fold",
                "schedule",
                "method",
            )
            mismatches = [
                key
                for key in comparable_keys
                if old_config.get(key)
                != config.get(key)
            ]
            if mismatches:
                raise RuntimeError(
                    "Resume configuration mismatch for keys: "
                    f"{mismatches}"
                )

    save_json_atomic(
        config,
        config_path,
    )

    state, state_path = load_or_initialize_state(
        run_dir,
        config,
        resume=args.resume,
        overwrite=args.overwrite,
    )

    commands_jsonl = (
        run_dir
        / "commands.jsonl"
    )

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = (
            args.cuda_visible_devices
        )

    state["status"] = (
        "dry_run"
        if args.dry_run
        else "running"
    )
    state["updated_at"] = utc_now()
    state["last_error"] = None
    save_json_atomic(state, state_path)

    stage_overwrite = (
        args.overwrite
        or args.resume
    )

    try:
        for dataset in datasets:
            is_3d = dimensions[dataset]

            for round_id in range(
                args.start_round,
                end_round + 1,
            ):
                full_count = schedule[
                    round_id
                ]
                round_tag = round_tags[
                    round_id
                ]

                print(
                    "\n"
                    + "=" * 88
                )
                print(
                    f"[ROUND] dataset={dataset} "
                    f"round={round_id}/{final_round} "
                    f"method={args.method} "
                    f"round_tag={round_tag} "
                    f"Full units={full_count} "
                    f"dimension={'3D' if is_3d else '2D'}"
                )
                print(
                    "=" * 88
                )

                previous_selection: Path | None = None

                if round_id > 0:
                    previous_round_tag = round_tags[
                        round_id - 1
                    ]
                    previous_run_id = f"{args.method}_{previous_round_tag}"
                    previous_selection = (
                        selection_snapshot_path(
                            run_dir,
                            dataset,
                            round_id - 1,
                            previous_run_id,
                        )
                    )

                    if (
                        not previous_selection.is_file()
                        and not args.dry_run
                    ):
                        previous_ok, previous_reason = (
                            validate_selection_stage(
                                args,
                                dataset,
                                args.method,
                                previous_round_tag,
                                schedule[round_id - 1],
                                schedule[round_id],
                                is_3d,
                            )
                        )
                        if not previous_ok:
                            raise RuntimeError(
                                "Cannot build external split for "
                                f"round {round_id}; previous selection "
                                f"is unavailable or invalid: "
                                f"{previous_reason}"
                            )

                        previous_selection = (
                            create_or_validate_selection_snapshot(
                                args,
                                run_dir,
                                dataset,
                                args.method,
                                previous_round_tag,
                                round_id - 1,
                                round_id,
                                schedule[round_id],
                            )
                        )

                split_command = build_split_command(
                    args,
                    scripts,
                    dataset,
                    args.method,
                    round_tag,
                    round_id,
                    full_count,
                    previous_selection,
                    stage_overwrite,
                )

                execute_or_skip_stage(
                    args=args,
                    state=state,
                    state_path=state_path,
                    run_dir=run_dir,
                    commands_jsonl=commands_jsonl,
                    env=env,
                    dataset=dataset,
                    round_id=round_id,
                    method=args.method,
                    full_count=full_count,
                    stage="split",
                    command=split_command,
                    validator=lambda d=dataset, m=args.method, rt=round_tag, f=full_count, dim=is_3d: (
                        validate_split_stage(
                            args,
                            d,
                            m,
                            rt,
                            f,
                            dim,
                        )
                    ),
                )

                if should_stop(
                    args.stop_after,
                    "split",
                ):
                    state["status"] = "stopped_after_split"
                    state["updated_at"] = utc_now()
                    save_json_atomic(state, state_path)
                    return

                template_command = (
                    build_template_command(
                        args,
                        scripts,
                        dataset,
                        args.method,
                        round_tag,
                        stage_overwrite,
                    )
                )

                execute_or_skip_stage(
                    args=args,
                    state=state,
                    state_path=state_path,
                    run_dir=run_dir,
                    commands_jsonl=commands_jsonl,
                    env=env,
                    dataset=dataset,
                    round_id=round_id,
                    method=args.method,
                    full_count=full_count,
                    stage="template",
                    command=template_command,
                    validator=lambda d=dataset, m=args.method, rt=round_tag: (
                        validate_template_stage(
                            args,
                            d,
                            m,
                            rt,
                        )
                    ),
                )

                if should_stop(
                    args.stop_after,
                    "template",
                ):
                    state["status"] = "stopped_after_template"
                    state["updated_at"] = utc_now()
                    save_json_atomic(state, state_path)
                    return

                train_command = build_train_command(
                    args,
                    scripts,
                    dataset,
                    args.method,
                    round_tag,
                    stage_overwrite,
                )

                execute_or_skip_stage(
                    args=args,
                    state=state,
                    state_path=state_path,
                    run_dir=run_dir,
                    commands_jsonl=commands_jsonl,
                    env=env,
                    dataset=dataset,
                    round_id=round_id,
                    method=args.method,
                    full_count=full_count,
                    stage="train",
                    command=train_command,
                    validator=lambda d=dataset, m=args.method, rt=round_tag: (
                        validate_train_stage(
                            args,
                            d,
                            m,
                            rt,
                        )
                    ),
                )

                if should_stop(
                    args.stop_after,
                    "train",
                ):
                    state["status"] = "stopped_after_train"
                    state["updated_at"] = utc_now()
                    save_json_atomic(state, state_path)
                    return

                paths = round_output_paths(
                    args,
                    dataset,
                    args.method,
                    round_tag,
                )
                ema_checkpoint = paths[
                    "train_ema"
                ]

                pseudo_command = (
                    build_pseudo_command(
                        args,
                        scripts,
                        dataset,
                        args.method,
                        round_tag,
                        ema_checkpoint,
                        stage_overwrite,
                    )
                )

                execute_or_skip_stage(
                    args=args,
                    state=state,
                    state_path=state_path,
                    run_dir=run_dir,
                    commands_jsonl=commands_jsonl,
                    env=env,
                    dataset=dataset,
                    round_id=round_id,
                    method=args.method,
                    full_count=full_count,
                    stage="pseudo",
                    command=pseudo_command,
                    validator=lambda d=dataset, m=args.method, rt=round_tag: (
                        validate_pseudo_stage(
                            args,
                            d,
                            m,
                            rt,
                        )
                    ),
                )

                if should_stop(
                    args.stop_after,
                    "pseudo",
                ):
                    state["status"] = "stopped_after_pseudo"
                    state["updated_at"] = utc_now()
                    save_json_atomic(state, state_path)
                    return

                if not is_formal_final_round(round_id, is_3d):
                    next_full_count = _formal_schedule(is_3d)[
                        round_id + 1
                    ]
                    default_select = 1 if is_3d else 5
                    select_count = (
                        args.select_count
                        if args.select_count is not None
                        else default_select
                    )
                    schedule_diff = next_full_count - full_count
                    if select_count != schedule_diff:
                        raise ValueError(
                            f"select_count mismatch: "
                            f"select_count={select_count}, "
                            f"schedule_diff={schedule_diff} "
                            f"(next={next_full_count} - "
                            f"current={full_count})"
                        )

                    select_command = (
                        build_select_command(
                            args,
                            scripts,
                            dataset,
                            args.method,
                            round_tag,
                            round_id,
                            select_count,
                            stage_overwrite,
                        )
                    )

                    execute_or_skip_stage(
                        args=args,
                        state=state,
                        state_path=state_path,
                        run_dir=run_dir,
                        commands_jsonl=commands_jsonl,
                        env=env,
                        dataset=dataset,
                        round_id=round_id,
                        method=args.method,
                        full_count=full_count,
                        stage="select",
                        command=select_command,
                        validator=lambda d=dataset, m=args.method, rt=round_tag, p=full_count, n=next_full_count, dim=is_3d: (
                            validate_selection_stage(
                                args,
                                d,
                                m,
                                rt,
                                p,
                                n,
                                dim,
                            )
                        ),
                    )

                    if not args.dry_run:
                        snapshot = (
                            create_or_validate_selection_snapshot(
                                args,
                                run_dir,
                                dataset,
                                args.method,
                                round_tag,
                                round_id,
                                round_id + 1,
                                next_full_count,
                            )
                        )
                        print(
                            "[SNAPSHOT] cumulative Full selection: "
                            f"{snapshot}"
                        )

                    if should_stop(
                        args.stop_after,
                        "select",
                    ):
                        state["status"] = "stopped_after_select"
                        state["updated_at"] = utc_now()
                        save_json_atomic(state, state_path)
                        return
                else:
                    update_stage_state(
                        state,
                        state_path,
                        dataset,
                        round_id,
                        args.method,
                        full_count,
                        "select",
                        "not_required_final_round",
                        (
                            "Final round keeps the generated "
                            "pseudo-labels and does not select "
                            "another Full increment."
                        ),
                    )

                # Write round summary
                is_final_round = is_formal_final_round(round_id, is_3d)
                round_paths = round_output_paths(
                    args, dataset, args.method, round_tag,
                )
                round_summary = {
                    "dataset": dataset,
                    "fold": args.fold,
                    "method": args.method,
                    "round_id": round_id,
                    "round_tag": round_tag,
                    "full_count": full_count,
                    "is_final_round": is_final_round,
                    "select_required": not is_final_round,
                    "select_status": (
                        "not_required"
                        if is_final_round
                        else "completed"
                    ),
                    "completed_at": utc_now(),
                }
                save_json_atomic(
                    round_summary, round_paths["round_summary"],
                )

                if args.stop_after == "round":
                    state["status"] = "stopped_after_round"
                    state["updated_at"] = utc_now()
                    save_json_atomic(state, state_path)
                    return

        state["status"] = (
            "dry_run_completed"
            if args.dry_run
            else "completed"
        )
        state["updated_at"] = utc_now()
        state["completed_at"] = utc_now()
        save_json_atomic(state, state_path)

        print("\n" + "=" * 88)
        print(
            "[OK] iterative sampling pipeline completed"
            if not args.dry_run
            else "[OK] dry-run command generation completed"
        )
        print(f"     run_dir = {run_dir}")
        print(f"     config  = {config_path}")
        print(f"     state   = {state_path}")
        print(f"     commands= {commands_jsonl}")
        print("=" * 88)

    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["updated_at"] = utc_now()
        state["last_error"] = "KeyboardInterrupt"
        save_json_atomic(state, state_path)
        print("\n[INTERRUPTED] state saved for --resume")
        raise
    except Exception as exc:
        state["status"] = "failed"
        state["updated_at"] = utc_now()
        state["last_error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        save_json_atomic(state, state_path)
        raise


def add_training_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    group = parser.add_argument_group(
        "SAC training"
    )

    group.add_argument(
        "--epochs",
        type=int,
        default=1,
    )
    group.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help=(
            "0 means no step cap. "
            "Use 5 for the smoke test."
        ),
    )
    group.add_argument(
        "--lr",
        type=float,
        default=1e-5,
    )
    group.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
    )
    group.add_argument(
        "--ema_decay",
        type=float,
        default=0.99,
    )
    group.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
    )
    group.add_argument(
        "--train_log_every",
        type=int,
        default=10,
    )


def add_support_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    group = parser.add_argument_group(
        "support template"
    )

    group.add_argument(
        "--kmax_shape",
        type=int,
        default=5,
    )
    group.add_argument(
        "--shape_size",
        type=int,
        default=64,
    )
    group.add_argument(
        "--cluster_max_iter",
        type=int,
        default=25,
    )
    group.add_argument(
        "--ring_expand_ratio",
        type=int,
        default=5,
    )


def add_pseudo_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    group = parser.add_argument_group(
        "pseudo-label generation"
    )

    group.add_argument(
        "--pseudo_max_samples",
        type=int,
        default=0,
        help=(
            "0 means all train samples. "
            "The formal iterative run should keep 0."
        ),
    )
    group.add_argument(
        "--alpha",
        type=float,
        default=0.70,
        help="Weight for P (prediction probability).",
    )
    group.add_argument(
        "--beta",
        type=float,
        default=0.10,
        help="Weight for A (adaptive shape prior).",
    )
    group.add_argument(
        "--gamma",
        type=float,
        default=0.20,
        help="Weight for QF (feature bank query).",
    )
    group.add_argument(
        "--tau_low",
        type=float,
        default=0.30,
        help="Score <= tau_low → background (0).",
    )
    group.add_argument(
        "--tau_high",
        type=float,
        default=0.70,
        help="Score >= tau_high → foreground (label_id).",
    )
    group.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for bank similarity softmax.",
    )
    group.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Top-K nearest bank neighbours for QF score.",
    )
    group.add_argument(
        "--pseudo_log_every",
        type=int,
        default=50,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run iterative Full/Box hard sampling for SAC-MedSAM. "
            "Each round rebuilds the support template, retrains "
            "from the same base MedSAM checkpoint, generates "
            "EMA pseudo-labels, and selects the lowest-IoU "
            "remaining Box units for the next round."
        ),
        formatter_class=(
            argparse.ArgumentDefaultsHelpFormatter
        ),
    )

    parser.add_argument(
        "--datasets",
        required=True,
        help=(
            "Comma-separated datasets. "
            "Do not mix 2D and 3D in one call."
        ),
    )
    parser.add_argument(
        "--fold",
        default="fold_0",
    )

    schedule_group = (
        parser.add_mutually_exclusive_group()
    )
    schedule_group.add_argument(
        "--preset",
        choices=(
            "custom",
            "formal_2d",
            "formal_3d",
        ),
        default="custom",
    )
    schedule_group.add_argument(
        "--schedule",
        help=(
            "Strictly increasing Full-unit counts, "
            "for example 5,7 or 5,10,15,20. "
            "2D units are images; 3D units are cases."
        ),
    )

    parser.add_argument(
        "--method",
        default=METHOD_DEFAULT,
        help="Method identifier (default: %(default)s)",
    )
    parser.add_argument(
        "--run_name",
        default="",
        help=(
            "Stable runner directory name. "
            "Specify this when using --resume."
        ),
    )

    parser.add_argument(
        "--code_root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--processed_root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
    )
    parser.add_argument(
        "--base_checkpoint",
        type=Path,
        default=DEFAULT_BASE_CHECKPOINT,
    )
    parser.add_argument(
        "--medsam_ft_root",
        type=Path,
        default=DEFAULT_MEDSAM_FT_ROOT,
    )
    parser.add_argument(
        "--runner_root",
        type=Path,
        default=DEFAULT_RUNNER_ROOT,
    )
    parser.add_argument(
        "--python",
        type=Path,
        default=Path(sys.executable),
        help=(
            "Python executable used for 01-05. "
            "The current medsam310 Python is recommended."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        default=None,
        help=(
            "Optional CUDA_VISIBLE_DEVICES value, "
            "for example 0 or 2."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    execution_group = parser.add_argument_group(
        "execution control"
    )
    mode_group = (
        execution_group.add_mutually_exclusive_group()
    )
    mode_group.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Validate and skip completed stages; "
            "rerun missing or invalid stages."
        ),
    )
    mode_group.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Rerun all selected rounds and pass "
            "--overwrite to writable stages."
        ),
    )
    execution_group.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "Write configuration and commands "
            "without running child scripts."
        ),
    )
    execution_group.add_argument(
        "--start_round",
        type=int,
        default=0,
    )
    execution_group.add_argument(
        "--end_round",
        type=int,
        default=None,
    )
    execution_group.add_argument(
        "--stop_after",
        choices=STOP_AFTER_CHOICES,
        default="all",
        help=(
            "Stop after the selected stage. "
            "'round' stops after one complete round."
        ),
    )

    add_support_arguments(parser)
    add_training_arguments(parser)
    add_pseudo_arguments(parser)

    select_group = parser.add_argument_group("hard-sample selection")
    select_group.add_argument(
        "--select_count",
        type=int,
        default=None,
        help=(
            "Number of hard samples to select per non-final round. "
            "Default: 5 for 2D, 1 for 3D datasets."
        ),
    )

    return parser


def validate_numeric_args(
    args: argparse.Namespace,
) -> None:
    positive_ints = (
        "epochs",
        "kmax_shape",
        "shape_size",
        "cluster_max_iter",
        "ring_expand_ratio",
        "top_k",
        "train_log_every",
        "pseudo_log_every",
    )
    for name in positive_ints:
        value = int(getattr(args, name))
        if value <= 0:
            raise ValueError(
                f"--{name} must be positive, got {value}"
            )

    nonnegative_ints = (
        "max_steps",
        "pseudo_max_samples",
        "seed",
    )
    for name in nonnegative_ints:
        value = int(getattr(args, name))
        if value < 0:
            raise ValueError(
                f"--{name} must be >= 0, got {value}"
            )

    if args.pseudo_max_samples > 0:
        raise ValueError(
            "Iterative hard selection requires pseudo-labels "
            "for every train sample. Keep "
            "--pseudo_max_samples 0."
        )


def main() -> None:
    args = build_parser().parse_args()

    args.code_root = args.code_root.resolve()
    args.processed_root = (
        args.processed_root.resolve()
    )
    args.base_checkpoint = (
        args.base_checkpoint.resolve()
    )
    args.medsam_ft_root = (
        args.medsam_ft_root.resolve()
    )
    args.runner_root = (
        args.runner_root.resolve()
    )
    args.python = args.python.resolve()

    validate_numeric_args(args)
    run_pipeline(args)


if __name__ == "__main__":
    main()
