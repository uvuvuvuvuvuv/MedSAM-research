#!/usr/bin/env python3
"""Validate round outputs for the iterative sampling pipeline.

Checks every stage output for a given (dataset, fold, method, round_tag)
combination.  With --strict, every check must pass.  Returns exit code
equal to the number of failures.

Usage::

    python 08_validate_round_outputs.py \\
        --processed_root /path/to/processed \\
        --dataset cvc_clinicdb \\
        --fold fold_0 \\
        --method idea1_iterclean_bank_adaptshape \\
        --round_tag r00_full5 \\
        [--medsam_ft_root /path/to/ft] \\
        [--strict]
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Import pipeline_common for path construction
# ---------------------------------------------------------------------------
try:
    from .pipeline_common import (
        METHOD_DEFAULT,
        build_fold_paths,
        get_round_tag,
        validate_writable_path,
    )
except ImportError:
    from pipeline_common import (  # type: ignore[no-redef]
        METHOD_DEFAULT,
        build_fold_paths,
        get_round_tag,
        validate_writable_path,
    )


# ============================================================================
# Helpers
# ============================================================================


def _ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _check_file(path: Path, label: str) -> bool:
    if not path.is_file():
        _fail(f"{label}: file missing — {path}")
        return False
    if path.stat().st_size == 0:
        _fail(f"{label}: file is empty — {path}")
        return False
    _ok(f"{label}: {path}")
    return True


def _check_no_symlink(path: Path, label: str, strict: bool) -> list[str]:
    """Return errors if *path* or any parent is a symlink."""
    errors: list[str] = []
    if path is None:
        return errors
    parts = list(path.parents) + [path]
    for part in parts:
        if part.is_symlink():
            msg = f"{label}: symlink detected — {part}"
            if strict:
                errors.append(msg)
                _fail(msg)
            else:
                _ok(f"{label}: (symlink, non-strict) {part}")
            break
    return errors


_FORMAL_2D_SCHEDULE = [5, 10, 15, 20]
_FORMAL_3D_SCHEDULE = [1, 2, 3, 4, 5]


def _is_formal_final_round(round_tag: str, is_3d: bool) -> bool:
    """True only when *round_tag* is the last round of the full formal schedule."""
    schedule = _FORMAL_3D_SCHEDULE if is_3d else _FORMAL_2D_SCHEDULE
    formal_final_round_id = len(schedule) - 1
    for rid in range(len(schedule)):
        if get_round_tag(rid, is_3d) == round_tag:
            return rid == formal_final_round_id
    return False


SYMLINK_KEYS = (
    "split_path", "split_summary_path",
    "support_path", "support_stats_path",
    "train_last_path", "train_ema_path",
    "train_log_path", "train_summary_path",
)


# ============================================================================
# Stage validators
# ============================================================================


def validate_split(paths: dict[str, Path], strict: bool) -> list[str]:
    errors: list[str] = []
    split_path = paths["split_path"]
    summary_path = paths["split_summary_path"]
    run_id = paths.get("run_id", "")

    # Check symlinks
    for key, label in (("split_path", "Split"), ("split_summary_path", "Split summary")):
        errors.extend(_check_no_symlink(paths.get(key), label, strict))

    # Check split filename contains run_id
    if run_id and split_path.is_file():
        if run_id not in split_path.name:
            errors.append(f"Split filename missing run_id: {split_path.name}")

    if not split_path.is_file():
        errors.append(f"Split file missing: {split_path}")
        return errors

    try:
        records = _load_json(split_path)
    except Exception as exc:
        errors.append(f"Cannot read split JSON: {exc}")
        return errors

    if not isinstance(records, list):
        errors.append("Split JSON is not a list")
        return errors

    full_records = [r for r in records if isinstance(r, dict) and r.get("label_mode") == "full"]
    box_records = [r for r in records if isinstance(r, dict) and r.get("label_mode") == "box"]
    train_records = [r for r in records if isinstance(r, dict) and r.get("split") == "train"]
    test_records = [r for r in records if isinstance(r, dict) and r.get("split") == "test"]

    if not full_records:
        errors.append("Split has no Full records")
    else:
        _ok(f"Split: {len(full_records)} Full records, {len(box_records)} Box records")

    if test_records:
        errors.append(f"Split contains {len(test_records)} test records — test leakage!")

    if train_records and not full_records:
        errors.append("Split has train records but no Full-mode records")

    # Check summary
    if summary_path.is_file():
        try:
            summary = _load_json(summary_path)
        except Exception as exc:
            errors.append(f"Cannot read split summary: {exc}")
        else:
            if isinstance(summary, dict):
                leakage = summary.get("test_leakage")
                if leakage not in (None, 0, False):
                    errors.append(f"Split summary reports test leakage: {leakage}")
                _ok(f"Split summary: {summary_path}")
    else:
        _ok("Split summary: (not present)")

    return errors


def validate_template(paths: dict[str, Path], strict: bool) -> list[str]:
    errors: list[str] = []
    support_path = paths["support_path"]
    stats_path = paths["support_stats_path"]

    errors.extend(_check_no_symlink(support_path, "Support NPZ", strict))
    errors.extend(_check_no_symlink(stats_path, "Support stats", strict))

    if not _check_file(support_path, "Support NPZ"):
        errors.append(f"Support NPZ missing: {support_path}")
        return errors

    # NPZ contract check
    try:
        with zipfile.ZipFile(str(support_path), "r") as zf:
            names = set(zf.namelist())
    except Exception as exc:
        errors.append(f"Invalid NPZ archive: {exc}")
        return errors

    # Required global keys
    for required_npy in ("class_ids.npy", "feature_dim.npy", "shape_size.npy"):
        if required_npy not in names:
            errors.append(f"NPZ missing global key: {required_npy}")

    if "class_ids.npy" not in names:
        errors.append("NPZ has no class_ids — cannot verify per-class keys")
    else:
        # Extract class_ids to verify per-class keys
        import io as _io
        import numpy as np
        with zipfile.ZipFile(str(support_path), "r") as zf:
            raw = zf.read("class_ids.npy")
        class_ids = np.load(_io.BytesIO(raw))
        for c in class_ids:
            c_int = int(c)
            for prefix in (
                "bank_fg_c", "bank_bg_c",
                "shape_templates_c", "shape_semantic_centers_c",
                "shape_cluster_instance_counts_c", "shape_cluster_support_counts_c",
            ):
                key = f"{prefix}{c_int}.npy"
                if key not in names:
                    errors.append(f"NPZ missing per-class key: {key}")

    # Forbidden old keys
    for forbidden_pattern in ("proto_fg_c", "proto_bg_c", "shape_A_c", "shape_R_c"):
        for name in names:
            if name.startswith(forbidden_pattern):
                errors.append(f"NPZ contains forbidden old key: {name}")

    # allow_pickle=False check
    try:
        import io as _io2
        import numpy as np
        with zipfile.ZipFile(str(support_path), "r") as zf:
            for name in zf.namelist():
                raw = zf.read(name)
                np.load(_io2.BytesIO(raw), allow_pickle=False)
        _ok("NPZ allow_pickle=False: all arrays load successfully")
    except Exception as exc:
        errors.append(f"NPZ allow_pickle=False check failed: {exc}")

    _ok(f"NPZ entries: {len(names)}")

    # Stats JSON
    if stats_path.is_file():
        try:
            stats = _load_json(stats_path)
        except Exception as exc:
            errors.append(f"Cannot read support stats: {exc}")
        else:
            if isinstance(stats, dict):
                # Required new fields.
                for field in (
                    "bg_ring_width_tokens",
                    "fg_coverage_threshold",
                    "max_global_non_bg_coverage",
                    "zero_bg_instance_count",
                    "instances_with_bg_tokens",
                    "instances_without_bg_tokens",
                    "total_bg_ring_candidate_tokens",
                    "total_valid_global_bg_tokens",
                    "bg_ring_valid_ratio",
                    "global_non_bg_coverage_ring_min",
                    "global_non_bg_coverage_ring_mean",
                    "global_non_bg_coverage_ring_max",
                ):
                    if field not in stats:
                        errors.append(f"Support stats missing field: {field}")

                # Forbidden old fields.
                for old_field in ("ring_expand_ratio", "bg_coverage_threshold"):
                    if old_field in stats:
                        errors.append(
                            f"Support stats contains forbidden old field: {old_field}"
                        )

                # Value checks.
                if stats.get("bg_ring_width_tokens") != 1:
                    errors.append(
                        "Support stats: bg_ring_width_tokens must be 1, "
                        f"got {stats.get('bg_ring_width_tokens')}"
                    )
                if abs(stats.get("max_global_non_bg_coverage", -1) - 0.10) > 0.001:
                    errors.append(
                        "Support stats: max_global_non_bg_coverage must be 0.10, "
                        f"got {stats.get('max_global_non_bg_coverage')}"
                    )

                # bg_ring_valid_ratio must be in [0, 1].
                ratio = stats.get("bg_ring_valid_ratio")
                if ratio is not None and not (0.0 <= ratio <= 1.0):
                    errors.append(
                        f"Support stats: bg_ring_valid_ratio out of [0,1]: {ratio}"
                    )

                _ok(f"Support stats: {stats_path}")
    else:
        _ok("Support stats: (not present)")

    return errors


def validate_train(paths: dict[str, Path], strict: bool) -> list[str]:
    errors: list[str] = []

    for key, label in (
        ("train_last_path", "Train last checkpoint"),
        ("train_ema_path", "Train EMA checkpoint"),
        ("train_log_path", "Train log"),
        ("train_summary_path", "Train summary"),
    ):
        path = paths.get(key)
        if path is None:
            _ok(f"{label}: (no ft root configured)")
            continue
        errors.extend(_check_no_symlink(path, label, strict))
        if not path.is_file():
            errors.append(f"{label} missing: {path}")
        elif path.stat().st_size == 0:
            errors.append(f"{label} is empty: {path}")
        else:
            _ok(f"{label}: {path}")

    # Validate summary content and path consistency
    summary_path = paths.get("train_summary_path")
    if summary_path and summary_path.is_file():
        try:
            summary = _load_json(summary_path)
        except Exception as exc:
            errors.append(f"Cannot read train summary: {exc}")
        else:
            if isinstance(summary, dict):
                if summary.get("trainable") != "mask_decoder_only":
                    errors.append(
                        f"Train summary: trainable={summary.get('trainable')!r}, "
                        "expected 'mask_decoder_only'"
                    )
                gs = summary.get("global_step", 0)
                if int(gs) <= 0:
                    errors.append(f"Train summary: global_step={gs} is not positive")

                # Check method/round_tag in summary
                for field in ("method", "round_tag", "run_id"):
                    expected = paths.get(field)
                    actual = summary.get(field)
                    if expected is not None and actual is not None and str(actual) != str(expected):
                        errors.append(
                            f"Train summary {field} mismatch: "
                            f"expected={expected!r}, got={actual!r}"
                        )

                # Check output paths in summary match build_fold_paths
                for key, summary_key in (
                    ("train_last_path", "train_last_path"),
                    ("train_ema_path", "train_ema_path"),
                ):
                    expected_path = str(paths.get(key, ""))
                    actual_path = str(summary.get(summary_key, ""))
                    if expected_path and actual_path and expected_path != actual_path:
                        errors.append(
                            f"Train summary path mismatch for {summary_key}: "
                            f"expected={expected_path}, got={actual_path}"
                        )

    return errors


def validate_pseudo(paths: dict[str, Path]) -> list[str]:
    errors: list[str] = []

    for key, label in (
        ("pseudo_teacher_dir", "Pseudo teacher dir"),
        ("pseudo_student_dir", "Pseudo student dir"),
    ):
        path = paths.get(key)
        if path is None:
            errors.append(f"{label} key missing from paths")
        elif not path.is_dir():
            errors.append(f"{label} not a directory: {path}")
        else:
            npy_count = sum(1 for f in path.iterdir() if f.suffix == ".npy")
            _ok(f"{label}: {npy_count} .npy files in {path}")

    # Audit/config/stats
    for key, label in (
        ("pseudo_audit", "Pseudo audit"),
        ("pseudo_config", "Pseudo config"),
        ("pseudo_stats", "Pseudo stats"),
    ):
        path = paths.get(key)
        if path is None:
            _ok(f"{label}: (key not in paths)")
        elif not path.is_file():
            _ok(f"{label}: (not present)")
        elif path.suffix == ".csv":
            if path.stat().st_size > 0:
                _ok(f"{label}: {path}")
            else:
                errors.append(f"{label} is empty: {path}")
        else:
            try:
                data = _load_json(path)
            except Exception as exc:
                errors.append(f"Cannot read {label}: {exc}")
            else:
                _ok(f"{label}: {path}")
                if key == "pseudo_audit" and isinstance(data, dict):
                    if data.get("output_complete") is not True:
                        errors.append("Pseudo audit: output_complete is not true")

    return errors


def validate_select(paths: dict[str, Path], is_3d: bool, round_tag: str) -> list[str]:
    errors: list[str] = []

    is_formal_final = _is_formal_final_round(round_tag, is_3d)

    # Read round_summary for cross-validation (but do NOT trust it blindly)
    round_summary_path = paths.get("round_summary_path")
    round_summary_is_final = None
    if round_summary_path and round_summary_path.is_file():
        try:
            summary = _load_json(round_summary_path)
        except Exception as exc:
            errors.append(f"Cannot read round summary: {exc}")
            summary = {}
        if isinstance(summary, dict):
            round_summary_is_final = summary.get("is_final_round")
            _ok(
                f"Round summary: is_final_round={summary.get('is_final_round')}, "
                f"select_required={summary.get('select_required')}, "
                f"select_status={summary.get('select_status')}"
            )
    else:
        _ok("Round summary: (not present)")

    # Cross-validate round_summary against formal schedule
    if round_summary_is_final is True and not is_formal_final:
        errors.append(
            f"Round summary: is_final_round=true but formal schedule "
            f"says {round_tag} is NOT the final round"
        )
    if round_summary_is_final is False and is_formal_final:
        errors.append(
            f"Round summary: is_final_round=false but formal schedule "
            f"says {round_tag} IS the final round"
        )

    if is_formal_final:
        _ok("Select: not required (final round per formal schedule)")
        # Verify no stale select leftovers
        for key, label in (
            ("hard_selection_path", "Hard selection"),
            ("hard_summary_path", "Hard summary"),
        ):
            path = paths.get(key)
            if path and path.is_file():
                errors.append(
                    f"{label}: unexpected file in final round: {path}"
                )
        return errors

    # Non-final round: select files are REQUIRED
    for key, label in (
        ("hard_selection_path", "Hard selection"),
        ("hard_summary_path", "Hard summary"),
    ):
        path = paths.get(key)
        if path is None:
            errors.append(f"{label}: key missing from paths")
        elif not _check_file(path, label):
            errors.append(f"{label} check failed")

    # Summary content
    summary_path = paths.get("hard_summary_path")
    if summary_path and summary_path.is_file():
        try:
            summary = _load_json(summary_path)
        except Exception as exc:
            errors.append(f"Cannot read hard summary: {exc}")
        else:
            if isinstance(summary, dict):
                if summary.get("selection_complete") is not True:
                    errors.append("Hard summary: selection_complete is not true")
                if int(summary.get("test_samples_used", -1)) != 0:
                    errors.append("Hard summary: test_samples_used != 0")

    # 3D ranking files
    if is_3d:
        for key, label in (
            ("hard_slice_ranking_3d", "3D slice ranking"),
            ("hard_case_ranking_3d", "3D case ranking"),
        ):
            path = paths.get(key)
            if path and path.is_file():
                _ok(f"{label}: {path}")
            else:
                _ok(f"{label}: (not present)")

    return errors


def validate_round_summary(paths: dict[str, Path], is_3d: bool, round_tag: str) -> list[str]:
    errors: list[str] = []
    path = paths.get("round_summary_path")
    if path is None:
        _ok("Round summary: (key not in paths)")
    elif not path.is_file():
        errors.append("Round summary: file missing (required per round)")
    else:
        try:
            data = _load_json(path)
        except Exception as exc:
            errors.append(f"Round summary invalid JSON: {exc}")
            return errors

        if not isinstance(data, dict):
            errors.append("Round summary is not an object")
            return errors

        is_final = data.get("is_final_round")
        select_required = data.get("select_required")
        select_status = data.get("select_status")

        if is_final is None:
            errors.append(
                "Round summary: missing 'is_final_round'"
            )
        if select_required is None:
            errors.append(
                "Round summary: missing 'select_required'"
            )
        if select_status is None:
            errors.append(
                "Round summary: missing 'select_status'"
            )

        # Cross-validate against formal schedule
        is_formal_final = _is_formal_final_round(round_tag, is_3d)
        if is_final is not None and is_final != is_formal_final:
            errors.append(
                f"Round summary: is_final_round={is_final} but "
                f"formal schedule says is_final_round={is_formal_final} "
                f"for {round_tag}"
            )

        if is_final is True:
            if select_required is not False:
                errors.append(
                    "Round summary: final round must have "
                    "select_required=false"
                )
            if select_status != "not_required":
                errors.append(
                    "Round summary: final round must have "
                    "select_status='not_required'"
                )
        elif is_final is False:
            if select_required is not True:
                errors.append(
                    "Round summary: non-final round must have "
                    "select_required=true"
                )
            if select_status != "completed":
                errors.append(
                    "Round summary: non-final round must have "
                    "select_status='completed'"
                )

        _ok(f"Round summary: {path}")
    return errors


# ============================================================================
# Main
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate round outputs for the iterative sampling pipeline.",
    )
    parser.add_argument(
        "--processed_root", type=Path, required=True,
        help="Processed data root (IDEA1 writable root).",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name.",
    )
    parser.add_argument(
        "--fold", type=str, default="fold_0",
        help="Fold name (default: fold_0).",
    )
    parser.add_argument(
        "--method", type=str, default=METHOD_DEFAULT,
        help=f"Method identifier (default: {METHOD_DEFAULT}).",
    )
    parser.add_argument(
        "--round_tag", type=str, required=True,
        help="Round tag, e.g. r00_full5.",
    )
    parser.add_argument(
        "--medsam_ft_root", type=Path, default=None,
        help="MedSAM fine-tuning root (enables training output checks).",
    )
    parser.add_argument(
        "--is_3d", action="store_true",
        help="Dataset is 3D (enables 3D-specific checks).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Treat any check failure as a hard error (non-zero exit).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    # Build paths
    p = build_fold_paths(
        processed_root=args.processed_root,
        dataset=args.dataset,
        fold=args.fold,
        method=args.method,
        round_tag=args.round_tag,
        medsam_ft_root=args.medsam_ft_root,
    )

    # Build extended path dict (matching round_output_paths in runner)
    meta_dir = p["meta_dir"]
    run_id = p["run_id"]
    paths: dict[str, Path] = {
        **p,
        "round_summary_path": p["round_summary_path"],
        "feature_audit": meta_dir / f"feature_extraction_audit_{run_id}.json",
        "pseudo_audit": meta_dir / f"pseudo_generation_output_audit_{run_id}.json",
        "pseudo_config": meta_dir / f"pseudo_generation_config_{run_id}.json",
        "pseudo_stats": meta_dir / f"pseudo_quality_stats_{run_id}.csv",
        "hard_slice_ranking_3d": meta_dir / f"hard_slice_ranking_{run_id}.csv",
        "hard_case_ranking_3d": meta_dir / f"hard_case_ranking_{run_id}.csv",
        "method_selection": meta_dir / f"full_selection_{run_id}.json",
    }

    print(f"\nValidating round outputs:")
    print(f"  dataset   = {args.dataset}")
    print(f"  fold      = {args.fold}")
    print(f"  method    = {args.method}")
    print(f"  round_tag = {args.round_tag}")
    print(f"  run_id    = {run_id}")
    print(f"  is_3d     = {args.is_3d}")
    print()

    all_errors: list[str] = []

    # ---- Split ----
    print("--- Split ---")
    all_errors.extend(validate_split(paths, args.strict))

    # ---- Template ----
    print("\n--- Template ---")
    all_errors.extend(validate_template(paths, args.strict))

    # ---- Train (only if medsam_ft_root provided) ----
    if args.medsam_ft_root is not None:
        print("\n--- Train ---")
        all_errors.extend(validate_train(paths, args.strict))

    # ---- Pseudo ----
    print("\n--- Pseudo ---")
    all_errors.extend(validate_pseudo(paths))

    # ---- Select ----
    print("\n--- Select ---")
    all_errors.extend(validate_select(paths, args.is_3d, args.round_tag))

    # ---- Round Summary ----
    print("\n--- Round Summary ---")
    all_errors.extend(validate_round_summary(paths, args.is_3d, args.round_tag))

    # ---- Stage success markers ----
    print("\n--- Stage Success Markers ---")
    stage_dir = p.get("stage_success_dir")
    if stage_dir and stage_dir.is_dir():
        for stage_file in sorted(stage_dir.iterdir()):
            if stage_file.suffix == ".json":
                try:
                    data = _load_json(stage_file)
                    status = data.get("status", "unknown")
                    _ok(f"{stage_file.stem}: status={status}")
                except Exception as exc:
                    all_errors.append(f"Stage marker {stage_file.name}: {exc}")
                    _fail(f"{stage_file.stem}: invalid JSON")
    else:
        _ok("Stage success dir: (not present)")

    # ---- Summary ----
    print()
    if all_errors:
        print(f"FAILURES: {len(all_errors)}")
        for err in all_errors:
            print(f"  - {err}")
        if args.strict:
            return min(len(all_errors), 255)
        else:
            print("(non-strict mode — exit 0 despite failures)")
            return 0
    else:
        print("All checks passed.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
