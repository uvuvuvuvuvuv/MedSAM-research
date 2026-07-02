"""Common pipeline utilities for idea1_iterclean_bank_adaptshape.

Path construction, stage-success markers, git-dirty detection,
writable-root protection, and atomic JSON write.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_DEFAULT: str = "idea1_iterclean_bank_adaptshape"

KNOWN_3D_DATASETS: set[str] = {
    "btcv",
    "synapse",
    "acdc",
    "prostate158",
}

PLAN_2D: list[int] = [5, 10, 15, 20]
PLAN_3D: list[int] = [1, 2, 3, 4, 5]

FROZEN_BASELINE_ROOT: Path = Path(
    "/storage/baiyuting/data/MedSAM-main/data/processed"
)
IDEA1_WRITABLE_ROOT: Path = Path(
    "/storage/baiyuting/data/out_data_idea1"
)

_VALID_STAGES: tuple[str, ...] = (
    "split",
    "support",
    "train",
    "pseudo",
    "select",
    "finalize",
)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def is_3d_dataset(dataset: str) -> bool:
    """Return True if *dataset* is a known 3D dataset."""
    return dataset.strip().lower() in KNOWN_3D_DATASETS


# ---------------------------------------------------------------------------
# Round tag and run ID
# ---------------------------------------------------------------------------

def get_round_tag(round_id: int, is_3d: bool) -> str:
    """Return a round tag like 'r0_full5' for 2D or 'r0_full1' for 3D."""
    plan = PLAN_3D if is_3d else PLAN_2D
    if round_id < 0 or round_id >= len(plan):
        dim_label = "3D" if is_3d else "2D"
        raise IndexError(
            f"round_id {round_id} out of range for {dim_label} plan "
            f"(max {len(plan) - 1})"
        )
    full_count = plan[round_id]
    if is_3d:
        return f"r{round_id:02d}_case{full_count}"
    else:
        return f"r{round_id:02d}_full{full_count}"


def make_run_id(method: str, round_tag: str) -> str:
    """Return a run identifier like 'idea1_..._r0_full5'."""
    return f"{method}_{round_tag}"


# ---------------------------------------------------------------------------
# Path construction
# ---------------------------------------------------------------------------

def build_fold_paths(
    processed_root: Union[str, Path],
    dataset: str,
    fold: str,
    method: str,
    round_tag: str,
    medsam_ft_root: Union[str, Path, None] = None,
) -> dict[str, Any]:
    """Build all paths for a given (dataset, fold, method, round_tag).

    All round-specific artifacts use *run_id* so that different rounds
    never overwrite each other.
    """
    run_id = make_run_id(method, round_tag)

    processed_root = Path(processed_root)
    fold_root = processed_root / dataset / fold
    meta_dir = fold_root / "meta"
    stage_success_dir = meta_dir / "stage_success" / run_id

    paths: dict[str, Any] = {
        # Identity
        "method": method,
        "round_tag": round_tag,
        "run_id": run_id,
        # Roots
        "fold_root": fold_root,
        "meta_dir": meta_dir,
        # Split
        "split_path": meta_dir / f"full_box_split_{run_id}.json",
        "split_summary_path": meta_dir / f"full_box_split_summary_{run_id}.json",
        # Support
        "support_path": meta_dir / f"support_bank_adaptshape_{run_id}.npz",
        "support_stats_path": meta_dir / f"support_bank_adaptshape_stats_{run_id}.json",
        # Pseudo
        "pseudo_teacher_dir": fold_root / "pseudo_teacher" / f"tri_train_{run_id}",
        "pseudo_student_dir": fold_root / "pseudo_student" / f"tri_train_{run_id}",
        # Hard selection
        "hard_selection_path": meta_dir / f"hard_selection_{run_id}.json",
        "hard_summary_path": meta_dir / f"hard_selection_summary_{run_id}.json",
        # Round summary
        "round_summary_path": meta_dir / f"round_summary_{run_id}.json",
        # Stage success
        "stage_success_dir": stage_success_dir,
    }

    # Training paths (only when medsam_ft_root is provided)
    if medsam_ft_root is not None:
        train_root = Path(medsam_ft_root) / method / dataset / fold / round_tag
        paths["train_root"] = train_root
        paths["train_last_path"] = train_root / "medsam_sac_last.pth"
        paths["train_ema_path"] = train_root / "medsam_sac_ema.pth"
        paths["train_log_path"] = train_root / f"medsam_ft_log_{run_id}.csv"
        paths["train_summary_path"] = train_root / f"medsam_ft_summary_{run_id}.json"
    else:
        for k in (
            "train_root",
            "train_last_path",
            "train_ema_path",
            "train_log_path",
            "train_summary_path",
        ):
            paths[k] = None

    return paths


# ---------------------------------------------------------------------------
# Stage success markers
# ---------------------------------------------------------------------------

def stage_success_path(
    stage_success_dir: Union[str, Path],
    stage: str,
) -> Path:
    """Return the marker file path for a specific stage."""
    if stage not in _VALID_STAGES:
        raise ValueError(
            f"Unknown stage {stage!r}. Valid: {_VALID_STAGES}"
        )
    return Path(stage_success_dir) / f"{stage}.json"


def write_stage_success(
    stage_success_dir: Union[str, Path],
    stage: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Write a stage-success marker file atomically."""
    target = stage_success_path(stage_success_dir, stage)
    payload: dict[str, Any] = {
        "stage": stage,
        "status": "success",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if metadata is not None:
        payload["metadata"] = metadata
    save_json_atomic(payload, target)


def read_stage_success(
    stage_success_dir: Union[str, Path],
    stage: str,
) -> dict[str, Any]:
    """Read a stage-success marker file."""
    target = stage_success_path(stage_success_dir, stage)
    with open(target, "r", encoding="utf-8") as fh:
        return json.load(fh)


def has_stage_succeeded(
    stage_success_dir: Union[str, Path],
    stage: str,
) -> bool:
    """Return True if the stage has a success marker on disk."""
    target = stage_success_path(stage_success_dir, stage)
    if not target.is_file():
        return False
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
        return data.get("status") == "success"
    except (json.JSONDecodeError, KeyError):
        return False


# ---------------------------------------------------------------------------
# Git dirty detection
# ---------------------------------------------------------------------------

def check_git_dirty(
    repo_path: Optional[Union[str, Path]] = None,
) -> dict[str, Any]:
    """Check whether the git repository has uncommitted changes.

    Uses ``git status --porcelain`` so that unstaged modifications,
    staged modifications, and untracked files all count as *dirty*.
    """
    if repo_path is None:
        repo_path = Path(__file__).resolve().parent
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return {"dirty": True, "status_porcelain": ""}

    porcelain = result.stdout.rstrip("\n")
    return {
        "dirty": bool(porcelain.strip()),
        "status_porcelain": porcelain,
    }


# ---------------------------------------------------------------------------
# Writable path validation
# ---------------------------------------------------------------------------

def validate_writable_path(
    path: Union[str, Path],
    allowed_root: Optional[Union[str, Path]] = None,
) -> None:
    """Raise RuntimeError if *path* is not safe to write to.

    Rules (checked in order):

    1. Must NOT reside under the frozen baseline root.
    2. Must reside under ``IDEA1_WRITABLE_ROOT`` **or** the caller-supplied
       *allowed_root*.
    """
    resolved = Path(path).resolve()
    frozen = FROZEN_BASELINE_ROOT.resolve()

    # Rule 1 — reject frozen baseline
    if resolved == frozen or frozen in resolved.parents:
        raise RuntimeError(
            f"Path resolves under frozen baseline root ({frozen}): {resolved}"
        )

    # Rule 2a — accept idea1 writable root
    idea1 = IDEA1_WRITABLE_ROOT.resolve()
    if resolved == idea1 or idea1 in resolved.parents:
        return

    # Rule 2b — accept caller-supplied allowed root
    if allowed_root is not None:
        allowed = Path(allowed_root).resolve()
        if resolved == allowed or allowed in resolved.parents:
            return

    raise RuntimeError(
        f"Path is not under IDEA1_WRITABLE_ROOT ({idea1}) "
        f"or allowed_root ({allowed_root!r}): {resolved}"
    )


# ---------------------------------------------------------------------------
# Atomic JSON I/O
# ---------------------------------------------------------------------------

def save_json_atomic(
    data: Any,
    target_path: Union[str, Path],
) -> None:
    """Atomically write JSON *data* to *target_path*.

    Uses a unique temporary file in the same directory as the target so
    that the final ``os.replace`` is atomic on the same filesystem.
    The temporary file is cleaned up on failure.
    """
    target = Path(target_path)
    target_dir = str(target.parent)
    os.makedirs(target_dir, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=target_dir,
        prefix=".tmp-",
        suffix=".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.replace(tmp_name, str(target))
    except Exception:
        # Best-effort cleanup of the temporary file
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _failures: list[str] = []

    def _check(name: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"  PASS  {name}")
        else:
            msg = f"  FAIL  {name}"
            if detail:
                msg += f"  --  {detail}"
            print(msg)
            _failures.append(name)

    # ----------------------------------------------------------------
    # 1. get_round_tag — 2D
    # ----------------------------------------------------------------
    print("=== get_round_tag 2D ===")

    _check("r00_full5", get_round_tag(0, False) == "r00_full5")
    _check("r01_full10", get_round_tag(1, False) == "r01_full10")
    _check("r02_full15", get_round_tag(2, False) == "r02_full15")
    _check("r03_full20", get_round_tag(3, False) == "r03_full20")

    try:
        get_round_tag(-1, False)
        _check("negative round_id raises IndexError", False, "no exception")
    except IndexError:
        _check("negative round_id raises IndexError", True)

    try:
        get_round_tag(4, False)
        _check("round_id=4 raises IndexError (2D)", False, "no exception")
    except IndexError:
        _check("round_id=4 raises IndexError (2D)", True)

    # ----------------------------------------------------------------
    # 2. get_round_tag — 3D
    # ----------------------------------------------------------------
    print("=== get_round_tag 3D ===")

    _check("r00_case1", get_round_tag(0, True) == "r00_case1")
    _check("r01_case2", get_round_tag(1, True) == "r01_case2")
    _check("r02_case3", get_round_tag(2, True) == "r02_case3")
    _check("r03_case4", get_round_tag(3, True) == "r03_case4")
    _check("r04_case5", get_round_tag(4, True) == "r04_case5")

    try:
        get_round_tag(-1, True)
        _check("negative round_id raises IndexError (3D)", False, "no exception")
    except IndexError:
        _check("negative round_id raises IndexError (3D)", True)

    try:
        get_round_tag(5, True)
        _check("round_id=5 raises IndexError (3D)", False, "no exception")
    except IndexError:
        _check("round_id=5 raises IndexError (3D)", True)

    # ----------------------------------------------------------------
    # 3. is_3d_dataset
    # ----------------------------------------------------------------
    print("=== is_3d_dataset ===")

    _check("BTCV is 3D", is_3d_dataset("BTCV") is True)
    _check("synapse is 3D", is_3d_dataset("synapse") is True)
    _check("acdc is 3D", is_3d_dataset("acdc") is True)
    _check("prostate158 is 3D", is_3d_dataset("prostate158") is True)
    _check("cvc_clinicdb is not 3D", is_3d_dataset("cvc_clinicdb") is False)
    _check("empty string is not 3D", is_3d_dataset("") is False)
    _check("whitespace-only is not 3D", is_3d_dataset("   ") is False)
    _check("mixed case btcv", is_3d_dataset("  BTCV  ") is True)

    # ----------------------------------------------------------------
    # 4. make_run_id
    # ----------------------------------------------------------------
    print("=== make_run_id ===")

    rid = make_run_id("idea1_iterclean_bank_adaptshape", "r00_full5")
    _check("basic", rid == "idea1_iterclean_bank_adaptshape_r00_full5")
    _check("round_tag in run_id", "r00_full5" in rid)
    _check("method in run_id", "idea1_iterclean_bank_adaptshape" in rid)

    # ----------------------------------------------------------------
    # 5. build_fold_paths — keys and basic shapes
    # ----------------------------------------------------------------
    print("=== build_fold_paths ===")

    p = build_fold_paths(
        processed_root="/fake/processed",
        dataset="btcv",
        fold="fold_0",
        method="test_method",
        round_tag="r00_full5",
    )

    required_keys = [
        "method", "round_tag", "run_id",
        "fold_root", "meta_dir",
        "split_path", "split_summary_path",
        "support_path", "support_stats_path",
        "pseudo_teacher_dir", "pseudo_student_dir",
        "hard_selection_path", "hard_summary_path",
        "round_summary_path", "stage_success_dir",
        "train_root", "train_last_path", "train_ema_path",
        "train_log_path", "train_summary_path",
    ]
    for k in required_keys:
        _check(f"key '{k}' present", k in p)

    _check("run_id equals make_run_id", p["run_id"] == make_run_id("test_method", "r00_full5"))

    # All round-artifact paths must embed run_id
    artifact_keys = [
        "split_path", "split_summary_path",
        "support_path", "support_stats_path",
        "pseudo_teacher_dir", "pseudo_student_dir",
        "hard_selection_path", "hard_summary_path",
        "round_summary_path",
    ]
    for k in artifact_keys:
        _check(
            f"{k} contains run_id",
            p["run_id"] in str(p[k]),
            f"got {p[k]}",
        )

    # --- support file naming ---
    _check(
        "support_path uses support_bank_adaptshape_ prefix",
        p["support_path"].name.startswith("support_bank_adaptshape_")
        and p["support_path"].name.endswith(".npz"),
        f"got {p['support_path'].name}",
    )
    _check(
        "support_stats_path uses support_bank_adaptshape_stats_ prefix",
        p["support_stats_path"].name.startswith("support_bank_adaptshape_stats_")
        and p["support_stats_path"].name.endswith(".json"),
        f"got {p['support_stats_path'].name}",
    )

    # ----------------------------------------------------------------
    # 6. Two round tags → different paths
    # ----------------------------------------------------------------
    print("=== round isolation ===")

    p0 = build_fold_paths(
        processed_root="/fake/processed",
        dataset="btcv",
        fold="fold_0",
        method="test_method",
        round_tag="r00_full5",
        medsam_ft_root="/fake/ft",
    )
    p1 = build_fold_paths(
        processed_root="/fake/processed",
        dataset="btcv",
        fold="fold_0",
        method="test_method",
        round_tag="r01_full10",
        medsam_ft_root="/fake/ft",
    )

    diff_keys = [
        "split_path", "support_path",
        "pseudo_teacher_dir", "pseudo_student_dir",
        "train_root", "train_last_path",
        "train_log_path", "round_summary_path",
    ]
    for k in diff_keys:
        _check(
            f"{k} differs between rounds",
            str(p0[k]) != str(p1[k]),
            f"both = {p0[k]}",
        )

    # ----------------------------------------------------------------
    # 7. Training paths — None without medsam_ft_root
    # ----------------------------------------------------------------
    print("=== training paths None ===")

    p_none = build_fold_paths(
        processed_root="/fake/processed",
        dataset="btcv",
        fold="fold_0",
        method="test_method",
        round_tag="r00_full5",
    )
    for k in ("train_root", "train_last_path", "train_ema_path",
              "train_log_path", "train_summary_path"):
        _check(f"{k} is None", p_none[k] is None, f"got {p_none[k]!r}")

    # ----------------------------------------------------------------
    # 8. Training paths present with medsam_ft_root
    # ----------------------------------------------------------------
    print("=== training paths present ===")

    p_ft = build_fold_paths(
        processed_root="/fake/processed",
        dataset="btcv",
        fold="fold_0",
        method="test_method",
        round_tag="r00_full5",
        medsam_ft_root="/fake/ft",
    )
    for k in ("train_root", "train_last_path", "train_ema_path",
              "train_log_path", "train_summary_path"):
        _check(f"{k} is not None", p_ft[k] is not None)
        _check(f"{k} is Path", isinstance(p_ft[k], Path))

    _check("train_root ends with round_tag",
           str(p_ft["train_root"]).endswith("r00_full5"))

    _check("train_log_path contains run_id",
           p_ft["run_id"] in str(p_ft["train_log_path"]))

    # train_root contains method/dataset/fold/round_tag hierarchy
    tr = str(p_ft["train_root"])
    _check("train_root contains method", "test_method" in tr)
    _check("train_root contains dataset", "btcv" in tr)
    _check("train_root contains fold", "fold_0" in tr)
    _check("train_root contains round_tag", "r00_full5" in tr)

    # ----------------------------------------------------------------
    # 8. validate_writable_path — frozen rejected
    # ----------------------------------------------------------------
    print("=== validate_writable_path ===")

    try:
        validate_writable_path(str(FROZEN_BASELINE_ROOT / "btcv" / "fold_0"))
        _check("frozen baseline rejected", False, "no exception")
    except RuntimeError:
        _check("frozen baseline rejected", True)

    # Exact frozen root itself
    try:
        validate_writable_path(str(FROZEN_BASELINE_ROOT))
        _check("frozen root itself rejected", False, "no exception")
    except RuntimeError:
        _check("frozen root itself rejected", True)

    # ----------------------------------------------------------------
    # 9. validate_writable_path — idea1 allowed
    # ----------------------------------------------------------------
    try:
        validate_writable_path(str(IDEA1_WRITABLE_ROOT / "some" / "output"))
        _check("idea1 path allowed", True)
    except RuntimeError as exc:
        _check("idea1 path allowed", False, str(exc))

    # Exact idea1 root itself
    try:
        validate_writable_path(str(IDEA1_WRITABLE_ROOT))
        _check("idea1 root itself allowed", True)
    except RuntimeError as exc:
        _check("idea1 root itself allowed", False, str(exc))

    # ----------------------------------------------------------------
    # 10. validate_writable_path — allowed_root
    # ----------------------------------------------------------------
    try:
        validate_writable_path(
            "/tmp/test_pipeline_allowed/subdir/file.json",
            allowed_root="/tmp/test_pipeline_allowed",
        )
        _check("allowed_root accepts sub-path", True)
    except RuntimeError as exc:
        _check("allowed_root accepts sub-path", False, str(exc))

    # ----------------------------------------------------------------
    # 11. validate_writable_path — source repo rejected
    # ----------------------------------------------------------------
    source_dir = str(Path(__file__).resolve().parent)
    try:
        validate_writable_path(source_dir)
        _check("source repo path rejected", False, f"accepted {source_dir}")
    except RuntimeError:
        _check("source repo path rejected", True)

    # ----------------------------------------------------------------
    # 12. stage_success_path — every stage has own file
    # ----------------------------------------------------------------
    print("=== stage success markers ===")

    stage_dir = "/fake/meta/stage_success/run_x"
    stage_paths = [
        str(stage_success_path(stage_dir, s))
        for s in _VALID_STAGES
    ]
    _check("6 stages = 6 unique paths", len(set(stage_paths)) == len(_VALID_STAGES))

    for s in _VALID_STAGES:
        sp = stage_success_path(stage_dir, s)
        _check(f"stage '{s}' path ends with {s}.json",
               sp.name == f"{s}.json")

    # invalid stage
    try:
        stage_success_path(stage_dir, "nonexistent")
        _check("invalid stage raises ValueError", False, "no exception")
    except ValueError:
        _check("invalid stage raises ValueError", True)

    # ----------------------------------------------------------------
    # 13. stage_success roundtrip
    # ----------------------------------------------------------------
    tmp_stage_dir = tempfile.mkdtemp(prefix="pipeline_test_stage_")
    try:
        _check("has_stage_succeeded false before write",
               not has_stage_succeeded(tmp_stage_dir, "train"))

        write_stage_success(tmp_stage_dir, "train",
                            metadata={"epochs": 1, "global_step": 100})

        _check("has_stage_succeeded true after write",
               has_stage_succeeded(tmp_stage_dir, "train"))

        data = read_stage_success(tmp_stage_dir, "train")
        _check("read status == success", data.get("status") == "success")
        _check("read stage == train", data.get("stage") == "train")
        _check("read metadata.epochs == 1",
               data.get("metadata", {}).get("epochs") == 1)

        # Non-existent stage
        _check("has_stage_succeeded false for missing stage",
               not has_stage_succeeded(tmp_stage_dir, "split"))
    finally:
        import shutil
        shutil.rmtree(tmp_stage_dir, ignore_errors=True)

    # ----------------------------------------------------------------
    # 14. Atomic JSON roundtrip
    # ----------------------------------------------------------------
    print("=== atomic JSON ===")

    tmp_json_dir = tempfile.mkdtemp(prefix="pipeline_test_json_")
    try:
        target = os.path.join(tmp_json_dir, "data.json")
        payload = {
            "hello": "world",
            "unicode": "中文测试",
            "nested": [1, 2, 3],
            "float": 3.14,
        }
        save_json_atomic(payload, target)

        _check("file exists after write", os.path.isfile(target))

        with open(target, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        _check("roundtrip matches", loaded == payload)

        # Write again (overwrite) — must still work
        payload2 = {"overwritten": True}
        save_json_atomic(payload2, target)
        with open(target, "r", encoding="utf-8") as fh:
            loaded2 = json.load(fh)
        _check("overwrite roundtrip", loaded2 == payload2)
    finally:
        import shutil
        shutil.rmtree(tmp_json_dir, ignore_errors=True)

    # ----------------------------------------------------------------
    # 15. Git dirty detection
    # ----------------------------------------------------------------
    print("=== git dirty ===")

    info = check_git_dirty()
    _check("returns dict with 'dirty'", "dirty" in info)
    _check("returns dict with 'status_porcelain'", "status_porcelain" in info)
    _check("dirty is bool", isinstance(info["dirty"], bool))

    # pipeline_common.py is untracked (?? in git status), so dirty must be True
    _check("dirty is True (untracked file exists)",
           info["dirty"] is True,
           f"dirty={info['dirty']}, porcelain={info['status_porcelain']!r}")

    _check("porcelain mentions pipeline_common.py",
           "pipeline_common.py" in info["status_porcelain"])

    # ----------------------------------------------------------------
    # Report
    # ----------------------------------------------------------------
    print()
    if _failures:
        print(f"{len(_failures)} FAILURE(S):")
        for f in _failures:
            print(f"  - {f}")
        sys.exit(1)
    else:
        print("All tests passed.")
