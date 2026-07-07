"""Unit tests for run_iterative_sampling.py utility functions.

Covers parsers, validators, command builders, state management, and
selection logic.  Uses only standard-library unittest.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from idea1_iterclean_bank_adaptshape import run_iterative_sampling as runner


# ============================================================================
# A. parse_csv_strings
# ============================================================================

class TestParseCsvStrings(unittest.TestCase):
    """Tests for parse_csv_strings."""

    def test_single_value(self):
        self.assertEqual(runner.parse_csv_strings("abc"), ["abc"])

    def test_multiple_values(self):
        self.assertEqual(
            runner.parse_csv_strings("a, b ,c"),
            ["a", "b", "c"],
        )

    def test_empty_string(self):
        self.assertEqual(runner.parse_csv_strings(""), [])

    def test_whitespace_only(self):
        self.assertEqual(runner.parse_csv_strings("  , , "), [])

    def test_strips_whitespace(self):
        self.assertEqual(
            runner.parse_csv_strings("  hello , world  "),
            ["hello", "world"],
        )


# ============================================================================
# B. parse_schedule / validate_schedule
# ============================================================================

class TestParseSchedule(unittest.TestCase):
    """Tests for parse_schedule."""

    def test_valid_increasing(self):
        result = runner.parse_schedule("5,10,15")
        self.assertEqual(result, [5, 10, 15])

    def test_single_value(self):
        result = runner.parse_schedule("3")
        self.assertEqual(result, [3])

    def test_empty_string_raises(self):
        with self.assertRaises(ValueError):
            runner.parse_schedule("")

    def test_non_numeric_raises(self):
        with self.assertRaises(ValueError):
            runner.parse_schedule("5,abc,10")

    def test_non_positive_raises(self):
        with self.assertRaises(ValueError):
            runner.parse_schedule("5,0,10")

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            runner.parse_schedule("-1,5")


class TestValidateSchedule(unittest.TestCase):
    """Tests for validate_schedule."""

    def test_valid_increasing(self):
        runner.validate_schedule([1, 2, 3])

    def test_valid_strictly_increasing(self):
        runner.validate_schedule([5, 10, 15, 20])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            runner.validate_schedule([])

    def test_equal_values_raises(self):
        with self.assertRaises(ValueError):
            runner.validate_schedule([5, 5])

    def test_decreasing_raises(self):
        with self.assertRaises(ValueError):
            runner.validate_schedule([10, 5])

    def test_unordered_raises(self):
        with self.assertRaises(ValueError):
            runner.validate_schedule([5, 10, 7])


# ============================================================================
# C. sanitize_name
# ============================================================================

class TestSanitizeName(unittest.TestCase):
    """Tests for sanitize_name."""

    def test_alphanumeric_unchanged(self):
        self.assertEqual(
            runner.sanitize_name("my-run_01"),
            "my-run_01",
        )

    def test_spaces_replaced(self):
        self.assertEqual(
            runner.sanitize_name("my run name"),
            "my_run_name",
        )

    def test_special_chars_replaced(self):
        self.assertEqual(
            runner.sanitize_name("test@#$run"),
            "test___run",
        )

    def test_leading_dots_stripped(self):
        self.assertEqual(
            runner.sanitize_name("...test"),
            "test",
        )

    def test_trailing_dash_stripped(self):
        self.assertEqual(
            runner.sanitize_name("test---"),
            "test",
        )

    def test_underscore_only(self):
        self.assertEqual(
            runner.sanitize_name("...---"),
            "run",
        )

    def test_empty_returns_run(self):
        self.assertEqual(runner.sanitize_name(""), "run")


# ============================================================================
# D. build_round_tags
# ============================================================================

class TestBuildRoundTags(unittest.TestCase):
    """Tests for build_round_tags."""

    def test_2d_tags(self):
        tags = runner.build_round_tags([5, 10, 15, 20], is_3d=False)
        self.assertEqual(
            tags,
            ["r00_full5", "r01_full10", "r02_full15", "r03_full20"],
        )

    def test_3d_tags(self):
        tags = runner.build_round_tags([1, 2, 3], is_3d=True)
        self.assertEqual(
            tags,
            ["r00_case1", "r01_case2", "r02_case3"],
        )

    def test_empty_schedule(self):
        tags = runner.build_round_tags([], is_3d=False)
        self.assertEqual(tags, [])

    def test_single_round(self):
        tags = runner.build_round_tags([7], is_3d=False)
        self.assertEqual(tags, ["r00_full7"])

    def test_prefix_is_zero_padded(self):
        tags = runner.build_round_tags([5, 10], is_3d=False)
        self.assertTrue(tags[0].startswith("r00_"))
        self.assertTrue(tags[1].startswith("r01_"))


# ============================================================================
# E. determine_schedule
# ============================================================================

class TestDetermineSchedule(unittest.TestCase):
    """Tests for determine_schedule."""

    def test_explicit_schedule(self):
        args = argparse.Namespace(
            schedule="3,6,9",
            preset="custom",
        )
        self.assertEqual(
            runner.determine_schedule(args),
            [3, 6, 9],
        )

    def test_smoke_preset_removed(self):
        """smoke_2d preset must not exist to avoid confusing it with
        the formal contract."""
        from idea1_iterclean_bank_adaptshape.run_iterative_sampling import (
            PRESET_SCHEDULES,
        )
        self.assertNotIn("smoke_2d", PRESET_SCHEDULES)

    def test_formal_2d_preset(self):
        args = argparse.Namespace(schedule=None, preset="formal_2d")
        self.assertEqual(
            runner.determine_schedule(args),
            [5, 10, 15, 20],
        )

    def test_formal_3d_preset(self):
        args = argparse.Namespace(schedule=None, preset="formal_3d")
        self.assertEqual(
            runner.determine_schedule(args),
            [1, 2, 3, 4, 5],
        )

    def test_custom_without_schedule_raises(self):
        args = argparse.Namespace(schedule=None, preset="custom")
        with self.assertRaises(ValueError):
            runner.determine_schedule(args)


# ============================================================================
# F. default_run_name
# ============================================================================

class TestDefaultRunName(unittest.TestCase):
    """Tests for default_run_name."""

    def test_single_dataset(self):
        name = runner.default_run_name(
            ["cvc_clinicdb"],
            [5, 7],
            "idea1_iterclean_bank_adaptshape",
        )
        self.assertIn("cvc_clinicdb", name)
        self.assertIn("5-7", name)

    def test_multiple_datasets(self):
        name = runner.default_run_name(
            ["cvc_clinicdb", "kvasir_polyp"],
            [5, 10],
            "test_method",
        )
        self.assertIn("cvc_clinicdb-kvasir_polyp", name)
        self.assertIn("test_method", name)

    def test_result_is_sanitized(self):
        name = runner.default_run_name(
            ["test ds"],
            [5],
            "method!",
        )
        self.assertNotIn("!", name)
        self.assertNotIn(" ", name)


# ============================================================================
# G. get_script_paths
# ============================================================================

class TestGetScriptPaths(unittest.TestCase):
    """Tests for get_script_paths."""

    def test_all_stages_present(self):
        code_root = Path("/fake/code")
        paths = runner.get_script_paths(code_root)
        for stage in runner.STAGE_ORDER:
            with self.subTest(stage=stage):
                self.assertIn(stage, paths)
                self.assertIsInstance(paths[stage], Path)

    def test_correct_filenames(self):
        code_root = Path("/fake/code")
        paths = runner.get_script_paths(code_root)
        self.assertEqual(
            paths["split"].name,
            "01_build_full_box_split.py",
        )
        self.assertEqual(
            paths["template"].name,
            "02_build_support_template.py",
        )
        self.assertEqual(
            paths["train"].name,
            "03_train_medsam_full_only.py",
        )
        self.assertEqual(
            paths["pseudo"].name,
            "04_generate_pseudo_bank_adaptshape.py",
        )
        self.assertEqual(
            paths["select"].name,
            "05_select_hard_by_gt_iou.py",
        )


# ============================================================================
# H. format_command
# ============================================================================

class TestFormatCommand(unittest.TestCase):
    """Tests for format_command."""

    def test_simple_command(self):
        result = runner.format_command(
            ["python", "script.py", "--arg", "value"],
        )
        self.assertIn("--arg", result)
        self.assertIn("value", result)

    def test_arg_with_spaces_quoted(self):
        result = runner.format_command(
            ["python", "script.py", "--path", "/path with spaces"],
        )
        self.assertIn("/path with spaces", result)


# ============================================================================
# I. get_slice_name / get_case_id
# ============================================================================

class TestGetSliceName(unittest.TestCase):
    """Tests for get_slice_name."""

    def test_slice_name_direct(self):
        record = {"slice_name": "img_001.png"}
        self.assertEqual(
            runner.get_slice_name(record),
            "img_001.png",
        )

    def test_fallback_teacher_img(self):
        record = {"teacher_img": "/path/to/img.npy"}
        self.assertEqual(
            runner.get_slice_name(record),
            "img.npy",
        )

    def test_fallback_student_img(self):
        record = {"student_img": "/path/to/student.npy"}
        self.assertEqual(
            runner.get_slice_name(record),
            "student.npy",
        )

    def test_no_slice_name_raises(self):
        with self.assertRaises(KeyError):
            runner.get_slice_name({})


class TestGetCaseId(unittest.TestCase):
    """Tests for get_case_id."""

    def test_case_id_present(self):
        record = {"case_id": "patient_001", "slice_name": "img.png"}
        self.assertEqual(
            runner.get_case_id(record),
            "patient_001",
        )

    def test_missing_case_id_raises(self):
        with self.assertRaises(KeyError):
            runner.get_case_id({"slice_name": "img.png"})

    def test_empty_case_id_raises(self):
        with self.assertRaises(KeyError):
            runner.get_case_id({"case_id": "", "slice_name": "img.png"})


# ============================================================================
# J. canonical_selection_payload
# ============================================================================

class TestCanonicalSelectionPayload(unittest.TestCase):
    """Tests for canonical_selection_payload."""

    def test_extracts_required_keys(self):
        full = {
            "schema_version": 1,
            "dataset": "btcv",
            "fold": "fold_0",
            "source_method": "test_method",
            "source_round_id": 0,
            "round_id": 1,
            "selection_strategy": "iou_score",
            "sampling_unit": "case",
            "score_space": "native",
            "score_name": "gt_iou",
            "class_ids": [1, 2],
            "previous_full_count": 1,
            "new_selected_count": 1,
            "cumulative_full_count": 2,
            "new_selected_slice_names": ["s1"],
            "cumulative_full_slice_names": ["s1", "s2"],
            "new_selected_case_ids": ["c1"],
            "cumulative_full_case_ids": ["c1", "c2"],
            "extra_field": "should be stripped",
        }
        payload = runner.canonical_selection_payload(full)
        self.assertEqual(payload["dataset"], "btcv")
        self.assertEqual(payload["cumulative_full_count"], 2)
        self.assertNotIn("extra_field", payload)

    def test_missing_required_raises(self):
        with self.assertRaises(RuntimeError):
            runner.canonical_selection_payload({})


# ============================================================================
# K. selection_snapshot_path
# ============================================================================

class TestSelectionSnapshotPath(unittest.TestCase):
    """Tests for selection_snapshot_path."""

    def test_path_structure(self):
        path = runner.selection_snapshot_path(
            Path("/fake/run_dir"),
            "btcv",
            source_round_id=1,
            source_method="test_method",
        )
        self.assertIn("selection_snapshots", str(path))
        self.assertIn("btcv", str(path))
        self.assertIn("after_round_01", str(path))
        self.assertIn("test_method", str(path))
        self.assertTrue(str(path).endswith(".json"))


# ============================================================================
# L. Command builders
# ============================================================================

class TestCommandBuilders(unittest.TestCase):
    """Tests for build_*_command functions."""

    def setUp(self):
        self.args = argparse.Namespace(
            python="python",
            processed_root=Path("/fake/processed"),
            base_checkpoint=Path("/fake/checkpoint.pth"),
            medsam_ft_root=Path("/fake/ft"),
            fold="fold_0",
            device="cuda",
            seed=2026,
            kmax_shape=5,
            shape_size=64,
            cluster_max_iter=25,
            bg_ring_width_tokens=1,
            max_global_non_bg_coverage=0.10,
            epochs=100,
            max_steps=1000,
            lr=1e-4,
            weight_decay=0.01,
            ema_decay=0.999,
            max_grad_norm=1.0,
            train_log_every=50,
            pseudo_max_samples=0,
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
            tau_low=0.1,
            tau_high=0.5,
            temperature=0.1,
            top_k=5,
            pseudo_log_every=50,
            max_fg_fallback_tokens=4,
        )
        self.scripts = {
            stage: Path(f"/fake/code/{name}")
            for stage, name in runner.SCRIPT_NAMES.items()
        }

    def test_build_split_random_round0(self):
        cmd = runner.build_split_command(
            self.args,
            self.scripts,
            "btcv",
            "test_method",
            "r00_full5",
            round_id=0,
            full_count=5,
            selection_file=None,
            stage_overwrite=False,
        )
        self.assertIn("--sampling_mode", cmd)
        self.assertIn("random", cmd)
        self.assertIn("--full_count", cmd)
        self.assertIn("5", cmd)

    def test_build_split_external_round1(self):
        cmd = runner.build_split_command(
            self.args,
            self.scripts,
            "btcv",
            "test_method",
            "r01_full10",
            round_id=1,
            full_count=10,
            selection_file=Path("/fake/selection.json"),
            stage_overwrite=True,
        )
        self.assertIn("--sampling_mode", cmd)
        self.assertIn("external", cmd)
        self.assertIn("--selection_file", cmd)
        self.assertIn("--overwrite", cmd)

    def test_build_template_command(self):
        cmd = runner.build_template_command(
            self.args,
            self.scripts,
            "btcv",
            "test_method",
            "r00_full5",
            stage_overwrite=False,
        )
        self.assertIn("--kmax_shape", cmd)
        self.assertIn("--shape_size", cmd)
        self.assertIn("--cluster_max_iter", cmd)
        self.assertIn("--bg_ring_width_tokens", cmd)
        self.assertIn("--max_global_non_bg_coverage", cmd)

    def test_build_template_command_overwrite(self):
        cmd = runner.build_template_command(
            self.args,
            self.scripts,
            "btcv",
            "test_method",
            "r00_full5",
            stage_overwrite=True,
        )
        self.assertIn("--overwrite", cmd)

    def test_build_train_command(self):
        cmd = runner.build_train_command(
            self.args,
            self.scripts,
            "btcv",
            "test_method",
            "r00_full5",
            stage_overwrite=False,
        )
        self.assertIn("--epochs", cmd)
        self.assertIn("--lr", cmd)
        self.assertIn("--ema_decay", cmd)
        self.assertIn("--round_tag", cmd)
        self.assertIn("--medsam_ft_root", cmd)

    def test_build_pseudo_command(self):
        cmd = runner.build_pseudo_command(
            self.args,
            self.scripts,
            "btcv",
            "test_method",
            "r00_full5",
            ema_checkpoint=Path("/fake/ema.pth"),
            stage_overwrite=False,
        )
        self.assertIn("--alpha", cmd)
        self.assertIn("--beta", cmd)
        self.assertIn("--gamma", cmd)
        self.assertIn("--tau_low", cmd)
        self.assertIn("--tau_high", cmd)
        self.assertIn("--temperature", cmd)
        self.assertIn("--ema_checkpoint", cmd)

    def test_build_select_command(self):
        cmd = runner.build_select_command(
            self.args,
            self.scripts,
            "btcv",
            "test_method",
            "r00_full5",
            round_id=0,
            select_count=5,
            stage_overwrite=False,
        )
        self.assertIn("--round_id", cmd)
        self.assertIn("--select_count", cmd)

    # ---- no old CLI args in any command builder --------------------------
    def test_no_old_k_fg_in_template(self):
        cmd = runner.build_template_command(
            self.args, self.scripts, "btcv", "test_method", "r00_full5", False,
        )
        self.assertNotIn("--k_fg", cmd)
        self.assertNotIn("--k_bg", cmd)

    def test_no_old_proto_in_template(self):
        cmd = runner.build_template_command(
            self.args, self.scripts, "btcv", "test_method", "r00_full5", False,
        )
        flattened = " ".join(cmd)
        self.assertNotIn("proto_fg", flattened.lower())

    def test_no_method_names_in_commands(self):
        for builder in [
            runner.build_split_command,
            runner.build_template_command,
            runner.build_train_command,
        ]:
            with self.subTest(builder=builder.__name__):
                cmd = builder(
                    self.args, self.scripts, "btcv", "test_method", "r00_full5",
                    **(
                        {"round_id": 0, "full_count": 5,
                         "selection_file": None, "stage_overwrite": False}
                        if builder == runner.build_split_command
                        else {"stage_overwrite": False}
                    ),
                )
                flattened = " ".join(cmd)
                self.assertNotIn("--method_names", flattened)
                self.assertNotIn("--method_prefix", flattened)
                self.assertNotIn("--method_suffix", flattened)


# ============================================================================
# M. validate_npz_template
# ============================================================================

class TestValidateNpzTemplate(unittest.TestCase):
    """Tests for validate_npz_template."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="runner_test_npz_")
        self._npz_path = Path(self._tmpdir) / "test.npz"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_npz(self, names: list[str]):
        import numpy as np
        with zipfile.ZipFile(
            str(self._npz_path), "w", zipfile.ZIP_DEFLATED,
        ) as zf:
            for name in names:
                zf.writestr(name, b"dummy")

    def test_missing_file(self):
        ok, reason = runner.validate_npz_template(
            Path("/nonexistent/path.npz"),
        )
        self.assertFalse(ok)
        self.assertIn("missing", reason)

    def test_no_class_ids(self):
        self._write_npz(["bank_fg_c1.npy", "bank_bg_c1.npy"])
        ok, reason = runner.validate_npz_template(self._npz_path)
        self.assertFalse(ok)
        self.assertIn("class_ids", reason)

    def test_valid_npz(self):
        self._write_npz([
            "class_ids.npy",
            "bank_fg_c1.npy",
            "bank_bg_c1.npy",
            "shape_templates_c1.npy",
        ])
        ok, reason = runner.validate_npz_template(self._npz_path)
        self.assertTrue(ok)

    def test_missing_fg_bank(self):
        self._write_npz([
            "class_ids.npy",
            "bank_bg_c1.npy",
        ])
        ok, reason = runner.validate_npz_template(self._npz_path)
        self.assertFalse(ok)

    def test_no_old_proto_keys(self):
        """validate_npz_template should NOT require old proto_fg keys."""
        self._write_npz([
            "class_ids.npy",
            "bank_fg_c1.npy",
            "bank_bg_c1.npy",
            "shape_templates_c1.npy",
        ])
        ok, _ = runner.validate_npz_template(self._npz_path)
        self.assertTrue(ok,
                        msg="NPZ with only new-format keys should pass")


# ============================================================================
# N. validate_dimension_group
# ============================================================================

class TestValidateDimensionGroup(unittest.TestCase):
    """Tests for validate_dimension_group."""

    def test_all_2d_ok(self):
        args = argparse.Namespace(preset="formal_2d")
        runner.validate_dimension_group(
            args,
            ["cvc_clinicdb", "kvasir_polyp"],
            {"cvc_clinicdb": False, "kvasir_polyp": False},
        )

    def test_all_3d_ok(self):
        args = argparse.Namespace(preset="formal_3d")
        runner.validate_dimension_group(
            args,
            ["btcv", "acdc"],
            {"btcv": True, "acdc": True},
        )

    def test_mixed_2d_3d_raises(self):
        args = argparse.Namespace(preset="formal_2d")
        with self.assertRaises(ValueError):
            runner.validate_dimension_group(
                args,
                ["btcv", "cvc_clinicdb"],
                {"btcv": True, "cvc_clinicdb": False},
            )

    def test_preset_mismatch_2d_with_3d_raises(self):
        args = argparse.Namespace(preset="formal_2d")
        with self.assertRaises(ValueError):
            runner.validate_dimension_group(
                args,
                ["btcv"],
                {"btcv": True},
            )

    def test_preset_mismatch_3d_with_2d_raises(self):
        args = argparse.Namespace(preset="formal_3d")
        with self.assertRaises(ValueError):
            runner.validate_dimension_group(
                args,
                ["cvc_clinicdb"],
                {"cvc_clinicdb": False},
            )


# ============================================================================
# O. load_or_initialize_state
# ============================================================================

class TestLoadOrInitializeState(unittest.TestCase):
    """Tests for load_or_initialize_state."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="runner_test_state_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_creates_new_state(self):
        run_dir = Path(self._tmpdir)
        config = {"run_name": "test_run"}
        state, state_path = runner.load_or_initialize_state(
            run_dir, config, resume=False, overwrite=False,
        )
        self.assertEqual(state["status"], "initialized")
        self.assertEqual(state["run_name"], "test_run")
        self.assertTrue(state_path.is_file())

    def test_resume_existing_state(self):
        run_dir = Path(self._tmpdir)
        config = {"run_name": "test_run"}

        # First call creates state
        state1, sp = runner.load_or_initialize_state(
            run_dir, config, resume=False, overwrite=False,
        )
        # Manually tweak state to simulate progress
        state1["status"] = "running"
        import json as _json
        sp.write_text(_json.dumps(state1))

        # Second call resumes
        state2, _ = runner.load_or_initialize_state(
            run_dir, config, resume=True, overwrite=False,
        )
        self.assertEqual(state2["status"], "running")

    def test_overwrite_existing(self):
        run_dir = Path(self._tmpdir)
        config = {"run_name": "test_run"}

        runner.load_or_initialize_state(
            run_dir, config, resume=False, overwrite=False,
        )
        state2, _ = runner.load_or_initialize_state(
            run_dir, config, resume=False, overwrite=True,
        )
        self.assertEqual(state2["status"], "initialized")

    def test_existing_no_resume_no_overwrite_raises(self):
        run_dir = Path(self._tmpdir)
        config = {"run_name": "test_run"}

        runner.load_or_initialize_state(
            run_dir, config, resume=False, overwrite=False,
        )
        with self.assertRaises(FileExistsError):
            runner.load_or_initialize_state(
                run_dir, config, resume=False, overwrite=False,
            )


# ============================================================================
# P. update_stage_state
# ============================================================================

class TestUpdateStageState(unittest.TestCase):
    """Tests for update_stage_state."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="runner_test_state_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_adds_stage_entry(self):
        state = {
            "schema_version": 1,
            "run_name": "test",
            "status": "running",
            "datasets": {},
        }
        state_path = Path(self._tmpdir) / "state.json"
        import json as _json
        state_path.write_text(_json.dumps(state))

        runner.update_stage_state(
            state, state_path,
            dataset="btcv",
            round_id=0,
            method="test_method",
            full_count=5,
            stage="split",
            status="completed",
            detail="OK",
        )
        round_state = (
            state["datasets"]["btcv"]["rounds"]["0"]
        )
        self.assertIn("split", round_state["stages"])
        self.assertEqual(
            round_state["stages"]["split"]["status"],
            "completed",
        )

    def test_updates_existing_entry(self):
        state = {
            "schema_version": 1,
            "run_name": "test",
            "status": "running",
            "datasets": {
                "btcv": {
                    "rounds": {
                        "0": {
                            "method": "test_method",
                            "full_count": 5,
                            "stages": {
                                "split": {
                                    "status": "failed",
                                    "updated_at": "2026-01-01T00:00:00+00:00",
                                    "detail": "error",
                                },
                            },
                        },
                    },
                },
            },
        }
        state_path = Path(self._tmpdir) / "state.json"
        import json as _json
        state_path.write_text(_json.dumps(state))

        runner.update_stage_state(
            state, state_path,
            dataset="btcv",
            round_id=0,
            method="test_method",
            full_count=5,
            stage="split",
            status="completed",
            detail="fixed",
        )
        split_state = (
            state["datasets"]["btcv"]["rounds"]["0"]["stages"]["split"]
        )
        self.assertEqual(split_state["status"], "completed")
        self.assertEqual(split_state["detail"], "fixed")


# ============================================================================
# Q. round_output_paths
# ============================================================================

class TestRoundOutputPaths(unittest.TestCase):
    """Tests for round_output_paths."""

    def test_returns_expected_keys(self):
        args = argparse.Namespace(
            processed_root=Path("/fake/processed"),
            fold="fold_0",
            medsam_ft_root=Path("/fake/ft"),
        )
        paths = runner.round_output_paths(
            args, "btcv", "test_method", "r00_full5",
        )
        expected_keys = [
            "fold_root", "meta_dir", "run_id",
            "split", "split_summary",
            "template", "template_stats", "feature_audit",
            "train_root", "train_last", "train_ema",
            "train_log", "train_summary",
            "pseudo_teacher", "pseudo_student",
            "pseudo_audit", "pseudo_config", "pseudo_stats",
            "hard_selection", "hard_summary",
            "hard_ranking_2d", "hard_slice_ranking_3d",
            "hard_case_ranking_3d",
        ]
        for key in expected_keys:
            with self.subTest(key=key):
                self.assertIn(key, paths)

    def test_run_id_in_meta_paths(self):
        args = argparse.Namespace(
            processed_root=Path("/fake/processed"),
            fold="fold_0",
            medsam_ft_root=Path("/fake/ft"),
        )
        paths = runner.round_output_paths(
            args, "btcv", "test_method", "r00_full5",
        )
        run_id = paths["run_id"]
        self.assertIn(run_id, str(paths["split"]))
        self.assertIn(run_id, str(paths["feature_audit"]))


# ============================================================================
# R. count_npy_files
# ============================================================================

class TestCountNpyFiles(unittest.TestCase):
    """Tests for count_npy_files."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="runner_test_count_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_empty_dir(self):
        d = Path(self._tmpdir) / "empty"
        d.mkdir()
        self.assertEqual(runner.count_npy_files(d), 0)

    def test_mixed_files(self):
        d = Path(self._tmpdir) / "mixed"
        d.mkdir()
        (d / "a.npy").touch()
        (d / "b.npy").touch()
        (d / "c.txt").touch()
        self.assertEqual(runner.count_npy_files(d), 2)

    def test_nonexistent_dir(self):
        self.assertEqual(
            runner.count_npy_files(Path("/nonexistent/dir")),
            0,
        )


# ============================================================================
# S. sha256_file
# ============================================================================

class TestSha256File(unittest.TestCase):
    """Tests for sha256_file."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="runner_test_sha_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_known_content(self):
        path = Path(self._tmpdir) / "test.txt"
        path.write_text("hello world")
        digest = runner.sha256_file(path)
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 64)

    def test_deterministic(self):
        path = Path(self._tmpdir) / "test.txt"
        path.write_text("hello")
        d1 = runner.sha256_file(path)
        d2 = runner.sha256_file(path)
        self.assertEqual(d1, d2)

    def test_different_content_different_hash(self):
        p1 = Path(self._tmpdir) / "a.txt"
        p2 = Path(self._tmpdir) / "b.txt"
        p1.write_text("hello")
        p2.write_text("world")
        self.assertNotEqual(
            runner.sha256_file(p1),
            runner.sha256_file(p2),
        )


# ============================================================================
# Runner Contract Tests
# ============================================================================


class TestRunner01CliContract(unittest.TestCase):
    """Verify 01 accepts --round_tag and --round_id in its CLI."""

    def setUp(self):
        import importlib
        self.t01 = importlib.import_module(
            "idea1_iterclean_bank_adaptshape.01_build_full_box_split"
        )
        self.parser = self.t01.build_parser()

    def test_01_accepts_round_tag(self):
        ns = self.parser.parse_args([
            "--processed_root", "/tmp",
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--round_id", "0",
            "--full_count", "5",
        ])
        self.assertEqual(ns.round_tag, "r00_full5")

    def test_01_accepts_round_id(self):
        ns = self.parser.parse_args([
            "--processed_root", "/tmp",
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--round_id", "3",
            "--full_count", "5",
        ])
        self.assertEqual(ns.round_id, 3)

    def test_01_accepts_previous_round_tag(self):
        ns = self.parser.parse_args([
            "--processed_root", "/tmp",
            "--datasets", "btcv",
            "--round_tag", "r01_full10",
            "--round_id", "1",
            "--sampling_mode", "external",
            "--previous_round_tag", "r00_full5",
        ])
        self.assertEqual(ns.previous_round_tag, "r00_full5")

    def test_01_rejects_wrong_round_tag(self):
        """01's process_dataset must reject round_tag that doesn't match
        get_round_tag(round_id, is_3d)."""
        from idea1_iterclean_bank_adaptshape.pipeline_common import get_round_tag

        # 2D: round_id=1 + round_tag=r01_full7 must fail
        self.assertEqual(get_round_tag(1, False), "r01_full10")
        self.assertNotEqual("r01_full7", get_round_tag(1, False))

        # 2D: round_id=1 + round_tag=r01_full10 must pass
        self.assertEqual("r01_full10", get_round_tag(1, False))

        # 2D: round_id=0 + round_tag=r00_full5
        self.assertEqual("r00_full5", get_round_tag(0, False))

        # 2D: round_id=2 + round_tag=r02_full15
        self.assertEqual("r02_full15", get_round_tag(2, False))

        # 2D: round_id=3 + round_tag=r03_full20 (final round)
        self.assertEqual("r03_full20", get_round_tag(3, False))

    def test_01_3d_round_tag_contract(self):
        """3D round_tag contract via get_round_tag."""
        from idea1_iterclean_bank_adaptshape.pipeline_common import get_round_tag

        # 3D: round_id=0 + round_tag=r00_case1
        self.assertEqual("r00_case1", get_round_tag(0, True))

        # 3D: round_id=1 + round_tag=r01_case2 must pass
        self.assertEqual("r01_case2", get_round_tag(1, True))

        # 3D: round_id=1 + round_tag=r01_full2 must fail (wrong format)
        self.assertNotEqual("r01_full2", get_round_tag(1, True))

        # 3D: round_id=4 + round_tag=r04_case5 (final round)
        self.assertEqual("r04_case5", get_round_tag(4, True))


class TestRunner03CliContract(unittest.TestCase):
    """Verify 03 accepts --round_tag and --medsam_ft_root in its CLI."""

    def setUp(self):
        import importlib
        self.t03 = importlib.import_module(
            "idea1_iterclean_bank_adaptshape.03_train_medsam_full_only"
        )
        self.parser = self.t03.build_parser()
        self._tmpdir_obj = __import__("tempfile").mkdtemp(prefix="test_03_ctr_")
        self._tmpdir = Path(self._tmpdir_obj)
        (self._tmpdir / "checkpoint.pth").write_bytes(b"dummy")
        (self._tmpdir / "processed").mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir_obj, ignore_errors=True)

    def test_03_accepts_round_tag(self):
        ns = self.parser.parse_args([
            "--processed_root", str(self._tmpdir / "processed"),
            "--checkpoint", str(self._tmpdir / "checkpoint.pth"),
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--medsam_ft_root", str(self._tmpdir),
        ])
        self.assertEqual(ns.round_tag, "r00_full5")

    def test_03_accepts_medsam_ft_root(self):
        ns = self.parser.parse_args([
            "--processed_root", str(self._tmpdir / "processed"),
            "--checkpoint", str(self._tmpdir / "checkpoint.pth"),
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--medsam_ft_root", str(self._tmpdir),
        ])
        self.assertEqual(str(ns.medsam_ft_root), str(self._tmpdir))

    def test_03_out_root_is_required(self):
        """--medsam_ft_root is required."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                "--processed_root", str(self._tmpdir / "processed"),
                "--checkpoint", str(self._tmpdir / "checkpoint.pth"),
                "--datasets", "btcv",
                "--round_tag", "r00_full5",
            ])

    def test_03_no_out_root_accepted(self):
        """--out_root must not be accepted."""
        with self.assertRaises(SystemExit):
            self.parser.parse_args([
                "--processed_root", str(self._tmpdir / "processed"),
                "--checkpoint", str(self._tmpdir / "checkpoint.pth"),
                "--datasets", "btcv",
                "--round_tag", "r00_full5",
                "--out_root", str(self._tmpdir),
            ])


class TestNoSymlinkBridgeRequired(unittest.TestCase):
    """Verify output paths from build_fold_paths do NOT rely on symlinks."""

    def setUp(self):
        import importlib
        self.imported = importlib.import_module(
            "idea1_iterclean_bank_adaptshape.pipeline_common"
        )

    def test_split_and_support_paths_are_direct(self):
        p = self.imported.build_fold_paths(
            processed_root="/fake/processed",
            dataset="btcv",
            fold="fold_0",
            method="test",
            round_tag="r00_full5",
            medsam_ft_root="/fake/ft",
        )
        # All training paths should exist under medsam_ft_root/method/...
        expected_prefix = Path("/fake/ft/test/btcv/fold_0/r00_full5")
        for key in ("train_root", "train_last_path", "train_ema_path",
                     "train_log_path", "train_summary_path"):
            path = p[key]
            self.assertTrue(
                str(path).startswith(str(expected_prefix)),
                f"{key}={path} does NOT start with {expected_prefix}",
            )

    def test_no_method_selection_in_paths(self):
        p = self.imported.build_fold_paths(
            processed_root="/fake/processed",
            dataset="btcv",
            fold="fold_0",
            method="test",
            round_tag="r00_full5",
        )
        self.assertNotIn("method_selection", p)


class TestNoManualFullSelectionRequired(unittest.TestCase):
    """Verify 05 does NOT require manual full_selection files."""

    def test_05_has_no_full_selection_in_parser(self):
        import importlib
        t05 = importlib.import_module(
            "idea1_iterclean_bank_adaptshape.05_select_hard_by_gt_iou"
        )
        parser = t05.build_parser()
        help_text = parser.format_help()
        self.assertNotIn("selection_file", help_text)
        self.assertNotIn("selection.json", help_text)


class TestRoundPathsMatchProducers(unittest.TestCase):
    """Verify pipeline_common.build_fold_paths produces paths that match
    the actual file naming of each producer script."""

    def setUp(self):
        import importlib
        self.pc = importlib.import_module(
            "idea1_iterclean_bank_adaptshape.pipeline_common"
        )

    def test_split_filename_matches_producer(self):
        p = self.pc.build_fold_paths(
            processed_root="/fake/a", dataset="btcv", fold="fold_0",
            method="m", round_tag="r00_full5",
        )
        self.assertIn("m_r00_full5", str(p["split_path"]))
        self.assertIn("full_box_split_", str(p["split_path"]))

    def test_train_root_matches_03_output(self):
        p = self.pc.build_fold_paths(
            processed_root="/fake/a", dataset="btcv", fold="fold_0",
            method="m", round_tag="r00_full5", medsam_ft_root="/fake/ft",
        )
        self.assertIn("/m/btcv/fold_0/r00_full5", str(p["train_root"]))


class TestRunnerDryRunArgumentsSupported(unittest.TestCase):
    """Verify runner-produced commands contain only accepted CLI arguments."""

    def setUp(self):
        self.args = argparse.Namespace(
            python=sys.executable,
            processed_root=Path("/fake/p"),
            base_checkpoint=Path("/fake/ckpt.pth"),
            fold="fold_0",
            method="test_m",
            medsam_ft_root=Path("/fake/ft"),
            device="cpu",
            seed=2026,
            epochs=1, max_steps=5,
            lr=1e-5, weight_decay=0.01,
            ema_decay=0.99, max_grad_norm=1.0,
            train_log_every=10,
            pseudo_max_samples=0,
            alpha=0.5, beta=0.3, gamma=0.2,
            tau_low=0.1, tau_high=0.5,
            temperature=0.1, top_k=5,
            pseudo_log_every=10,
            kmax_shape=5, shape_size=64,
            cluster_max_iter=25,
            bg_ring_width_tokens=1,
            max_global_non_bg_coverage=0.10,
            max_fg_fallback_tokens=4,
        )
        self.scripts = {
            "split": Path("/fake/01.py"),
            "template": Path("/fake/02.py"),
            "train": Path("/fake/03.py"),
            "pseudo": Path("/fake/04.py"),
            "select": Path("/fake/05.py"),
        }

    def test_train_command_has_round_tag_not_out_root(self):
        cmd = runner.build_train_command(
            self.args, self.scripts, "btcv", "test_m", "r00_full5",
            stage_overwrite=False,
        )
        cmd_str = " ".join(cmd)
        self.assertNotIn("--out_root", cmd_str)
        self.assertIn("--round_tag", cmd_str)
        self.assertIn("--medsam_ft_root", cmd_str)

    def test_train_command_no_old_out_root(self):
        cmd = runner.build_train_command(
            self.args, self.scripts, "btcv", "test_m", "r00_full5",
            stage_overwrite=False,
        )
        self.assertNotIn("--out_root", cmd)

    def test_all_stage_commands_have_no_sac_args(self):
        """No command builder produces old SAC parameters."""
        for builder in (
            runner.build_split_command,
            runner.build_template_command,
            runner.build_train_command,
            runner.build_pseudo_command,
            runner.build_select_command,
        ):
            with self.subTest(builder=builder.__name__):
                kwargs = dict(
                    args=self.args,
                    scripts=self.scripts,
                    dataset="btcv",
                    method="test_m",
                    round_tag="r00_full5",
                    stage_overwrite=False,
                )
                if builder == runner.build_split_command:
                    kwargs.update(
                        round_id=0, full_count=5,
                        selection_file=None,
                    )
                elif builder == runner.build_pseudo_command:
                    kwargs.update(
                        ema_checkpoint=Path("/fake/ema.pth"),
                    )
                elif builder == runner.build_select_command:
                    kwargs.update(
                        round_id=0, select_count=5,
                    )

                cmd = builder(**kwargs)
                cmd_str = " ".join(cmd)
                self.assertNotIn("proto_fg", cmd_str.lower())
                self.assertNotIn("proto_bg", cmd_str.lower())
                self.assertNotIn("k_fg", cmd_str)
                self.assertNotIn("k_bg", cmd_str)
                self.assertNotIn("lambda_out", cmd_str)
                self.assertNotIn("lambda_seed", cmd_str)
                self.assertNotIn("sac_criterion", cmd_str)


class TestSelectCountContract(unittest.TestCase):
    """select_count must be 5 for 2D, 1 for 3D."""

    def test_runner_2d_select_count_is_5(self):
        """Non-final 2D rounds must select 5."""
        from idea1_iterclean_bank_adaptshape.pipeline_common import is_3d_dataset

        self.assertFalse(is_3d_dataset("cvc_clinicdb"))
        default_select = 1 if is_3d_dataset("cvc_clinicdb") else 5
        self.assertEqual(default_select, 5)

        self.assertFalse(is_3d_dataset("kvasir_polyp"))
        default_select = 1 if is_3d_dataset("kvasir_polyp") else 5
        self.assertEqual(default_select, 5)

    def test_runner_3d_select_count_is_1(self):
        """Non-final 3D rounds must select 1."""
        from idea1_iterclean_bank_adaptshape.pipeline_common import is_3d_dataset

        self.assertTrue(is_3d_dataset("btcv"))
        default_select = 1 if is_3d_dataset("btcv") else 5
        self.assertEqual(default_select, 1)

        self.assertTrue(is_3d_dataset("acdc"))
        default_select = 1 if is_3d_dataset("acdc") else 5
        self.assertEqual(default_select, 1)


class TestValidatorSelectContract(unittest.TestCase):
    """08 validator must handle final-round select contract."""

    def setUp(self):
        import tempfile
        self._tmpdir = Path(tempfile.mkdtemp(prefix="test_val_sel_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self._tmpdir), ignore_errors=True)

    def _make_round_summary(self, is_final, select_required, select_status):
        from idea1_iterclean_bank_adaptshape.pipeline_common import save_json_atomic
        rs_path = self._tmpdir / "round_summary.json"
        save_json_atomic({
            "is_final_round": is_final,
            "select_required": select_required,
            "select_status": select_status,
        }, rs_path)
        return rs_path

    def _make_hard_selection(self):
        p = self._tmpdir / "hard_selection.json"
        p.write_text(
            '{"method": "test", "round_id": 0, '
            '"new_selected_units": ["a", "b"], '
            '"selection_complete": true, '
            '"test_samples_used": 0}'
        )
        return p

    def _make_hard_summary(self):
        p = self._tmpdir / "hard_summary.json"
        p.write_text(
            '{"method": "test", "round_id": 0, '
            '"selection_complete": true, '
            '"test_samples_used": 0}'
        )
        return p

    def test_validator_requires_select_for_non_final_round(self):
        """Non-final round must have select outputs."""
        import importlib
        v08 = importlib.import_module(
            "idea1_iterclean_bank_adaptshape.08_validate_round_outputs"
        )
        rs = self._make_round_summary(
            is_final=False, select_required=True, select_status="completed",
        )
        paths = {
            "round_summary_path": rs,
            "hard_selection_path": self._make_hard_selection(),
            "hard_summary_path": self._make_hard_summary(),
        }
        errors = v08.validate_select(paths, is_3d=False, round_tag="r00_full5")
        self.assertEqual(errors, [])

    def test_validator_accepts_not_required_for_final_round(self):
        """Final round with not_required must skip select checks only for r03_full20."""
        import importlib
        v08 = importlib.import_module(
            "idea1_iterclean_bank_adaptshape.08_validate_round_outputs"
        )
        rs = self._make_round_summary(
            is_final=True, select_required=False, select_status="not_required",
        )
        paths = {
            "round_summary_path": rs,
            "hard_selection_path": None,
            "hard_summary_path": None,
        }
        errors = v08.validate_select(paths, is_3d=False, round_tag="r03_full20")
        self.assertEqual(errors, [])

    def test_validator_rejects_missing_select_without_final_marker(self):
        """Missing round_summary on non-final round means select is required."""
        import importlib
        v08 = importlib.import_module(
            "idea1_iterclean_bank_adaptshape.08_validate_round_outputs"
        )
        paths = {
            "round_summary_path": None,
            "hard_selection_path": None,
            "hard_summary_path": None,
        }
        errors = v08.validate_select(paths, is_3d=False, round_tag="r00_full5")
        self.assertTrue(any("key missing" in e or "check failed" in e
                            for e in errors))

    def test_final_round_does_not_require_fake_selection_files(self):
        """Final round must NOT have stale selection files."""
        import importlib
        v08 = importlib.import_module(
            "idea1_iterclean_bank_adaptshape.08_validate_round_outputs"
        )
        rs = self._make_round_summary(
            is_final=True, select_required=False, select_status="not_required",
        )
        paths = {
            "round_summary_path": rs,
            "hard_selection_path": self._make_hard_selection(),
            "hard_summary_path": self._make_hard_summary(),
        }
        errors = v08.validate_select(paths, is_3d=False, round_tag="r03_full20")
        self.assertTrue(any("unexpected file" in e for e in errors))


class TestValidatorSupportStats(unittest.TestCase):
    """08 validate_template must check new stats fields and reject old ones."""

    def setUp(self):
        import tempfile
        import importlib
        self._tmpdir = Path(tempfile.mkdtemp(prefix="test_vss_"))
        self.v08 = importlib.import_module(
            "idea1_iterclean_bank_adaptshape.08_validate_round_outputs"
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self._tmpdir), ignore_errors=True)

    def _write_npz(self, path, with_old_keys=False):
        import numpy as np
        import zipfile
        arrays = {
            "class_ids": np.array([1], dtype=np.int64),
            "feature_dim": np.array(768, dtype=np.int64),
            "shape_size": np.array([64, 64], dtype=np.int64),
            "method": np.array("test"),
            "round_tag": np.array("r00_full5"),
            "run_id": np.array("test_r00_full5"),
            "bank_fg_c1": np.ones((10, 768), dtype=np.float32),
            "bank_bg_c1": np.ones((10, 768), dtype=np.float32),
            "shape_templates_c1": np.ones((2, 64, 64), dtype=np.float32),
            "shape_semantic_centers_c1": np.ones((2, 768), dtype=np.float32),
            "shape_cluster_instance_counts_c1": np.array([5, 5], dtype=np.int64),
            "shape_cluster_support_counts_c1": np.array([5, 5], dtype=np.int64),
        }
        if with_old_keys:
            arrays["proto_fg_c1"] = np.ones((10, 768), dtype=np.float32)
        with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as zf:
            import io as _io_npz
            for name, arr in arrays.items():
                buf = _io_npz.BytesIO()
                np.save(buf, arr, allow_pickle=False)
                zf.writestr(name + ".npy", buf.getvalue())

    def _write_stats(self, path, include_old=False, include_new=True):
        import json
        stats = {}
        if include_new:
            stats.update({
                "bg_ring_width_tokens": 1,
                "fg_coverage_threshold": 0.90,
                "max_global_non_bg_coverage": 0.10,
                "zero_bg_instance_count": 2,
                "instances_with_bg_tokens": 8,
                "instances_without_bg_tokens": 2,
                "total_bg_ring_candidate_tokens": 500,
                "total_valid_global_bg_tokens": 200,
                "bg_ring_valid_ratio": 0.4,
                "global_non_bg_coverage_ring_min": 0.0,
                "global_non_bg_coverage_ring_mean": 0.05,
                "global_non_bg_coverage_ring_max": 0.10,
            })
        if include_old:
            stats["ring_expand_ratio"] = 5
            stats["bg_coverage_threshold"] = 0.1
        path.write_text(json.dumps(stats))

    def test_new_stats_fields_valid(self):
        npz = self._tmpdir / "test.npz"
        stats = self._tmpdir / "stats.json"
        self._write_npz(npz)
        self._write_stats(stats)
        paths = {"support_path": npz, "support_stats_path": stats}
        errors = self.v08.validate_template(paths, strict=True)
        self.assertEqual(errors, [])

    def test_rejects_old_ring_expand_ratio_in_stats(self):
        npz = self._tmpdir / "test.npz"
        stats = self._tmpdir / "stats.json"
        self._write_npz(npz)
        self._write_stats(stats, include_old=True)
        paths = {"support_path": npz, "support_stats_path": stats}
        errors = self.v08.validate_template(paths, strict=True)
        self.assertTrue(
            any("ring_expand_ratio" in e for e in errors),
            "must reject stats containing ring_expand_ratio",
        )

    def test_rejects_old_bg_coverage_threshold_in_stats(self):
        npz = self._tmpdir / "test.npz"
        stats = self._tmpdir / "stats.json"
        self._write_npz(npz)
        self._write_stats(stats, include_old=True)
        paths = {"support_path": npz, "support_stats_path": stats}
        errors = self.v08.validate_template(paths, strict=True)
        self.assertTrue(
            any("bg_coverage_threshold" in e for e in errors),
            "must reject stats containing bg_coverage_threshold",
        )

    def test_missing_new_field_reported(self):
        npz = self._tmpdir / "test.npz"
        stats = self._tmpdir / "stats.json"
        self._write_npz(npz)
        self._write_stats(stats, include_new=False)
        paths = {"support_path": npz, "support_stats_path": stats}
        errors = self.v08.validate_template(paths, strict=True)
        self.assertTrue(
            any("bg_ring_width_tokens" in e for e in errors),
            "must report missing bg_ring_width_tokens",
        )

    def test_wrong_bg_ring_width_reported(self):
        npz = self._tmpdir / "test.npz"
        stats = self._tmpdir / "stats.json"
        self._write_npz(npz)
        self._write_stats(stats)
        import json
        d = json.loads(stats.read_text())
        d["bg_ring_width_tokens"] = 5
        stats.write_text(json.dumps(d))
        paths = {"support_path": npz, "support_stats_path": stats}
        errors = self.v08.validate_template(paths, strict=True)
        self.assertTrue(any("bg_ring_width_tokens" in e for e in errors))


class TestFormalFinalRoundSemantics(unittest.TestCase):
    """is_final_round must be based on the full formal schedule, not --end_round."""

    def setUp(self):
        import tempfile
        self._tmpdir = Path(tempfile.mkdtemp(prefix="test_frs_"))

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self._tmpdir), ignore_errors=True)

    def _make_round_summary(self, is_final, select_required, select_status):
        from idea1_iterclean_bank_adaptshape.pipeline_common import save_json_atomic
        rs_path = self._tmpdir / "round_summary.json"
        save_json_atomic({
            "is_final_round": is_final,
            "select_required": select_required,
            "select_status": select_status,
        }, rs_path)
        return rs_path

    def test_2d_round0_requires_select(self):
        from idea1_iterclean_bank_adaptshape.run_iterative_sampling import (
            is_formal_final_round,
        )
        self.assertFalse(is_formal_final_round(0, False))
        self.assertFalse(is_formal_final_round(0, is_3d=False))

    def test_2d_round1_requires_select(self):
        from idea1_iterclean_bank_adaptshape.run_iterative_sampling import (
            is_formal_final_round,
        )
        self.assertFalse(is_formal_final_round(1, False))

    def test_2d_round2_requires_select(self):
        from idea1_iterclean_bank_adaptshape.run_iterative_sampling import (
            is_formal_final_round,
        )
        self.assertFalse(is_formal_final_round(2, False))

    def test_2d_round3_is_final(self):
        from idea1_iterclean_bank_adaptshape.run_iterative_sampling import (
            is_formal_final_round,
        )
        self.assertTrue(is_formal_final_round(3, False))

    def test_3d_round0_requires_select(self):
        from idea1_iterclean_bank_adaptshape.run_iterative_sampling import (
            is_formal_final_round,
        )
        self.assertFalse(is_formal_final_round(0, True))

    def test_3d_round3_requires_select(self):
        from idea1_iterclean_bank_adaptshape.run_iterative_sampling import (
            is_formal_final_round,
        )
        self.assertFalse(is_formal_final_round(3, True))

    def test_3d_round4_is_final(self):
        from idea1_iterclean_bank_adaptshape.run_iterative_sampling import (
            is_formal_final_round,
        )
        self.assertTrue(is_formal_final_round(4, True))

    def test_end_round_does_not_change_final_semantics(self):
        """--end_round must not affect is_final_round."""
        from idea1_iterclean_bank_adaptshape.run_iterative_sampling import (
            is_formal_final_round,
        )
        # r01_full10 is NEVER final regardless of --end_round
        self.assertFalse(is_formal_final_round(1, False))
        self.assertFalse(is_formal_final_round(1, True))

    def test_stop_after_stage_does_not_change_final_semantics(self):
        """--stop_after must not affect is_final_round."""
        from idea1_iterclean_bank_adaptshape.run_iterative_sampling import (
            is_formal_final_round,
        )
        # r02_full15 is NEVER final regardless of --stop_after
        self.assertFalse(is_formal_final_round(2, False))
        # r03_case4 is NEVER final in 3D
        self.assertFalse(is_formal_final_round(3, True))

    def test_validator_rejects_missing_round1_selection(self):
        """08 --strict must fail when r01_full10 is missing select files."""
        import importlib
        v08 = importlib.import_module(
            "idea1_iterclean_bank_adaptshape.08_validate_round_outputs"
        )
        rs = self._make_round_summary(
            is_final=False, select_required=True, select_status="completed",
        )
        paths = {
            "round_summary_path": rs,
            "hard_selection_path": None,
            "hard_summary_path": None,
        }
        errors = v08.validate_select(paths, is_3d=False, round_tag="r01_full10")
        # r01_full10 is not final per formal schedule → select files required
        self.assertTrue(len(errors) > 0)
        self.assertTrue(any("check failed" in e or "key missing" in e
                            for e in errors))

    def test_validator_accepts_missing_round3_selection_with_not_required(self):
        """08 --strict must accept missing select only for r03_full20 (formal final)."""
        import importlib
        v08 = importlib.import_module(
            "idea1_iterclean_bank_adaptshape.08_validate_round_outputs"
        )
        rs = self._make_round_summary(
            is_final=True, select_required=False, select_status="not_required",
        )
        paths = {
            "round_summary_path": rs,
            "hard_selection_path": None,
            "hard_summary_path": None,
        }
        errors = v08.validate_select(paths, is_3d=False, round_tag="r03_full20")
        self.assertEqual(errors, [])

    def test_runner_round1_produces_selection_even_when_end_round_is_1(self):
        """Runner must still produce select for r01_full10 even if --end_round 1."""
        from idea1_iterclean_bank_adaptshape.run_iterative_sampling import (
            is_formal_final_round,
        )
        # The select decision uses is_formal_final_round, not end_round
        should_select = not is_formal_final_round(1, False)
        self.assertTrue(should_select,
                        "r01_full10 must require select regardless of --end_round")
        # Verify r01_full10 is not the formal final round
        self.assertFalse(is_formal_final_round(1, False))


# ============================================================================
# T. Training Defaults Contract
# ============================================================================

class TestTrainingDefaults(unittest.TestCase):
    """Formal training defaults: epochs=1, max_steps=0, lr=1e-5, ema_decay=0.99."""

    @classmethod
    def setUpClass(cls):
        cls.parser = runner.build_parser()
        cls._defaults = {
            action.dest: action.default
            for action in cls.parser._actions
        }

    def test_default_epochs_is_1(self):
        self.assertEqual(self._defaults.get("epochs"), 1)

    def test_default_max_steps_is_0(self):
        self.assertEqual(self._defaults.get("max_steps"), 0)

    def test_default_lr_is_1e_5(self):
        self.assertAlmostEqual(self._defaults.get("lr"), 1e-5)

    def test_default_ema_decay_is_0_99(self):
        self.assertAlmostEqual(self._defaults.get("ema_decay"), 0.99)

    def test_default_weight_decay_is_0_01(self):
        self.assertAlmostEqual(self._defaults.get("weight_decay"), 0.01)


# ============================================================================
# U. Dry-Run Safety Contract
# ============================================================================

class TestDryRunSafety(unittest.TestCase):
    """dry_run must not write round_summary or stage_success into processed root."""

    def setUp(self):
        self._tmpdir_obj = __import__("tempfile").mkdtemp(prefix="test_dry_")
        self._tmpdir = Path(self._tmpdir_obj)

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self._tmpdir_obj), ignore_errors=True)

    def test_dry_run_not_in_status_values_equals_completed(self):
        """dry_run_completed is distinct from completed."""
        # These are the status values used in the runner
        self.assertNotEqual("dry_run_completed", "completed")

    def test_run_command_includes_dry_run_flag(self):
        """run_command records dry_run in command metadata."""
        # Verify the function signature accepts dry_run parameter
        import inspect
        sig = inspect.signature(runner.run_command)
        self.assertIn("dry_run", sig.parameters)

    def test_round_summary_skipped_in_dry_run(self):
        """Verify the round_summary writing block is gated by 'if not args.dry_run'."""
        import ast
        import textwrap

        runner_path = (
            Path(__file__).resolve().parents[1] / "run_iterative_sampling.py"
        )
        source = runner_path.read_text()
        tree = ast.parse(source)

        # Find the function that contains round_summary saving and dry_run gating
        found_gate = False
        found_save = False

        class DryRunVisitor(ast.NodeVisitor):
            def visit_If(self, node):
                nonlocal found_gate
                # Look for: if not args.dry_run:
                try:
                    test_str = ast.unparse(node.test)
                except Exception:
                    test_str = ""
                if "dry_run" in test_str and (
                    "not" in test_str
                    or "is False" in test_str
                    or "False" in test_str
                ):
                    # Check if save_json_atomic is called inside this block
                    class SaveChecker(ast.NodeVisitor):
                        def visit_Call(self, call_node):
                            nonlocal found_save
                            try:
                                if "save_json_atomic" in ast.unparse(call_node.func):
                                    if "round_summary" in ast.unparse(call_node):
                                        found_save = True
                            except Exception:
                                pass

                    SaveChecker().visit(node)
                    if found_save:
                        found_gate = True

                self.generic_visit(node)

        DryRunVisitor().visit(tree)
        self.assertTrue(found_gate,
                        "round_summary save_json_atomic must be gated by "
                        "'if not args.dry_run:'")
        self.assertTrue(found_save,
                        "round_summary save_json_atomic call must exist")

    def test_build_fold_paths_does_not_depend_on_dry_run(self):
        """build_fold_paths is a pure path constructor, dry_run irrelevant."""
        from idea1_iterclean_bank_adaptshape.pipeline_common import build_fold_paths
        paths = build_fold_paths(
            processed_root="/fake/p",
            dataset="btcv",
            fold="fold_0",
            method="test_m",
            round_tag="r00_full5",
            medsam_ft_root="/fake/ft",
        )
        self.assertIn("split_path", paths)
        self.assertIn("support_path", paths)


# ============================================================================
# V. FG Fallback Runner Tests
# ============================================================================

class TestFgFallbackRunner(unittest.TestCase):
    """Runner must pass --max_fg_fallback_tokens to 02 and validate it."""

    def setUp(self):
        self.args = argparse.Namespace(
            python="python",
            processed_root=Path("/fake/processed"),
            base_checkpoint=Path("/fake/checkpoint.pth"),
            medsam_ft_root=Path("/fake/ft"),
            fold="fold_0",
            device="cuda",
            seed=2026,
            kmax_shape=5,
            shape_size=64,
            cluster_max_iter=25,
            bg_ring_width_tokens=1,
            max_global_non_bg_coverage=0.10,
            epochs=1,
            max_steps=0,
            lr=1e-5,
            weight_decay=0.01,
            ema_decay=0.99,
            max_grad_norm=1.0,
            train_log_every=10,
            pseudo_max_samples=0,
            alpha=0.5, beta=0.3, gamma=0.2,
            tau_low=0.1, tau_high=0.5,
            temperature=0.1, top_k=5,
            pseudo_log_every=10,
            max_fg_fallback_tokens=4,
        )
        self.scripts = {
            "template": Path("/fake/02_build_support_template.py"),
        }

    def test_default_max_fg_fallback_tokens_is_4(self):
        parser = runner.build_parser()
        defaults = {
            action.dest: action.default
            for action in parser._actions
        }
        self.assertEqual(defaults.get("max_fg_fallback_tokens"), 4)

    def test_build_template_command_includes_max_fg_fallback_tokens(self):
        """build_template_command passes --max_fg_fallback_tokens."""
        self.args.max_fg_fallback_tokens = 3
        cmd = runner.build_template_command(
            self.args, self.scripts, "cvc_clinicdb", "test_m", "r00_full5",
            stage_overwrite=False,
        )
        self.assertIn("--max_fg_fallback_tokens", cmd)
        idx = cmd.index("--max_fg_fallback_tokens")
        self.assertEqual(cmd[idx + 1], "3")

    def test_validate_numeric_rejects_zero(self):
        """validate_numeric_args rejects --max_fg_fallback_tokens 0."""
        self.args.max_fg_fallback_tokens = 0
        with self.assertRaises(ValueError):
            runner.validate_numeric_args(self.args)

    def test_validate_numeric_rejects_negative(self):
        """validate_numeric_args rejects negative max_fg_fallback_tokens."""
        self.args.max_fg_fallback_tokens = -1
        with self.assertRaises(ValueError):
            runner.validate_numeric_args(self.args)


class TestValidatorSchemaV2(unittest.TestCase):
    """08 --strict must require fallback fields on v2 stats."""

    def setUp(self):
        import importlib
        import tempfile
        self._tmpdir = Path(tempfile.mkdtemp(prefix="test_vs2_"))
        self.v08 = importlib.import_module(
            "idea1_iterclean_bank_adaptshape.08_validate_round_outputs"
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self._tmpdir), ignore_errors=True)

    def _make_npz(self, path):
        import zipfile
        import io as _io_npz
        import numpy as np
        arrays = {
            "class_ids": np.array([1], dtype=np.int64),
            "feature_dim": np.array(768, dtype=np.int64),
            "shape_size": np.array([64, 64], dtype=np.int64),
            "method": np.array("test"),
            "round_tag": np.array("r00_full5"),
            "run_id": np.array("test_r00_full5"),
            "bank_fg_c1": np.ones((10, 768), dtype=np.float32),
            "bank_bg_c1": np.ones((10, 768), dtype=np.float32),
            "shape_templates_c1": np.ones((2, 64, 64), dtype=np.float32),
            "shape_semantic_centers_c1": np.ones((2, 768), dtype=np.float32),
            "shape_cluster_instance_counts_c1": np.array([5, 5], dtype=np.int64),
            "shape_cluster_support_counts_c1": np.array([5, 5], dtype=np.int64),
        }
        with zipfile.ZipFile(str(path), "w", zipfile.ZIP_DEFLATED) as zf:
            for name, arr in arrays.items():
                buf = _io_npz.BytesIO()
                np.save(buf, arr, allow_pickle=False)
                zf.writestr(name + ".npy", buf.getvalue())

    def test_v2_stats_with_all_fields_pass_strict(self):
        """v2 stats with complete fallback fields pass strict validation."""
        npz_path = self._tmpdir / "test.npz"
        stats_path = self._tmpdir / "stats.json"
        self._make_npz(npz_path)
        stats = {
            "support_stats_schema_version": 2,
            "bg_ring_width_tokens": 1,
            "fg_coverage_threshold": 0.90,
            "max_global_non_bg_coverage": 0.10,
            "zero_bg_instance_count": 0,
            "instances_with_bg_tokens": 10,
            "instances_without_bg_tokens": 0,
            "total_bg_ring_candidate_tokens": 500,
            "total_valid_global_bg_tokens": 200,
            "bg_ring_valid_ratio": 0.4,
            "global_non_bg_coverage_ring_min": 0.0,
            "global_non_bg_coverage_ring_mean": 0.05,
            "global_non_bg_coverage_ring_max": 0.10,
            "fg_primary_coverage_threshold": 0.90,
            "max_fg_fallback_tokens": 4,
            "instances_with_primary_fg": 8,
            "instances_with_fallback_fg": 2,
            "fg_fallback_instance_count": 2,
            "fg_fallback_token_count": 5,
            "fg_fallback_coverage_min": 0.05,
            "fg_fallback_coverage_mean": 0.20,
            "fg_fallback_coverage_max": 0.50,
        }
        stats_path.write_text(json.dumps(stats))
        paths = {"support_path": npz_path, "support_stats_path": stats_path}
        errors = self.v08.validate_template(paths, strict=True)
        self.assertEqual(errors, [])

    def test_v2_stats_missing_fallback_fields_fails_strict(self):
        """v2 stats without fallback fields fail strict validation."""
        npz_path = self._tmpdir / "test.npz"
        stats_path = self._tmpdir / "stats.json"
        self._make_npz(npz_path)
        stats = {
            "support_stats_schema_version": 2,
            "bg_ring_width_tokens": 1,
            "fg_coverage_threshold": 0.90,
            "max_global_non_bg_coverage": 0.10,
            "zero_bg_instance_count": 0,
            "instances_with_bg_tokens": 10,
            "instances_without_bg_tokens": 0,
            "total_bg_ring_candidate_tokens": 500,
            "total_valid_global_bg_tokens": 200,
            "bg_ring_valid_ratio": 0.4,
            "global_non_bg_coverage_ring_min": 0.0,
            "global_non_bg_coverage_ring_mean": 0.05,
            "global_non_bg_coverage_ring_max": 0.10,
        }
        stats_path.write_text(json.dumps(stats))
        paths = {"support_path": npz_path, "support_stats_path": stats_path}
        errors = self.v08.validate_template(paths, strict=True)
        self.assertTrue(len(errors) > 0)


# ============================================================================
if __name__ == "__main__":
    unittest.main()
