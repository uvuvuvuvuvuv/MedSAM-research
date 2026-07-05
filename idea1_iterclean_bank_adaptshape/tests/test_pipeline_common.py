"""Comprehensive unit tests for pipeline_common module.

Uses only standard-library unittest.  Run with::

    PYTHONPATH=/storage/baiyuting/data/MedSAM-main \\
        python -m idea1_iterclean_bank_adaptshape.tests.test_pipeline_common

"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

# -- path plumbing ----------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from idea1_iterclean_bank_adaptshape import pipeline_common as pc


# ============================================================================
# A.  get_round_tag — 2D
# ============================================================================

class TestGetRoundTag2D(unittest.TestCase):
    """Tests for pc.get_round_tag(round_id, is_3d=False)."""

    def test_r00_full5(self):
        self.assertEqual(pc.get_round_tag(0, False), "r00_full5")

    def test_r01_full10(self):
        self.assertEqual(pc.get_round_tag(1, False), "r01_full10")

    def test_r02_full15(self):
        self.assertEqual(pc.get_round_tag(2, False), "r02_full15")

    def test_r03_full20(self):
        self.assertEqual(pc.get_round_tag(3, False), "r03_full20")

    def test_all_2d_rounds(self):
        expected = {
            0: "r00_full5",
            1: "r01_full10",
            2: "r02_full15",
            3: "r03_full20",
        }
        for rid, exp in expected.items():
            with self.subTest(round_id=rid):
                self.assertEqual(pc.get_round_tag(rid, False), exp)

    def test_negative_round_id_raises_IndexError(self):
        with self.assertRaises(IndexError, msg="Negative round_id must raise IndexError"):
            pc.get_round_tag(-1, False)

    def test_round_id_too_large_raises_IndexError(self):
        with self.assertRaises(IndexError,
                               msg="round_id >= len(PLAN_2D) must raise IndexError"):
            pc.get_round_tag(4, False)

    def test_very_negative_round_id(self):
        with self.assertRaises(IndexError):
            pc.get_round_tag(-100, False)

    def test_very_large_round_id(self):
        with self.assertRaises(IndexError):
            pc.get_round_tag(999, False)


# ============================================================================
# B.  get_round_tag — 3D
# ============================================================================

class TestGetRoundTag3D(unittest.TestCase):
    """Tests for pc.get_round_tag(round_id, is_3d=True)."""

    def test_r00_case1(self):
        self.assertEqual(pc.get_round_tag(0, True), "r00_case1")

    def test_r01_case2(self):
        self.assertEqual(pc.get_round_tag(1, True), "r01_case2")

    def test_r02_case3(self):
        self.assertEqual(pc.get_round_tag(2, True), "r02_case3")

    def test_r03_case4(self):
        self.assertEqual(pc.get_round_tag(3, True), "r03_case4")

    def test_r04_case5(self):
        self.assertEqual(pc.get_round_tag(4, True), "r04_case5")

    def test_all_3d_rounds(self):
        expected = {
            0: "r00_case1",
            1: "r01_case2",
            2: "r02_case3",
            3: "r03_case4",
            4: "r04_case5",
        }
        for rid, exp in expected.items():
            with self.subTest(round_id=rid):
                self.assertEqual(pc.get_round_tag(rid, True), exp)

    def test_negative_round_id_raises_IndexError(self):
        with self.assertRaises(IndexError,
                               msg="Negative round_id must raise IndexError for 3D"):
            pc.get_round_tag(-1, True)

    def test_round_id_too_large_raises_IndexError(self):
        with self.assertRaises(IndexError,
                               msg="round_id >= len(PLAN_3D) must raise IndexError"):
            pc.get_round_tag(5, True)

    def test_very_negative_round_id(self):
        with self.assertRaises(IndexError):
            pc.get_round_tag(-100, True)

    def test_very_large_round_id(self):
        with self.assertRaises(IndexError):
            pc.get_round_tag(999, True)


# ============================================================================
# C.  is_3d_dataset
# ============================================================================

class TestIs3DDataset(unittest.TestCase):
    """Tests for pc.is_3d_dataset(dataset)."""

    # ---- known 3D datasets (lowercase) -------------------------------------
    def test_btcv_is_3d(self):
        self.assertTrue(pc.is_3d_dataset("btcv"),
                        msg="btcv should be recognized as 3D")

    def test_synapse_is_3d(self):
        self.assertTrue(pc.is_3d_dataset("synapse"),
                        msg="synapse should be recognized as 3D")

    def test_acdc_is_3d(self):
        self.assertTrue(pc.is_3d_dataset("acdc"),
                        msg="acdc should be recognized as 3D")

    def test_prostate158_is_3d(self):
        self.assertTrue(pc.is_3d_dataset("prostate158"),
                        msg="prostate158 should be recognized as 3D")

    # ---- 2D datasets return False ------------------------------------------
    def test_cvc_clinicdb_is_not_3d(self):
        self.assertFalse(pc.is_3d_dataset("cvc_clinicdb"),
                         msg="cvc_clinicdb is a 2D dataset")

    def test_empty_string_is_not_3d(self):
        self.assertFalse(pc.is_3d_dataset(""),
                         msg="empty string should not be 3D")

    def test_unknown_dataset_is_not_3d(self):
        self.assertFalse(pc.is_3d_dataset("some_unknown_dataset"),
                         msg="unknown dataset should not be 3D")

    # ---- case insensitive --------------------------------------------------
    def test_uppercase_btcv(self):
        self.assertTrue(pc.is_3d_dataset("BTCV"),
                        msg="BTCV (uppercase) should be recognized as 3D")

    def test_mixed_case_synapse(self):
        self.assertTrue(pc.is_3d_dataset("SynApSe"),
                        msg="Mixed-case should work")

    def test_mixed_case_acdc(self):
        self.assertTrue(pc.is_3d_dataset("ACDC"),
                        msg="ACDC (uppercase) should be recognized as 3D")

    def test_mixed_case_prostate158(self):
        self.assertTrue(pc.is_3d_dataset("Prostate158"),
                        msg="Mixed-case prostate158 should work")

    # ---- whitespace handling ------------------------------------------------
    def test_leading_trailing_spaces(self):
        self.assertTrue(pc.is_3d_dataset("  btcv  "),
                        msg="Leading/trailing spaces should be ignored")

    def test_tabs_and_newlines(self):
        self.assertTrue(pc.is_3d_dataset("\tbtcv\n"),
                        msg="Tabs and newlines should be stripped")

    def test_whitespace_only_is_not_3d(self):
        self.assertFalse(pc.is_3d_dataset("   "),
                         msg="Whitespace-only input should not be 3D")

    # ---- both case and whitespace combined ---------------------------------
    def test_whitespace_plus_case(self):
        self.assertTrue(pc.is_3d_dataset("  BTCV  "),
                        msg="Combined whitespace and case should work")


# ============================================================================
# D.  make_run_id
# ============================================================================

class TestMakeRunID(unittest.TestCase):
    """Tests for pc.make_run_id(method, round_tag)."""

    def test_basic_format(self):
        result = pc.make_run_id("idea1_iterclean_bank_adaptshape", "r00_full5")
        self.assertEqual(result, "idea1_iterclean_bank_adaptshape_r00_full5",
                         msg="Expected method_roundtag format")

    def test_contains_method_substring(self):
        method = "my_method"
        result = pc.make_run_id(method, "r01_full10")
        self.assertIn(method, result,
                      msg="run_id must contain the method name")

    def test_contains_round_tag_substring(self):
        round_tag = "r02_full15"
        result = pc.make_run_id("foo", round_tag)
        self.assertIn(round_tag, result,
                       msg="run_id must contain the round_tag")

    def test_different_methods_produce_different_ids(self):
        r1 = pc.make_run_id("method_A", "r00_full5")
        r2 = pc.make_run_id("method_B", "r00_full5")
        self.assertNotEqual(r1, r2,
                            msg="Different methods must produce different run_ids")

    def test_different_round_tags_produce_different_ids(self):
        r1 = pc.make_run_id("method_A", "r00_full5")
        r2 = pc.make_run_id("method_A", "r01_full10")
        self.assertNotEqual(r1, r2,
                            msg="Different round_tags must produce different run_ids")

    def test_result_is_string(self):
        result = pc.make_run_id("m", "r00_full5")
        self.assertIsInstance(result, str,
                              msg="make_run_id must return a string")


# ============================================================================
# E.  build_fold_paths
# ============================================================================

class TestBuildFoldPaths(unittest.TestCase):
    """Tests for pc.build_fold_paths(...)."""

    # ---- helper constants --------------------------------------------------
    _REQUIRED_KEYS: list[str] = [
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

    _ROUND_ARTIFACT_KEYS: list[str] = [
        "split_path", "split_summary_path",
        "support_path", "support_stats_path",
        "pseudo_teacher_dir", "pseudo_student_dir",
        "hard_selection_path", "hard_summary_path",
        "round_summary_path",
    ]

    def _make_paths(self, **overrides):
        kwargs = {
            "processed_root": "/fake/processed",
            "dataset": "btcv",
            "fold": "fold_0",
            "method": "test_method",
            "round_tag": "r00_full5",
        }
        kwargs.update(overrides)
        return pc.build_fold_paths(**kwargs)

    # ---- all required keys present -----------------------------------------
    def test_all_required_keys_present(self):
        p = self._make_paths()
        for k in self._REQUIRED_KEYS:
            with self.subTest(key=k):
                self.assertIn(k, p, msg=f"Key '{k}' missing from build_fold_paths result")

    # ---- run_id equals make_run_id ------------------------------------------
    def test_run_id_equals_make_run_id(self):
        p = self._make_paths(method="test_method", round_tag="r00_full5")
        expected = pc.make_run_id("test_method", "r00_full5")
        self.assertEqual(p["run_id"], expected,
                         msg="run_id must equal make_run_id(method, round_tag)")

    def test_run_id_updates_with_different_round(self):
        p = self._make_paths(method="test_method", round_tag="r01_full10")
        expected = pc.make_run_id("test_method", "r01_full10")
        self.assertEqual(p["run_id"], expected)

    # ---- round-artifact paths embed run_id ----------------------------------
    def test_round_artifact_paths_embed_run_id(self):
        p = self._make_paths()
        for k in self._ROUND_ARTIFACT_KEYS:
            with self.subTest(key=k):
                self.assertIn(p["run_id"], str(p[k]),
                              msg=f"{k} path must contain run_id, got: {p[k]}")

    # ---- support_path uses "support_bank_adaptshape_" prefix ----------------
    def test_support_path_prefix(self):
        p = self._make_paths()
        name = p["support_path"].name
        self.assertTrue(name.startswith("support_bank_adaptshape_"),
                        msg=f"support_path should start with 'support_bank_adaptshape_', got: {name}")
        self.assertTrue(name.endswith(".npz"),
                        msg=f"support_path should end with '.npz', got: {name}")

    def test_support_stats_path_prefix(self):
        p = self._make_paths()
        name = p["support_stats_path"].name
        self.assertTrue(name.startswith("support_bank_adaptshape_stats_"),
                        msg=f"support_stats_path should start with 'support_bank_adaptshape_stats_', got: {name}")
        self.assertTrue(name.endswith(".json"),
                        msg=f"support_stats_path should end with '.json', got: {name}")

    # ---- two different rounds produce different paths -----------------------
    def test_different_rounds_produce_different_paths(self):
        p0 = self._make_paths(round_tag="r00_full5", medsam_ft_root="/fake/ft")
        p1 = self._make_paths(round_tag="r01_full10", medsam_ft_root="/fake/ft")

        diff_keys = [
            "split_path", "support_path",
            "pseudo_teacher_dir", "pseudo_student_dir",
            "train_root", "train_last_path",
            "train_log_path", "round_summary_path",
        ]
        for k in diff_keys:
            with self.subTest(key=k):
                self.assertNotEqual(str(p0[k]), str(p1[k]),
                                    msg=f"{k} must differ between rounds, got same: {p0[k]}")

    # ---- training paths None without medsam_ft_root -------------------------
    def test_training_paths_none_without_ft_root(self):
        p = self._make_paths()  # no medsam_ft_root
        train_keys = ("train_root", "train_last_path", "train_ema_path",
                      "train_log_path", "train_summary_path")
        for k in train_keys:
            with self.subTest(key=k):
                self.assertIsNone(p[k],
                                  msg=f"{k} must be None without medsam_ft_root, got {p[k]!r}")

    # ---- training paths are Path objects with medsam_ft_root -----------------
    def test_training_paths_not_none_with_ft_root(self):
        p = self._make_paths(medsam_ft_root="/fake/ft")
        train_keys = ("train_root", "train_last_path", "train_ema_path",
                      "train_log_path", "train_summary_path")
        for k in train_keys:
            with self.subTest(key=k):
                self.assertIsNotNone(p[k],
                                     msg=f"{k} must not be None when medsam_ft_root is given")

    def test_training_paths_are_path_objects(self):
        p = self._make_paths(medsam_ft_root="/fake/ft")
        train_keys = ("train_root", "train_last_path", "train_ema_path",
                      "train_log_path", "train_summary_path")
        for k in train_keys:
            with self.subTest(key=k):
                self.assertIsInstance(p[k], Path,
                                      msg=f"{k} must be a Path object, got {type(p[k])}")

    # ---- train_root hierarchy -----------------------------------------------
    def test_train_root_hierarchy(self):
        p = self._make_paths(medsam_ft_root="/fake/ft")
        tr = str(p["train_root"])
        self.assertIn("test_method", tr,
                      msg="train_root must contain method name")
        self.assertIn("btcv", tr,
                      msg="train_root must contain dataset name")
        self.assertIn("fold_0", tr,
                      msg="train_root must contain fold name")
        self.assertIn("r00_full5", tr,
                      msg="train_root must contain round_tag")

    def test_train_root_ends_with_round_tag(self):
        p = self._make_paths(medsam_ft_root="/fake/ft")
        self.assertTrue(str(p["train_root"]).endswith("r00_full5"),
                        msg=f"train_root should end with round_tag, got: {p['train_root']}")

    # ---- train_log_path contains run_id -------------------------------------
    def test_train_log_path_contains_run_id(self):
        p = self._make_paths(medsam_ft_root="/fake/ft")
        self.assertIn(p["run_id"], str(p["train_log_path"]),
                      msg="train_log_path must contain run_id")

    # ---- fold_root identity ------------------------------------------------
    def test_fold_root_construction(self):
        p = self._make_paths(processed_root="/root", dataset="ds", fold="f0")
        self.assertEqual(str(p["fold_root"]), str(Path("/root/ds/f0")))


# ============================================================================
# F.  validate_writable_path
# ============================================================================

class TestValidateWritablePath(unittest.TestCase):
    """Tests for pc.validate_writable_path(path, allowed_root=None)."""

    @property
    def _frozen_root(self):
        return pc.FROZEN_BASELINE_ROOT

    @property
    def _idea1_root(self):
        return pc.IDEA1_WRITABLE_ROOT

    # ---- frozen baseline paths rejected ------------------------------------
    def test_frozen_subpath_rejected(self):
        """A path *under* the frozen baseline root must raise RuntimeError."""
        frozen_sub = str(self._frozen_root / "btcv" / "fold_0")
        with self.assertRaises(RuntimeError,
                               msg="Frozen baseline sub-path must be rejected"):
            pc.validate_writable_path(frozen_sub)

    def test_frozen_root_itself_rejected(self):
        """The frozen baseline root itself must raise RuntimeError."""
        with self.assertRaises(RuntimeError,
                               msg="Frozen baseline root itself must be rejected"):
            pc.validate_writable_path(str(self._frozen_root))

    # ---- IDEA1_WRITABLE_ROOT paths accepted ---------------------------------
    def test_idea1_subpath_accepted(self):
        """Sub-paths under IDEA1_WRITABLE_ROOT are accepted."""
        try:
            pc.validate_writable_path(str(self._idea1_root / "some" / "output"))
        except RuntimeError as e:
            self.fail(f"IDEA1_WRITABLE_ROOT sub-path should be accepted, got: {e}")

    def test_idea1_root_itself_accepted(self):
        """The exact IDEA1_WRITABLE_ROOT is accepted."""
        try:
            pc.validate_writable_path(str(self._idea1_root))
        except RuntimeError as e:
            self.fail(f"IDEA1_WRITABLE_ROOT itself should be accepted, got: {e}")

    # ---- allowed_root paths accepted ----------------------------------------
    def test_allowed_root_subpath_accepted(self):
        """Sub-paths under an explicit allowed_root are accepted."""
        try:
            pc.validate_writable_path(
                "/tmp/test_pipeline_allowed/subdir/file.json",
                allowed_root="/tmp/test_pipeline_allowed",
            )
        except RuntimeError as e:
            self.fail(f"allowed_root sub-path should be accepted, got: {e}")

    def test_allowed_root_itself_accepted(self):
        """The exact allowed_root itself is accepted."""
        try:
            pc.validate_writable_path(
                "/tmp/test_pipeline_allowed",
                allowed_root="/tmp/test_pipeline_allowed",
            )
        except RuntimeError as e:
            self.fail(f"allowed_root itself should be accepted, got: {e}")

    # ---- source repo directory rejected -------------------------------------
    def test_source_repo_dir_rejected(self):
        """The source repo directory (not under any writable root) is rejected."""
        source_dir = str(REPO_ROOT / "idea1_iterclean_bank_adaptshape")
        with self.assertRaises(RuntimeError,
                               msg="Source repo directory must be rejected"):
            pc.validate_writable_path(source_dir)

    # ---- random path not under any allowed root is rejected -----------------
    def test_random_path_rejected(self):
        """A path not under either IDEA1_WRITABLE_ROOT or allowed_root is rejected."""
        with self.assertRaises(RuntimeError,
                               msg="Path outside any allowed root must be rejected"):
            pc.validate_writable_path("/some/random/path")

    # ---- allowed_root overrides idea1 check for non-idea1 paths ------------
    def test_allowed_root_independent_of_idea1(self):
        """A path under allowed_root but NOT under IDEA1_WRITABLE_ROOT is accepted."""
        try:
            pc.validate_writable_path(
                "/tmp/independent_allowed/dir/file.txt",
                allowed_root="/tmp/independent_allowed",
            )
        except RuntimeError as e:
            self.fail(f"allowed_root should accept paths outside idea1 root, got: {e}")

    # ---- edge: frozen root check takes priority over allowed_root -----------
    def test_frozen_root_check_priority(self):
        """Even with allowed_root set, frozen baseline path is rejected."""
        frozen_sub = str(self._frozen_root / "btcv")
        with self.assertRaises(RuntimeError,
                               msg="Frozen path must be rejected regardless of allowed_root"):
            pc.validate_writable_path(frozen_sub, allowed_root=str(self._frozen_root))


# ============================================================================
# G.  stage_success markers
# ============================================================================

class TestStageSuccessMarkers(unittest.TestCase):
    """Tests for stage_success marker functions."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="pipeline_test_stage_")
        self._stage_dir = os.path.join(self._tmpdir, "stage_success", "run_test")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ---- has_stage_succeeded returns False before write ---------------------
    def test_has_stage_succeeded_false_before_write(self):
        self.assertFalse(
            pc.has_stage_succeeded(self._stage_dir, "train"),
            msg="has_stage_succeeded must return False when no marker exists",
        )

    # ---- has_stage_succeeded returns True after write -----------------------
    def test_has_stage_succeeded_true_after_write(self):
        pc.write_stage_success(self._stage_dir, "train")
        self.assertTrue(
            pc.has_stage_succeeded(self._stage_dir, "train"),
            msg="has_stage_succeeded must return True after write_stage_success",
        )

    # ---- read_stage_success returns correct data ----------------------------
    def test_read_stage_success_returns_correct_data(self):
        pc.write_stage_success(self._stage_dir, "split",
                               metadata={"count": 42, "ratio": 0.8})
        data = pc.read_stage_success(self._stage_dir, "split")
        self.assertEqual(data.get("status"), "success",
                         msg="status field should be 'success'")
        self.assertEqual(data.get("stage"), "split",
                         msg="stage field should match the written stage")
        self.assertEqual(data.get("metadata", {}).get("count"), 42)
        self.assertEqual(data.get("metadata", {}).get("ratio"), 0.8)

    def test_read_stage_success_timestamp_present(self):
        pc.write_stage_success(self._stage_dir, "train")
        data = pc.read_stage_success(self._stage_dir, "train")
        self.assertIn("timestamp", data,
                      msg="success marker must contain a timestamp")

    # ---- write without metadata works ---------------------------------------
    def test_write_stage_success_without_metadata(self):
        pc.write_stage_success(self._stage_dir, "finalize")
        self.assertTrue(
            pc.has_stage_succeeded(self._stage_dir, "finalize"),
            msg="write_stage_success should work without metadata",
        )

    # ---- invalid stage raises ValueError ------------------------------------
    def test_stage_success_path_invalid_stage_raises_ValueError(self):
        with self.assertRaises(ValueError,
                               msg="Invalid stage name must raise ValueError"):
            pc.stage_success_path(self._stage_dir, "nonexistent_stage")

    def test_write_stage_success_invalid_stage_raises_ValueError(self):
        with self.assertRaises(ValueError,
                               msg="write_stage_success with invalid stage must raise ValueError"):
            pc.write_stage_success(self._stage_dir, "bad_stage")

    def test_read_stage_success_invalid_stage_raises_ValueError(self):
        with self.assertRaises(ValueError,
                               msg="read_stage_success with invalid stage must raise ValueError"):
            pc.read_stage_success(self._stage_dir, "bad_stage")

    def test_has_stage_succeeded_invalid_stage_raises_ValueError(self):
        with self.assertRaises(ValueError,
                               msg="has_stage_succeeded with invalid stage must raise ValueError"):
            pc.has_stage_succeeded(self._stage_dir, "bad_stage")

    # ---- all valid stages produce unique paths ------------------------------
    def test_all_valid_stages_produce_unique_paths(self):
        import shutil
        paths = [str(pc.stage_success_path(self._stage_dir, s))
                 for s in pc._VALID_STAGES]
        self.assertEqual(
            len(paths), len(set(paths)),
            msg=f"All {len(pc._VALID_STAGES)} valid stages must produce unique paths, got: {paths}",
        )

    def test_valid_stage_path_names_end_with_json(self):
        for s in pc._VALID_STAGES:
            with self.subTest(stage=s):
                sp = pc.stage_success_path(self._stage_dir, s)
                self.assertEqual(sp.name, f"{s}.json",
                                 msg=f"Stage '{s}' path should be '{s}.json', got: {sp.name}")

    # ---- multiple stages independently --------------------------------------
    def test_multiple_stages_independent(self):
        """Writing one stage does not affect has_stage_succeeded for another."""
        pc.write_stage_success(self._stage_dir, "split")
        self.assertTrue(pc.has_stage_succeeded(self._stage_dir, "split"))
        self.assertFalse(pc.has_stage_succeeded(self._stage_dir, "train"),
                         msg="train stage should still be False after writing only split")


# ============================================================================
# H.  save_json_atomic
# ============================================================================

class TestSaveJsonAtomic(unittest.TestCase):
    """Tests for pc.save_json_atomic(data, target_path)."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="pipeline_test_json_")
        self._target = os.path.join(self._tmpdir, "data.json")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ---- roundtrip works ----------------------------------------------------
    def test_roundtrip(self):
        payload = {
            "hello": "world",
            "numbers": [1, 2, 3],
            "nested": {"a": 1, "b": [4, 5, 6]},
            "float": 3.14159,
        }
        pc.save_json_atomic(payload, self._target)
        self.assertTrue(
            os.path.isfile(self._target),
            msg="save_json_atomic must create the target file",
        )
        with open(self._target, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded, payload,
                         msg=f"Roundtrip failed: written data differs from original.\n"
                             f"Expected: {payload!r}\nGot:       {loaded!r}")

    # ---- unicode content preserved ------------------------------------------
    def test_unicode_roundtrip(self):
        payload = {
            "chinese": "中文测试",
            "japanese": "日本語テスト",
            "emoji_text": "(emoji)",
            "mixed": "Hello, 世界! こんにちは",
        }
        pc.save_json_atomic(payload, self._target)
        with open(self._target, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        for key, expected_val in payload.items():
            with self.subTest(key=key):
                self.assertEqual(loaded[key], expected_val,
                                 msg=f"Unicode value for '{key}' not preserved")

    # ---- overwrite works ----------------------------------------------------
    def test_overwrite(self):
        payload1 = {"first": True, "value": 100}
        pc.save_json_atomic(payload1, self._target)

        payload2 = {"second": True, "overwritten": "yes", "value": 200}
        pc.save_json_atomic(payload2, self._target)

        with open(self._target, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded, payload2,
                         msg=f"Overwrite should replace content entirely. Got: {loaded!r}")

    # ---- creates parent directories -----------------------------------------
    def test_creates_parent_dirs(self):
        deep_target = os.path.join(self._tmpdir, "sub", "deep", "nested", "file.json")
        pc.save_json_atomic({"ok": True}, deep_target)
        self.assertTrue(
            os.path.isfile(deep_target),
            msg="save_json_atomic must create parent directories",
        )

    # ---- empty data structures ----------------------------------------------
    def test_empty_dict(self):
        pc.save_json_atomic({}, self._target)
        with open(self._target, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded, {},
                         msg="Empty dict roundtrip should work")

    def test_empty_list(self):
        pc.save_json_atomic([], self._target)
        with open(self._target, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        self.assertEqual(loaded, [],
                         msg="Empty list roundtrip should work")

    # ---- file is valid JSON with proper formatting --------------------------
    def test_output_is_valid_json(self):
        pc.save_json_atomic({"a": 1}, self._target)
        with open(self._target, "r", encoding="utf-8") as fh:
            raw = fh.read()
        # Check it parses
        parsed = json.loads(raw)
        self.assertEqual(parsed, {"a": 1})
        # Check indentation is used (should have newlines)
        self.assertIn("\n", raw, msg="Output should be pretty-printed with newlines")

    # ---- atomic: target does not exist before completion --------------------
    # (We test the post-condition: the file either doesn't exist or is complete)
    def test_target_is_complete_file(self):
        payload = {"key": "value"}
        pc.save_json_atomic(payload, self._target)
        # After save_json_atomic returns, the file must exist and be readable
        self.assertTrue(os.path.isfile(self._target))
        size = os.path.getsize(self._target)
        self.assertGreater(size, 0, msg="Written file must be non-empty")


# ============================================================================
# I.  check_git_dirty
# ============================================================================

class TestCheckGitDirty(unittest.TestCase):
    """Tests for pc.check_git_dirty(repo_path=None)."""

    def test_returns_dict(self):
        info = pc.check_git_dirty()
        self.assertIsInstance(info, dict,
                              msg="check_git_dirty must return a dict")

    def test_has_dirty_key(self):
        info = pc.check_git_dirty()
        self.assertIn("dirty", info,
                      msg="Returned dict must contain 'dirty' key")

    def test_has_status_porcelain_key(self):
        info = pc.check_git_dirty()
        self.assertIn("status_porcelain", info,
                      msg="Returned dict must contain 'status_porcelain' key")

    def test_dirty_is_bool(self):
        """'dirty' must be either True or False (boolean)."""
        info = pc.check_git_dirty()
        self.assertIsInstance(info["dirty"], bool,
                              msg=f"'dirty' must be bool, got {type(info['dirty'])}")
        # semantic check: must be exactly True or False (not just truthy/falsy)
        self.assertIn(info["dirty"], (True, False),
                      msg="'dirty' must be True or False")

    def test_status_porcelain_is_string(self):
        """'status_porcelain' must be a string (may be empty)."""
        info = pc.check_git_dirty()
        self.assertIsInstance(info["status_porcelain"], str,
                              msg=f"'status_porcelain' must be str, got {type(info['status_porcelain'])}")

    def test_only_expected_keys(self):
        """The returned dict should not contain unexpected keys."""
        info = pc.check_git_dirty()
        allowed_keys = {"dirty", "status_porcelain"}
        extra = set(info.keys()) - allowed_keys
        self.assertEqual(extra, set(),
                         msg=f"Unexpected keys in check_git_dirty result: {extra}")

    def test_with_explicit_repo_path(self):
        """Calling with the current repo dir should return the same keys."""
        info = pc.check_git_dirty(repo_path=str(REPO_ROOT))
        self.assertIn("dirty", info)
        self.assertIn("status_porcelain", info)
        self.assertIsInstance(info["dirty"], bool)
        self.assertIsInstance(info["status_porcelain"], str)

    def test_with_nonexistent_path(self):
        """Calling with a nonexistent path should not crash."""
        try:
            info = pc.check_git_dirty(repo_path="/nonexistent/path/12345")
            self.assertIsInstance(info, dict)
            self.assertIn("dirty", info)
            self.assertIn("status_porcelain", info)
        except Exception as e:
            self.fail(f"check_git_dirty should not raise with nonexistent path, got: {e}")


# ============================================================================
# J.  Edge cases / cross-cutting
# ============================================================================

class TestCrossCutting(unittest.TestCase):
    """Cross-cutting edge cases and invariants."""

    def test_get_round_tag_2d_3d_different_for_same_id(self):
        """Same round_id gives different tags for 2D vs 3D."""
        tag_2d = pc.get_round_tag(0, False)
        tag_3d = pc.get_round_tag(0, True)
        self.assertNotEqual(tag_2d, tag_3d,
                            msg="2D and 3D round tags must differ for the same round_id")

    def test_build_fold_paths_run_id_is_string(self):
        p = pc.build_fold_paths(
            processed_root="/fake/processed",
            dataset="btcv",
            fold="fold_0",
            method="test_method",
            round_tag="r00_full5",
        )
        self.assertIsInstance(p["run_id"], str)

    def test_build_fold_paths_stage_success_dir_contains_run_id(self):
        p = pc.build_fold_paths(
            processed_root="/fake/processed",
            dataset="btcv",
            fold="fold_0",
            method="test_method",
            round_tag="r00_full5",
        )
        self.assertIn(p["run_id"], str(p["stage_success_dir"]),
                      msg="stage_success_dir must contain run_id")

    def test_build_fold_paths_3d_dataset_round_tags(self):
        """Verify build_fold_paths works with 3D round tags."""
        for round_id in range(5):
            tag = pc.get_round_tag(round_id, is_3d=True)
            with self.subTest(round_tag=tag):
                p = pc.build_fold_paths(
                    processed_root="/fake/processed",
                    dataset="btcv",
                    fold="fold_0",
                    method="test_method",
                    round_tag=tag,
                    medsam_ft_root="/fake/ft",
                )
                self.assertEqual(p["round_tag"], tag)
                self.assertIn(tag, str(p["train_root"]))

    def test_build_fold_paths_2d_dataset_round_tags(self):
        """Verify build_fold_paths works with 2D round tags."""
        for round_id in range(4):
            tag = pc.get_round_tag(round_id, is_3d=False)
            with self.subTest(round_tag=tag):
                p = pc.build_fold_paths(
                    processed_root="/fake/processed",
                    dataset="cvc_clinicdb",
                    fold="fold_0",
                    method="test_method",
                    round_tag=tag,
                )
                self.assertEqual(p["round_tag"], tag)
                self.assertIn(tag, p["run_id"])


# ============================================================================
if __name__ == "__main__":
    unittest.main()
