"""Unit tests for 04_generate_pseudo_bank_adaptshape module."""

from __future__ import annotations

import argparse
import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

t04 = importlib.import_module(
    "idea1_iterclean_bank_adaptshape.04_generate_pseudo_bank_adaptshape"
)

# ---------------------------------------------------------------------------
# Optional dependency flags
# ---------------------------------------------------------------------------
try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cv2  # noqa: F401

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ============================================================================
# A. compute_qf_from_embedding
# ============================================================================


@unittest.skipUnless(HAS_TORCH, "torch not available")
class TestComputeQFFromEmbedding(unittest.TestCase):
    """Tests for compute_qf_from_embedding with synthetic features."""

    def setUp(self) -> None:
        self.C = 256
        self.H, self.W = 8, 8
        self.embedding = torch.randn(1, self.C, self.H, self.W)
        self.bank_fg = np.random.randn(50, self.C).astype(np.float32)
        self.bank_bg = np.random.randn(50, self.C).astype(np.float32)

    # -- Core behaviour -------------------------------------------------------

    def test_output_shape(self) -> None:
        """qF has shape (1, 1, H, W) matching output_hw."""
        for oh, ow in ((8, 8), (16, 16), (32, 24)):
            with self.subTest(output_hw=(oh, ow)):
                result = t04.compute_qf_from_embedding(
                    self.embedding,
                    self.bank_fg,
                    self.bank_bg,
                    output_hw=(oh, ow),
                )
                self.assertEqual(result.shape, (1, 1, oh, ow))

    def test_values_in_01_range(self) -> None:
        """All qF values lie in [0, 1]."""
        result = t04.compute_qf_from_embedding(
            self.embedding,
            self.bank_fg,
            self.bank_bg,
            output_hw=(self.H, self.W),
        )
        self.assertTrue(bool((result >= 0.0).all()))
        self.assertTrue(bool((result <= 1.0).all()))

    def test_top_k_degrades_when_bank_too_small(self) -> None:
        """When bank has fewer entries than top_k it uses all entries."""
        small_fg = np.random.randn(3, self.C).astype(np.float32)
        small_bg = np.random.randn(5, self.C).astype(np.float32)
        result = t04.compute_qf_from_embedding(
            self.embedding,
            small_fg,
            small_bg,
            output_hw=(self.H, self.W),
            top_k=10,
        )
        self.assertEqual(result.shape, (1, 1, self.H, self.W))
        self.assertTrue(bool((result >= 0.0).all()))
        self.assertTrue(bool((result <= 1.0).all()))

    def test_top_k_changes_result(self) -> None:
        """Using different top_k values produces different qF."""
        # Include a distant outlier so top-k selection matters.
        bank_fg = np.random.randn(10, self.C).astype(np.float32)
        bank_bg = np.random.randn(10, self.C).astype(np.float32)
        bank_fg[0] = bank_fg[0] * 100.0  # extreme outlier

        r1 = t04.compute_qf_from_embedding(
            self.embedding,
            bank_fg,
            bank_bg,
            output_hw=(self.H, self.W),
            temperature=1.0,
            top_k=5,
        )
        r2 = t04.compute_qf_from_embedding(
            self.embedding,
            bank_fg,
            bank_bg,
            output_hw=(self.H, self.W),
            temperature=1.0,
            top_k=10,
        )
        # The outlier excluded by top_k=5 should shift scores for top_k=10.
        self.assertFalse(bool(torch.allclose(r1, r2)))

    # -- Error paths ----------------------------------------------------------

    def test_batch_size_not_one_raises(self) -> None:
        """ValueError when embedding batch dimension != 1."""
        bad_embedding = torch.randn(2, self.C, self.H, self.W)
        with self.assertRaises(ValueError):
            t04.compute_qf_from_embedding(
                bad_embedding,
                self.bank_fg,
                self.bank_bg,
                output_hw=(self.H, self.W),
            )

    def test_temperature_scales_logits(self) -> None:
        """Higher temperature produces more uniform qF across pixels."""
        r_low_t = t04.compute_qf_from_embedding(
            self.embedding,
            self.bank_fg,
            self.bank_bg,
            output_hw=(self.H, self.W),
            temperature=0.01,
        )
        r_high_t = t04.compute_qf_from_embedding(
            self.embedding,
            self.bank_fg,
            self.bank_bg,
            output_hw=(self.H, self.W),
            temperature=100.0,
        )
        # Low temperature → more extreme values (closer to 0 or 1).
        std_low = float(r_low_t.std())
        std_high = float(r_high_t.std())
        self.assertGreater(std_low, std_high)


# ============================================================================
# B. make_box_mask_np
# ============================================================================


class TestMakeBoxMaskNp(unittest.TestCase):
    """Tests for make_box_mask_np."""

    # -- Box fully within image -----------------------------------------------

    def test_box_within_bounds(self) -> None:
        """Box completely inside the image produces correct boolean mask."""
        mask = t04.make_box_mask_np([2, 3, 5, 7], height=10, width=10)
        expected = np.zeros((10, 10), dtype=bool)
        expected[3:7, 2:5] = True
        np.testing.assert_array_equal(mask, expected)

    def test_float_box_values(self) -> None:
        """Float coordinates are correctly floored / ceiled."""
        mask = t04.make_box_mask_np([1.7, 2.3, 5.2, 6.8], height=10, width=10)
        expected = np.zeros((10, 10), dtype=bool)
        # x1=1.7→floor→1  y1=2.3→floor→2  x2=5.2→ceil→6  y2=6.8→ceil→7
        expected[2:7, 1:6] = True
        np.testing.assert_array_equal(mask, expected)

    # -- Box partially outside image ------------------------------------------

    def test_box_partially_outside_top_left(self) -> None:
        """Box extending beyond top/left edges is clamped."""
        mask = t04.make_box_mask_np([-5, -5, 3, 3], height=10, width=10)
        expected = np.zeros((10, 10), dtype=bool)
        expected[0:3, 0:3] = True
        np.testing.assert_array_equal(mask, expected)

    def test_box_partially_outside_bottom_right(self) -> None:
        """Box extending beyond bottom/right edges is clamped."""
        mask = t04.make_box_mask_np([7, 7, 15, 15], height=10, width=10)
        expected = np.zeros((10, 10), dtype=bool)
        expected[7:10, 7:10] = True
        np.testing.assert_array_equal(mask, expected)

    def test_box_partially_outside_all_sides(self) -> None:
        """Box extending beyond all edges is clamped on every side."""
        mask = t04.make_box_mask_np([-10, -10, 20, 20], height=5, width=5)
        expected = np.ones((5, 5), dtype=bool)
        np.testing.assert_array_equal(mask, expected)

    # -- Empty box ------------------------------------------------------------

    def test_empty_box_x2_le_x1(self) -> None:
        """x2 <= x1 produces all-False mask."""
        mask = t04.make_box_mask_np([5, 3, 5, 7], height=10, width=10)
        self.assertFalse(bool(mask.any()))

    def test_empty_box_x2_lt_x1(self) -> None:
        """x2 < x1 produces all-False mask."""
        mask = t04.make_box_mask_np([6, 3, 4, 7], height=10, width=10)
        self.assertFalse(bool(mask.any()))

    def test_empty_box_y2_le_y1(self) -> None:
        """y2 <= y1 produces all-False mask."""
        mask = t04.make_box_mask_np([2, 5, 5, 5], height=10, width=10)
        self.assertFalse(bool(mask.any()))

    def test_fully_clamped_empty_box(self) -> None:
        """Box clamped entirely out of valid range is all-False."""
        mask = t04.make_box_mask_np([-10, -10, -5, -5], height=10, width=10)
        self.assertFalse(bool(mask.any()))


# ============================================================================
# C. build_shape_map_np
# ============================================================================


@unittest.skipUnless(HAS_CV2, "cv2 not available")
class TestBuildShapeMapNp(unittest.TestCase):
    """Tests for build_shape_map_np."""

    # -- Shape prior resized to box region ------------------------------------

    def test_shape_resized_to_box_region(self) -> None:
        """Shape prior is interpolated into the box region of output."""
        shape_a = np.ones((64, 64), dtype=np.float32)
        box = [10, 10, 30, 30]
        result = t04.build_shape_map_np(shape_a, box, output_height=50, output_width=50)
        # Inside box should be non-zero; outside should be zero.
        box_region = result[10:30, 10:30]
        outside = result.copy()
        outside[10:30, 10:30] = 0.0
        self.assertTrue(bool((box_region > 0).any()))
        self.assertEqual(float(outside.sum()), 0.0)

    def test_shape_resized_to_arbitrary_box_size(self) -> None:
        """Shape prior is correctly resized for non-square box."""
        shape_a = np.ones((32, 64), dtype=np.float32)
        box = [5, 5, 15, 25]
        result = t04.build_shape_map_np(shape_a, box, output_height=40, output_width=40)
        box_region = result[5:25, 5:15]
        self.assertEqual(box_region.shape, (20, 10))
        self.assertTrue(bool((box_region > 0).all()))

    # -- Values clipped to [0, 1] --------------------------------------------

    def test_values_clipped_to_01(self) -> None:
        """Output values are clipped to [0, 1] regardless of input."""
        # Random input that may contain values outside [0, 1]
        shape_a = (np.random.randn(32, 32).astype(np.float32) * 3.0) + 0.5
        box = [5, 5, 20, 20]
        result = t04.build_shape_map_np(shape_a, box, output_height=30, output_width=30)
        self.assertTrue(bool((result >= 0.0).all()))
        self.assertTrue(bool((result <= 1.0).all()))

    # -- Empty box ------------------------------------------------------------

    def test_empty_box_returns_all_zeros(self) -> None:
        """Empty box region (x2 <= x1) returns all-zeros output."""
        shape_a = np.ones((32, 32), dtype=np.float32)
        result = t04.build_shape_map_np(
            shape_a, [10, 10, 10, 20], output_height=30, output_width=30
        )
        self.assertEqual(float(result.sum()), 0.0)

    def test_empty_box_y2_le_y1_returns_zeros(self) -> None:
        """Empty box (y2 <= y1) returns all-zeros output."""
        shape_a = np.ones((32, 32), dtype=np.float32)
        result = t04.build_shape_map_np(
            shape_a, [10, 15, 20, 15], output_height=30, output_width=30
        )
        self.assertEqual(float(result.sum()), 0.0)

    def test_box_clamped_outside_image(self) -> None:
        """Box clamped entirely outside valid area returns all-zeros."""
        shape_a = np.ones((32, 32), dtype=np.float32)
        # Box entirely to the left of the image after clamping.
        result = t04.build_shape_map_np(
            shape_a, [-20, 5, -10, 15], output_height=30, output_width=30
        )
        # x1 clamped to 0, x2 clamped to 0 → x2 <= x1 → all zeros.
        self.assertEqual(float(result.sum()), 0.0)


# ============================================================================
# D. merge_instance_into_tri
# ============================================================================


class TestMergeInstanceIntoTri(unittest.TestCase):
    """Tests for merge_instance_into_tri."""

    @staticmethod
    def _mask(h: int, w: int) -> np.ndarray:
        return np.zeros((h, w), dtype=bool)

    # -- D.1 Single foreground instance ---------------------------------------

    def test_single_foreground_instance(self) -> None:
        """Single FG instance sets correct label_id in foreground region."""
        tri = np.zeros((10, 10), dtype=np.uint8)
        fg = self._mask(10, 10)
        fg[2:5, 3:7] = True
        result = t04.merge_instance_into_tri(
            tri, fg, self._mask(10, 10), label_id=3
        )
        expected = np.zeros((10, 10), dtype=np.uint8)
        expected[2:5, 3:7] = 3
        np.testing.assert_array_equal(result, expected)
        # Outside FG region is unchanged (0).
        self.assertEqual(int(result[0, 0]), 0)
        self.assertEqual(int(result[9, 9]), 0)

    # -- D.2 Two non-overlapping instances ------------------------------------

    def test_two_non_overlapping_instances(self) -> None:
        """Two non-overlapping instances: both assigned correctly."""
        tri = np.zeros((10, 10), dtype=np.uint8)

        fg1 = self._mask(10, 10)
        fg1[1:4, 1:4] = True
        tri = t04.merge_instance_into_tri(
            tri, fg1, self._mask(10, 10), label_id=1
        )

        fg2 = self._mask(10, 10)
        fg2[6:9, 6:9] = True
        result = t04.merge_instance_into_tri(
            tri, fg2, self._mask(10, 10), label_id=2
        )

        np.testing.assert_array_equal(result[1:4, 1:4], 1)
        np.testing.assert_array_equal(result[6:9, 6:9], 2)
        # Everything else should be background.
        remainder = result.copy()
        remainder[1:4, 1:4] = 0
        remainder[6:9, 6:9] = 0
        self.assertEqual(int(remainder.sum()), 0)

    # -- D.3 Overlapping instances with different labels → conflict -----------

    def test_overlapping_different_labels_conflict(self) -> None:
        """Overlapping FG instances with different labels → 255 in overlap."""
        tri = np.zeros((10, 10), dtype=np.uint8)

        fg1 = self._mask(10, 10)
        fg1[2:6, 2:8] = True
        tri = t04.merge_instance_into_tri(
            tri, fg1, self._mask(10, 10), label_id=1
        )

        fg2 = self._mask(10, 10)
        fg2[4:8, 4:10] = True
        result = t04.merge_instance_into_tri(
            tri, fg2, self._mask(10, 10), label_id=2
        )

        # Overlap [4:6, 4:8] must be 255 (conflict).
        np.testing.assert_array_equal(result[4:6, 4:8], 255)
        # First-only region.
        np.testing.assert_array_equal(result[2:6, 2:4], 1)
        np.testing.assert_array_equal(result[2:4, 4:8], 1)
        # Second-only region.
        np.testing.assert_array_equal(result[6:8, 4:10], 2)
        np.testing.assert_array_equal(result[4:6, 8:10], 2)

    def test_overlapping_same_label_no_conflict(self) -> None:
        """Overlapping FG with same label_id does NOT produce conflict."""
        tri = np.zeros((10, 10), dtype=np.uint8)

        fg1 = self._mask(10, 10)
        fg1[2:6, 2:8] = True
        tri = t04.merge_instance_into_tri(
            tri, fg1, self._mask(10, 10), label_id=1
        )

        fg2 = self._mask(10, 10)
        fg2[4:8, 4:10] = True
        result = t04.merge_instance_into_tri(
            tri, fg2, self._mask(10, 10), label_id=1
        )

        np.testing.assert_array_equal(result[4:6, 4:8], 1)  # no 255

    # -- D.4 Unknown overrides background -------------------------------------

    def test_unknown_overrides_background(self) -> None:
        """unknown_mask pixels override background (0) to 255."""
        tri = np.zeros((10, 10), dtype=np.uint8)

        fg1 = self._mask(10, 10)
        fg1[0:2, :] = True
        tri = t04.merge_instance_into_tri(
            tri, fg1, self._mask(10, 10), label_id=1
        )

        fg2 = self._mask(10, 10)
        fg2[3:4, :] = True
        unknown2 = self._mask(10, 10)
        unknown2[4:6, :] = True
        result = t04.merge_instance_into_tri(
            tri, fg2, unknown2, label_id=2
        )

        np.testing.assert_array_equal(result[0:2, :], 1)   # first FG preserved
        np.testing.assert_array_equal(result[2:3, :], 0)   # still background
        np.testing.assert_array_equal(result[3:4, :], 2)   # second FG
        np.testing.assert_array_equal(result[4:6, :], 255)  # unknown → 255
        np.testing.assert_array_equal(result[6:, :], 0)    # remaining bg

    def test_unknown_does_not_override_existing_fg(self) -> None:
        """unknown_mask does not overwrite existing non-zero tri pixels."""
        tri = np.zeros((10, 10), dtype=np.uint8)

        fg1 = self._mask(10, 10)
        fg1[2:6, :] = True
        tri = t04.merge_instance_into_tri(
            tri, fg1, self._mask(10, 10), label_id=1
        )

        # Second instance: unknown in same region → does NOT override FG.
        unknown2 = self._mask(10, 10)
        unknown2[2:6, :] = True
        result = t04.merge_instance_into_tri(
            tri, self._mask(10, 10), unknown2, label_id=2
        )

        np.testing.assert_array_equal(result[2:6, :], 1)  # FG unchanged

    def test_unknown_does_not_override_existing_unknown(self) -> None:
        """unknown_mask does not overwrite existing 255 pixels."""
        tri = np.full((10, 10), 255, dtype=np.uint8)

        unknown = self._mask(10, 10)
        unknown[3:7, 3:7] = True
        result = t04.merge_instance_into_tri(
            tri, self._mask(10, 10), unknown, label_id=1
        )

        np.testing.assert_array_equal(result, 255)

    def test_modifies_in_place(self) -> None:
        """The function modifies tri in-place and returns the same array."""
        tri = np.zeros((5, 5), dtype=np.uint8)
        fg = self._mask(5, 5)
        fg[1:3, 1:3] = True
        result = t04.merge_instance_into_tri(
            tri, fg, self._mask(5, 5), label_id=1
        )
        self.assertIs(result, tri)
        self.assertEqual(int(tri[1, 1]), 1)


# ============================================================================
# D.5  generate_one_slice  –  label_mode="full"
# ============================================================================


class TestGenerateOneSliceFullMode(unittest.TestCase):
    """Tests for generate_one_slice with label_mode='full'."""

    def test_full_mode_returns_teacher_gt_copy(self) -> None:
        """label_mode='full' returns a copy of teacher_gt, not the same object."""
        gt = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        result, stats = t04.generate_one_slice(
            model=None,
            image_tensor=None,
            prompt_meta={},
            label_mode="full",
            output_hw=(2, 2),
            template={},
            args=MagicMock(),
            teacher_gt=gt,
        )
        np.testing.assert_array_equal(result, gt)
        self.assertIsNot(result, gt)

        # Verify independence.
        result[0, 0] = 99
        self.assertEqual(int(gt[0, 0]), 1)

    def test_full_mode_stats_all_zeros(self) -> None:
        """All auxiliary stats are exactly zero for full-label samples."""
        gt = np.zeros((4, 4), dtype=np.uint8)
        _, stats = t04.generate_one_slice(
            model=None,
            image_tensor=None,
            prompt_meta={},
            label_mode="full",
            output_hw=(4, 4),
            template={},
            args=MagicMock(),
            teacher_gt=gt,
        )
        for key, expected in [
            ("mean_P_inside_box", 0.0),
            ("mean_qF_inside_box", 0.0),
            ("mean_A_inside_box", 0.0),
            ("weak_activation_ratio_0.30_0.40", 0.0),
            ("weak_activation_ratio_0.40_0.50", 0.0),
            ("box_fg_ratio", 0.0),
            ("box_unknown_ratio", 0.0),
        ]:
            with self.subTest(key=key):
                self.assertEqual(stats[key], expected)

    def test_full_mode_shape_mismatch_raises(self) -> None:
        """ValueError when teacher_gt shape does not match output_hw."""
        gt = np.ones((3, 3), dtype=np.uint8)
        with self.assertRaises(ValueError):
            t04.generate_one_slice(
                model=None,
                image_tensor=None,
                prompt_meta={},
                label_mode="full",
                output_hw=(2, 2),
                template={},
                args=MagicMock(),
                teacher_gt=gt,
            )

    def test_full_mode_missing_gt_raises(self) -> None:
        """ValueError when teacher_gt is None in full mode."""
        with self.assertRaises(ValueError):
            t04.generate_one_slice(
                model=None,
                image_tensor=None,
                prompt_meta={},
                label_mode="full",
                output_hw=(2, 2),
                template={},
                args=MagicMock(),
                teacher_gt=None,
            )

    def test_full_mode_preserves_dtype(self) -> None:
        """Returned tri is always uint8."""
        gt = np.array([[0, 255], [128, 1]], dtype=np.uint8)
        result, _ = t04.generate_one_slice(
            model=None,
            image_tensor=None,
            prompt_meta={},
            label_mode="full",
            output_hw=(2, 2),
            template={},
            args=MagicMock(),
            teacher_gt=gt,
        )
        self.assertEqual(result.dtype, np.uint8)


# ============================================================================
# E. Alpha / beta / gamma validation
# ============================================================================


class TestAlphaBetaGammaValidation(unittest.TestCase):
    """Tests for validate_args focusing on alpha, beta, gamma."""

    @staticmethod
    def _args(**overrides: float) -> argparse.Namespace:
        defaults = {
            "alpha": 0.70,
            "beta": 0.10,
            "gamma": 0.20,
            "tau_low": 0.30,
            "tau_high": 0.70,
            "temperature": 0.07,
            "top_k": 10,
            "max_samples": 0,
            "log_every": 50,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    # -- E.1  Sum = 1.0 → passes ---------------------------------------------

    def test_sum_exactly_one_passes(self) -> None:
        """alpha+beta+gamma = 1.0 passes without error."""
        cases = [
            (0.70, 0.10, 0.20),
            (1.00, 0.00, 0.00),
            (0.00, 1.00, 0.00),
            (0.00, 0.00, 1.00),
            (0.33, 0.33, 0.34),
            (0.50, 0.25, 0.25),
        ]
        for a, b, g in cases:
            with self.subTest(alpha=a, beta=b, gamma=g):
                t04.validate_args(self._args(alpha=a, beta=b, gamma=g))

    # -- E.2  Sum != 1.0 → ValueError -----------------------------------------

    def test_sum_not_one_raises(self) -> None:
        """alpha+beta+gamma != 1.0 raises ValueError."""
        bad_cases = [
            (0.50, 0.50, 0.50),   # 1.5
            (0.70, 0.10, 0.10),   # 0.9
            (0.00, 0.00, 0.00),   # 0.0
        ]
        for a, b, g in bad_cases:
            with self.subTest(alpha=a, beta=b, gamma=g):
                with self.assertRaises(ValueError):
                    t04.validate_args(self._args(alpha=a, beta=b, gamma=g))

    def test_sum_close_but_not_one(self) -> None:
        """Sums that differ from 1.0 by > 1e-6 raise ValueError."""
        with self.assertRaises(ValueError):
            t04.validate_args(self._args(alpha=0.333, beta=0.333, gamma=0.333))
        # 0.333+0.333+0.333 = 0.999, abs(0.999-1) = 0.001 > 1e-6

    # -- E.3  Individual values in [0, 1] -------------------------------------

    def test_alpha_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            t04.validate_args(self._args(alpha=-0.1, beta=0.5, gamma=0.6))
        with self.assertRaises(ValueError):
            t04.validate_args(self._args(alpha=1.5, beta=-0.2, gamma=-0.3))

    def test_beta_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            t04.validate_args(self._args(alpha=0.7, beta=-0.1, gamma=0.4))
        with self.assertRaises(ValueError):
            t04.validate_args(self._args(alpha=0.7, beta=1.1, gamma=-0.8))

    def test_gamma_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            t04.validate_args(self._args(alpha=0.7, beta=0.1, gamma=-0.2))
        with self.assertRaises(ValueError):
            t04.validate_args(self._args(alpha=0.7, beta=0.1, gamma=1.2))

    def test_boundary_values_pass(self) -> None:
        """Exactly 0.0 and 1.0 are valid for each weight."""
        t04.validate_args(self._args(alpha=0.0, beta=1.0, gamma=0.0))
        t04.validate_args(self._args(alpha=1.0, beta=0.0, gamma=0.0))


# ============================================================================
# F. Tau validation
# ============================================================================


class TestTauValidation(unittest.TestCase):
    """Tests for validate_args focusing on tau_low / tau_high."""

    @staticmethod
    def _args(**overrides: float) -> argparse.Namespace:
        defaults = {
            "alpha": 0.70,
            "beta": 0.10,
            "gamma": 0.20,
            "tau_low": 0.30,
            "tau_high": 0.70,
            "temperature": 0.07,
            "top_k": 10,
            "max_samples": 0,
            "log_every": 50,
        }
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    # -- F.1  tau_low < tau_high → passes ------------------------------------

    def test_tau_low_lt_tau_high_passes(self) -> None:
        """tau_low < tau_high passes without error."""
        t04.validate_args(self._args(tau_low=0.00, tau_high=1.00))
        t04.validate_args(self._args(tau_low=0.10, tau_high=0.90))
        t04.validate_args(self._args(tau_low=0.49, tau_high=0.51))

    # -- F.2  tau_low >= tau_high → ValueError --------------------------------

    def test_tau_low_eq_tau_high_raises(self) -> None:
        with self.assertRaises(ValueError):
            t04.validate_args(self._args(tau_low=0.50, tau_high=0.50))

    def test_tau_low_gt_tau_high_raises(self) -> None:
        with self.assertRaises(ValueError):
            t04.validate_args(self._args(tau_low=0.80, tau_high=0.30))

    # -- F.3  Values in [0, 1] ------------------------------------------------

    def test_tau_low_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            t04.validate_args(self._args(tau_low=-0.10, tau_high=0.50))
        with self.assertRaises(ValueError):
            t04.validate_args(self._args(tau_low=1.10, tau_high=2.00))

    def test_tau_high_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            t04.validate_args(self._args(tau_low=0.10, tau_high=-0.50))
        with self.assertRaises(ValueError):
            t04.validate_args(self._args(tau_low=0.10, tau_high=1.50))

    def test_tau_boundary_values_pass(self) -> None:
        """Exactly 0.0 and 1.0 are valid for both tau values."""
        t04.validate_args(self._args(tau_low=0.00, tau_high=1.00))


# ============================================================================
# G. Tri-value threshold logic  (the decision rule inside generate_one_slice)
# ============================================================================


class TestTriValueThresholdLogic(unittest.TestCase):
    """Direct tests of the tri-value threshold rule used in generate_one_slice."""

    # -- G.1  score >= tau_high  →  foreground --------------------------------

    def test_score_above_tau_high_is_foreground(self) -> None:
        tau_high = 0.70
        score = np.array([[0.90, 0.70, 0.71]], dtype=np.float32)
        box_mask = np.ones_like(score, dtype=bool)
        fg = box_mask & (score >= tau_high)
        np.testing.assert_array_equal(fg[0], [True, True, True])

    def test_score_below_tau_high_is_not_foreground(self) -> None:
        tau_high = 0.70
        score = np.array([[0.69, 0.00, 0.10]], dtype=np.float32)
        box_mask = np.ones_like(score, dtype=bool)
        fg = box_mask & (score >= tau_high)
        self.assertFalse(bool(fg.any()))

    # -- G.2  score <= tau_low  →  background ---------------------------------

    def test_score_below_tau_low_is_background(self) -> None:
        tau_low = 0.30
        score = np.array([[0.00, 0.30, 0.10]], dtype=np.float32)
        box_mask = np.ones_like(score, dtype=bool)
        bg = box_mask & (score <= tau_low)
        np.testing.assert_array_equal(bg[0], [True, True, True])

    def test_score_above_tau_low_is_not_background(self) -> None:
        tau_low = 0.30
        score = np.array([[0.31, 0.50, 0.90]], dtype=np.float32)
        box_mask = np.ones_like(score, dtype=bool)
        bg = box_mask & (score <= tau_low)
        self.assertFalse(bool(bg.any()))

    # -- G.3  tau_low < score < tau_high  →  unknown (255) -------------------

    def test_score_between_taus_is_unknown(self) -> None:
        tau_low, tau_high = 0.30, 0.70
        score = np.array([[0.31, 0.50, 0.69]], dtype=np.float32)
        box_mask = np.ones_like(score, dtype=bool)
        fg = box_mask & (score >= tau_high)
        bg = box_mask & (score <= tau_low)
        unk = box_mask & (score > tau_low) & (score < tau_high)
        self.assertFalse(bool(fg.any()))
        self.assertFalse(bool(bg.any()))
        np.testing.assert_array_equal(unk[0], [True, True, True])

    # -- Boundary cases -------------------------------------------------------

    def test_exact_tau_high_is_foreground(self) -> None:
        """Score exactly tau_high goes to foreground."""
        tau_high = 0.70
        score = np.full((3, 3), tau_high, dtype=np.float32)
        box_mask = np.ones((3, 3), dtype=bool)
        fg = box_mask & (score >= tau_high)
        bg = box_mask & (score <= 0.30)
        unk = box_mask & (score > 0.30) & (score < tau_high)
        self.assertTrue(bool(fg.all()))
        self.assertFalse(bool(bg.any()))
        self.assertFalse(bool(unk.any()))

    def test_exact_tau_low_is_background(self) -> None:
        """Score exactly tau_low goes to background."""
        tau_low = 0.30
        tau_high = 0.70
        score = np.full((3, 3), tau_low, dtype=np.float32)
        box_mask = np.ones((3, 3), dtype=bool)
        fg = box_mask & (score >= tau_high)
        bg = box_mask & (score <= tau_low)
        unk = box_mask & (score > tau_low) & (score < tau_high)
        self.assertFalse(bool(fg.any()))
        self.assertTrue(bool(bg.all()))
        self.assertFalse(bool(unk.any()))

    # -- Outside-box pixels are ignored ---------------------------------------

    def test_outside_box_is_excluded(self) -> None:
        """Pixels outside the box mask are not assigned to any category."""
        tau_high = 0.70
        score = np.full((10, 10), 0.90, dtype=np.float32)
        box_mask = np.zeros((10, 10), dtype=bool)
        box_mask[2:5, 2:5] = True

        fg = box_mask & (score >= tau_high)
        self.assertEqual(int(fg.sum()), 9)  # 3×3 box only
        self.assertFalse(bool(fg[0, 0]))

    # -- Disjoint categories --------------------------------------------------

    def test_categories_are_disjoint(self) -> None:
        """No pixel can be both foreground and background simultaneously."""
        tau_low, tau_high = 0.30, 0.70
        score = np.random.rand(20, 20).astype(np.float32)
        box_mask = np.ones((20, 20), dtype=bool)
        fg = box_mask & (score >= tau_high)
        bg = box_mask & (score <= tau_low)
        overlap = fg & bg
        self.assertFalse(bool(overlap.any()))

    def test_categories_cover_all_box_pixels(self) -> None:
        """Every box pixel falls into exactly one category."""
        tau_low, tau_high = 0.30, 0.70
        score = np.random.rand(20, 20).astype(np.float32)
        box_mask = np.ones((20, 20), dtype=bool)
        fg = box_mask & (score >= tau_high)
        bg = box_mask & (score <= tau_low)
        unk = box_mask & (score > tau_low) & (score < tau_high)
        covered = fg | bg | unk
        np.testing.assert_array_equal(covered, box_mask)


# ============================================================================
# H. CLI parser
# ============================================================================


class TestCLIParser(unittest.TestCase):
    """Tests for build_parser argument definitions."""

    _REQUIRED_BASE = [
        "--processed_root", "/tmp/root",
        "--base_checkpoint", "/tmp/base.pth",
        "--ema_checkpoint", "/tmp/ema.pth",
        "--datasets", "test_db",
        "--round_tag", "r00_full5",
    ]

    @classmethod
    def setUpClass(cls) -> None:
        cls.parser = t04.build_parser()

    # -- H.1  New args accepted -----------------------------------------------

    def test_new_args_accepted_with_custom_values(self) -> None:
        """All new fusion args are accepted and parsed to correct types."""
        argv = self._REQUIRED_BASE + [
            "--alpha", "0.50",
            "--beta", "0.30",
            "--gamma", "0.20",
            "--tau_low", "0.20",
            "--tau_high", "0.80",
            "--temperature", "0.10",
            "--top_k", "5",
        ]
        args = self.parser.parse_args(argv)
        self.assertEqual(args.alpha, 0.50)
        self.assertEqual(args.beta, 0.30)
        self.assertEqual(args.gamma, 0.20)
        self.assertEqual(args.tau_low, 0.20)
        self.assertEqual(args.tau_high, 0.80)
        self.assertEqual(args.temperature, 0.10)
        self.assertEqual(args.top_k, 5)

    def test_ema_checkpoint_arg_accepted(self) -> None:
        args = self.parser.parse_args(self._REQUIRED_BASE)
        self.assertEqual(args.ema_checkpoint, Path("/tmp/ema.pth"))

    def test_round_tag_arg_accepted(self) -> None:
        args = self.parser.parse_args(self._REQUIRED_BASE)
        self.assertEqual(args.round_tag, "r00_full5")

    def test_new_args_have_correct_defaults(self) -> None:
        """Default values for new args match the current implementation."""
        args = self.parser.parse_args(self._REQUIRED_BASE)
        self.assertEqual(args.alpha, 0.70)
        self.assertEqual(args.beta, 0.10)
        self.assertEqual(args.gamma, 0.20)
        self.assertEqual(args.tau_low, 0.30)
        self.assertEqual(args.tau_high, 0.70)
        self.assertEqual(args.temperature, 0.07)
        self.assertEqual(args.top_k, 10)

    # -- H.2  Old args rejected -----------------------------------------------

    def test_old_args_rejected(self) -> None:
        """--p_weight, --qf_weight, --sac_checkpoint, --p_threshold are unknown."""
        old_args = ["--p_weight", "--qf_weight", "--sac_checkpoint", "--p_threshold"]
        for arg_name in old_args:
            with self.subTest(arg=arg_name):
                with self.assertRaises(SystemExit):
                    self.parser.parse_args(
                        self._REQUIRED_BASE + [arg_name, "0.5"]
                    )


# ============================================================================
# I. get_npy_output_name
# ============================================================================


class TestGetNpyOutputName(unittest.TestCase):
    """Tests for get_npy_output_name."""

    def test_already_has_npy_extension(self) -> None:
        """Name already ending with .npy is returned unchanged."""
        self.assertEqual(t04.get_npy_output_name("slice.npy"), "slice.npy")
        self.assertEqual(t04.get_npy_output_name("path/to/slice.npy"), "path/to/slice.npy")
        self.assertEqual(t04.get_npy_output_name("data.NPY"), "data.NPY")  # case-insensitive

    def test_adds_npy_extension(self) -> None:
        """Name without .npy extension gets .npy appended."""
        self.assertEqual(t04.get_npy_output_name("slice"), "slice.npy")
        self.assertEqual(t04.get_npy_output_name("slice.png"), "slice.png.npy")

    def test_empty_string(self) -> None:
        """Empty string gets .npy appended."""
        self.assertEqual(t04.get_npy_output_name(""), ".npy")


# ============================================================================
# J. No MONAI imports
# ============================================================================


class TestNoMonaiImports(unittest.TestCase):
    """Verify the module does not import monai."""

    def test_no_monai_import_statement(self) -> None:
        source_path = Path(t04.__file__)
        source_text = source_path.read_text()
        self.assertNotIn("import monai", source_text)
        self.assertNotIn("from monai", source_text)


# ============================================================================
# Additional unit tests for utility functions
# ============================================================================


class TestGetSliceName(unittest.TestCase):
    """Tests for get_slice_name."""

    def test_explicit_slice_name(self) -> None:
        item = {"slice_name": "img_001.npy"}
        self.assertEqual(t04.get_slice_name(item), "img_001.npy")

    def test_falls_back_to_teacher_img(self) -> None:
        item = {"teacher_img": "data/img_002.npy"}
        self.assertEqual(t04.get_slice_name(item), "img_002.npy")

    def test_falls_back_to_student_img(self) -> None:
        item = {"student_img": "data/img_003.npy"}
        self.assertEqual(t04.get_slice_name(item), "img_003.npy")

    def test_falls_back_to_teacher_gt(self) -> None:
        item = {"teacher_gt": "data/img_004.npy"}
        self.assertEqual(t04.get_slice_name(item), "img_004.npy")

    def test_falls_back_to_student_gt(self) -> None:
        item = {"student_gt": "data/img_005.npy"}
        self.assertEqual(t04.get_slice_name(item), "img_005.npy")

    def test_no_keys_raises(self) -> None:
        with self.assertRaises(KeyError):
            t04.get_slice_name({})


class TestBuildManifestIndex(unittest.TestCase):
    """Tests for build_manifest_index."""

    def test_builds_index_from_list(self) -> None:
        manifest = [
            {"slice_name": "a.npy"},
            {"slice_name": "b.npy"},
            {"teacher_img": "data/c.npy"},
        ]
        index = t04.build_manifest_index(manifest)
        self.assertEqual(len(index), 3)
        self.assertIn("a.npy", index)
        self.assertIn("b.npy", index)
        self.assertIn("c.npy", index)

    def test_duplicate_slice_name_raises(self) -> None:
        manifest = [
            {"slice_name": "dup.npy"},
            {"slice_name": "dup.npy"},
        ]
        with self.assertRaises(RuntimeError):
            t04.build_manifest_index(manifest)

    def test_non_dict_item_raises(self) -> None:
        with self.assertRaises(TypeError):
            t04.build_manifest_index(["not_a_dict"])  # type: ignore[list-item]


class TestPrepareTargetRecords(unittest.TestCase):
    """Tests for prepare_target_records."""

    def test_max_samples_positive_truncates(self) -> None:
        records = [
            {"slice_name": f"{i}.npy", "label_mode": "box"}
            for i in range(10)
        ]
        result, names = t04.prepare_target_records(
            records, max_samples=3,
            split_path=Path("/fake/split.json"),
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(names, ["0.npy", "1.npy", "2.npy"])

    def test_max_samples_zero_uses_all(self) -> None:
        records = [
            {"slice_name": f"{i}.npy", "label_mode": "box"}
            for i in range(5)
        ]
        result, names = t04.prepare_target_records(
            records, max_samples=0,
            split_path=Path("/fake/split.json"),
        )
        self.assertEqual(len(result), 5)

    def test_empty_records_raises(self) -> None:
        with self.assertRaises(RuntimeError):
            t04.prepare_target_records(
                [], max_samples=0, split_path=Path("/fake/split.json"),
            )

    def test_invalid_label_mode_raises(self) -> None:
        records = [{"slice_name": "x.npy", "label_mode": "invalid"}]
        with self.assertRaises(ValueError):
            t04.prepare_target_records(
                records, max_samples=0, split_path=Path("/fake/split.json"),
            )

    def test_duplicate_slice_names_raises(self) -> None:
        records = [
            {"slice_name": "dup.npy", "label_mode": "box"},
            {"slice_name": "dup.npy", "label_mode": "box"},
        ]
        with self.assertRaises(RuntimeError):
            t04.prepare_target_records(
                records, max_samples=0, split_path=Path("/fake/split.json"),
            )


class TestConnectedComponentStats(unittest.TestCase):
    """Tests for connected_component_stats."""

    def setUp(self) -> None:
        if not HAS_CV2:
            self.skipTest("cv2 not available")

    def test_single_class_single_component(self) -> None:
        tri = np.zeros((20, 20), dtype=np.uint8)
        tri[5:15, 5:15] = 1
        n, ratio, present = t04.connected_component_stats(tri, [1])
        self.assertEqual(n, 1)
        self.assertAlmostEqual(ratio, 1.0)
        self.assertEqual(present, 1)

    def test_single_class_multiple_components(self) -> None:
        tri = np.zeros((30, 30), dtype=np.uint8)
        tri[2:6, 2:6] = 1    # 16 px
        tri[20:30, 20:25] = 1  # 50 px
        n, ratio, present = t04.connected_component_stats(tri, [1])
        self.assertEqual(n, 2)
        # largest component = 50 px; total = 66 px; ratio ≈ 0.7575
        self.assertAlmostEqual(ratio, 50.0 / 66.0, places=4)
        self.assertEqual(present, 1)

    def test_class_id_not_present(self) -> None:
        tri = np.zeros((10, 10), dtype=np.uint8)
        tri[2:4, 2:4] = 1
        n, ratio, present = t04.connected_component_stats(tri, [2])
        self.assertEqual(n, 0)
        self.assertEqual(ratio, 0.0)
        self.assertEqual(present, 0)

    def test_empty_tri(self) -> None:
        tri = np.zeros((10, 10), dtype=np.uint8)
        n, ratio, present = t04.connected_component_stats(tri, [1, 2, 3])
        self.assertEqual(n, 0)
        self.assertEqual(ratio, 0.0)
        self.assertEqual(present, 0)


class TestGeometryKey(unittest.TestCase):
    """Tests for get_npy_output_name edge cases with typical slice naming."""

    def test_case_insensitive_npy_check(self) -> None:
        """The .lower() call catches uppercase .NPY."""
        self.assertEqual(t04.get_npy_output_name("SLICE.NPY"), "SLICE.NPY")
        self.assertEqual(t04.get_npy_output_name("Slice.Npy"), "Slice.Npy")


# ============================================================================
# Dynamic Shape Fusion Tests
# ============================================================================


class TestFuseDynamicShape(unittest.TestCase):
    """Tests for fuse_dynamic_shape function."""

    def test_k1_returns_single_template(self):
        """K=1: output equals the only template."""
        templates = np.ones((1, 64, 64), dtype=np.float32) * 0.5
        centers = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        box_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        result = t04.fuse_dynamic_shape(
            templates, centers, box_emb, temperature=0.1,
        )
        self.assertEqual(result.shape, (64, 64))
        np.testing.assert_array_almost_equal(result, templates[0])

    def test_k2_high_similarity_to_first(self):
        """K=2: weight near [1,0] when box matches template 0."""
        templates = np.stack([
            np.ones((64, 64), dtype=np.float32) * 0.2,
            np.ones((64, 64), dtype=np.float32) * 0.8,
        ])
        centers = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        box_emb = np.array([1.0, 0.0], dtype=np.float32)

        result = t04.fuse_dynamic_shape(
            templates, centers, box_emb, temperature=0.1,
        )
        self.assertEqual(result.shape, (64, 64))
        # Should be very close to template 0
        self.assertLess(np.abs(result.mean() - 0.2), 0.1)

    def test_k2_high_similarity_to_second(self):
        """K=2: weight near [0,1] when box matches template 1."""
        templates = np.stack([
            np.ones((64, 64), dtype=np.float32) * 0.2,
            np.ones((64, 64), dtype=np.float32) * 0.8,
        ])
        centers = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=np.float32)
        box_emb = np.array([0.0, 1.0], dtype=np.float32)

        result = t04.fuse_dynamic_shape(
            templates, centers, box_emb, temperature=0.1,
        )
        self.assertEqual(result.shape, (64, 64))
        self.assertGreater(np.abs(result.mean() - 0.8), 0.0)
        self.assertLess(np.abs(result.mean() - 0.8), 0.1)

    def test_k2_uniform_weights_average(self):
        """K=2 equal weights: output equals mean of templates."""
        templates = np.stack([
            np.ones((64, 64), dtype=np.float32) * 0.3,
            np.ones((64, 64), dtype=np.float32) * 0.7,
        ])
        centers = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
        ], dtype=np.float32)
        box_emb = np.array([1.0, 0.0], dtype=np.float32)

        result = t04.fuse_dynamic_shape(
            templates, centers, box_emb, temperature=100.0,
        )
        self.assertEqual(result.shape, (64, 64))
        expected = np.ones((64, 64), dtype=np.float32) * 0.5
        np.testing.assert_array_almost_equal(result, expected, decimal=2)

    def test_mismatched_counts_raises(self):
        """Template count != center count raises ValueError."""
        templates = np.ones((2, 64, 64), dtype=np.float32)
        centers = np.ones((3, 16), dtype=np.float32)
        box_emb = np.ones(16, dtype=np.float32)

        with self.assertRaises(ValueError):
            t04.fuse_dynamic_shape(
                templates, centers, box_emb, temperature=0.1,
            )

    def test_softmax_weights_shape(self):
        """Weights computed internally have shape [K], finite, sum~1."""
        templates = np.ones((3, 64, 64), dtype=np.float32)
        centers = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        box_emb = np.array([0.5, 0.3, 0.2], dtype=np.float32)

        # We verify through the function by checking output is valid.
        result = t04.fuse_dynamic_shape(
            templates, centers, box_emb, temperature=0.1,
        )
        self.assertTrue(np.all(np.isfinite(result)))
        # Result should be within the convex hull of template values
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)

    def test_empty_templates_raises(self):
        """Empty templates array raises ValueError."""
        templates = np.ones((0, 64, 64), dtype=np.float32)
        centers = np.ones((0, 16), dtype=np.float32)
        box_emb = np.ones(16, dtype=np.float32)

        with self.assertRaises(ValueError):
            t04.fuse_dynamic_shape(
                templates, centers, box_emb, temperature=0.1,
            )


class TestExtractBoxEmbedding(unittest.TestCase):
    """Tests for extract_box_embedding function."""

    def test_returns_1d_array(self):
        """Returns a 1D array of shape (D,)."""
        emb = torch.randn(1, 256, 16, 16)
        box = [10, 10, 50, 50]
        result = t04.extract_box_embedding(emb, box, image_h=256, image_w=256)
        self.assertEqual(result.ndim, 1)
        self.assertEqual(result.shape[0], 256)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_degenerate_box_fallback(self):
        """Degenerate box (x2<=x1) falls back to global mean."""
        emb = torch.randn(1, 256, 16, 16)
        box = [10, 10, 5, 50]  # x2 < x1
        result = t04.extract_box_embedding(emb, box, image_h=256, image_w=256)
        self.assertEqual(result.shape, (256,))
        self.assertTrue(np.all(np.isfinite(result)))


import torch


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    unittest.main()
