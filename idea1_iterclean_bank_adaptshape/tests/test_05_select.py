from __future__ import annotations

import argparse
import importlib
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

t05 = importlib.import_module(
    "idea1_iterclean_bank_adaptshape.05_select_hard_by_gt_iou"
)


# ---------------------------------------------------------------------------
# Test A: compute_binary_metrics
# ---------------------------------------------------------------------------

class TestComputeBinaryMetrics(unittest.TestCase):
    """A. compute_binary_metrics (synthetic 2D masks)."""

    def test_perfect_prediction(self):
        """Perfect prediction yields IoU=1.0, Dice=1.0."""
        gt = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ], dtype=np.uint8)

        pseudo = gt.copy()
        iou, dice, inter, pred_px, gt_px = t05.compute_binary_metrics(pseudo, gt)

        self.assertAlmostEqual(iou, 1.0)
        self.assertAlmostEqual(dice, 1.0)
        self.assertEqual(inter, 4)
        self.assertEqual(pred_px, 4)
        self.assertEqual(gt_px, 4)

    def test_completely_wrong(self):
        """Non-overlapping foregrounds yield IoU=0.0, Dice=0.0."""
        gt = np.array([
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
        ], dtype=np.uint8)

        pseudo = np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 1, 0],
        ], dtype=np.uint8)

        iou, dice, inter, pred_px, gt_px = t05.compute_binary_metrics(pseudo, gt)

        self.assertAlmostEqual(iou, 0.0)
        self.assertAlmostEqual(dice, 0.0)
        self.assertEqual(inter, 0)

    def test_partial_overlap(self):
        """50 % overlap yields correct IoU=1/3 and Dice=0.5."""
        gt = np.array([
            [0, 0, 0],
            [0, 1, 1],
            [0, 1, 1],
        ], dtype=np.uint8)

        # Shift one column to the right
        pseudo = np.array([
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
        ], dtype=np.uint8)

        iou, dice, inter, pred_px, gt_px = t05.compute_binary_metrics(pseudo, gt)

        # Intersection = 2, Union = 4+2-2 = 4, IoU = 0.5
        # Pred = 2, GT = 4, Denom = 6, Dice = 4/6 = 2/3
        self.assertAlmostEqual(iou, 0.5)
        self.assertAlmostEqual(dice, 2.0 / 3.0)
        self.assertEqual(inter, 2)
        self.assertEqual(pred_px, 2)
        self.assertEqual(gt_px, 4)

    def test_both_empty(self):
        """Both masks empty -> IoU=1.0, Dice=1.0 (edge case)."""
        gt = np.zeros((5, 5), dtype=np.uint8)
        pseudo = np.zeros((5, 5), dtype=np.uint8)

        iou, dice, inter, pred_px, gt_px = t05.compute_binary_metrics(pseudo, gt)

        self.assertAlmostEqual(iou, 1.0)
        self.assertAlmostEqual(dice, 1.0)
        self.assertEqual(inter, 0)
        self.assertEqual(pred_px, 0)
        self.assertEqual(gt_px, 0)

    def test_unknown_label_treated_as_background(self):
        """UNKNOWN_LABEL (255) pixels are excluded from foreground."""
        gt = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=np.uint8)

        pseudo = np.array([
            [0, 255, 0],
            [255, 1, 255],
            [0, 255, 0],
        ], dtype=np.uint8)

        # pred_fg excludes 255, so only (1,1) counts
        iou, dice, inter, pred_px, gt_px = t05.compute_binary_metrics(pseudo, gt)

        # pred_fg pixels: only 1 at (1,1), gt_fg=5
        # intersection=1 (at (1,1)), union=1+5-1=5
        # IoU = 0.2, Dice = 2/6 = 1/3
        self.assertAlmostEqual(iou, 0.2)
        self.assertAlmostEqual(dice, 1.0 / 3.0)
        self.assertEqual(inter, 1)
        self.assertEqual(pred_px, 1)
        self.assertEqual(gt_px, 5)

    def test_pred_background_gt_foreground(self):
        """Pseudo all-zero with foreground GT yields IoU=0, Dice=0."""
        gt = np.ones((4, 4), dtype=np.uint8)
        pseudo = np.zeros((4, 4), dtype=np.uint8)

        iou, dice, _inter, _p, _g = t05.compute_binary_metrics(pseudo, gt)
        self.assertAlmostEqual(iou, 0.0)
        self.assertAlmostEqual(dice, 0.0)

    def test_pred_foreground_gt_background(self):
        """Pseudo all-ones with empty GT yields IoU=0, Dice=0."""
        gt = np.zeros((3, 3), dtype=np.uint8)
        pseudo = np.ones((3, 3), dtype=np.uint8)

        iou, dice, _inter, _p, _g = t05.compute_binary_metrics(pseudo, gt)
        self.assertAlmostEqual(iou, 0.0)
        self.assertAlmostEqual(dice, 0.0)

    def test_non_square_mask(self):
        """Works with rectangular masks."""
        gt = np.array([
            [0, 1, 1, 1],
            [0, 1, 1, 1],
        ], dtype=np.uint8)

        pseudo = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ], dtype=np.uint8)

        iou, dice, inter, pred_px, gt_px = t05.compute_binary_metrics(pseudo, gt)

        # intersection = 4, union = 4+6-4 = 6
        self.assertAlmostEqual(iou, 4.0 / 6.0)
        self.assertAlmostEqual(dice, 8.0 / 10.0)
        self.assertEqual(inter, 4)


# ---------------------------------------------------------------------------
# Test B: compute_macro_class_iou
# ---------------------------------------------------------------------------

class TestComputeMacroClassIou(unittest.TestCase):
    """B. compute_macro_class_iou (multi-class synthetic masks)."""

    def test_both_classes_perfect(self):
        """Two classes both perfect -> macro IoU=1.0."""
        gt = np.array([
            [0, 1, 1, 2],
            [0, 1, 1, 2],
            [0, 1, 1, 2],
        ], dtype=np.uint8)

        pseudo = gt.copy()
        macro_iou, num_present = t05.compute_macro_class_iou(
            pseudo, gt, class_ids=[1, 2],
        )

        self.assertAlmostEqual(macro_iou, 1.0)
        self.assertEqual(num_present, 2)

    def test_one_class_perfect_one_absent(self):
        """One class perfect, one absent (nan handling)."""
        gt = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ], dtype=np.uint8)

        pseudo = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ], dtype=np.uint8)

        # class_ids=[1, 2] but class 2 is absent from gt
        macro_iou, num_present = t05.compute_macro_class_iou(
            pseudo, gt, class_ids=[1, 2],
        )

        # Only class 1 present -> macro IoU = 1.0
        self.assertAlmostEqual(macro_iou, 1.0)
        self.assertEqual(num_present, 1)

    def test_all_classes_absent(self):
        """No class present in gt -> returns nan, 0."""
        gt = np.zeros((4, 4), dtype=np.uint8)
        pseudo = np.zeros((4, 4), dtype=np.uint8)

        macro_iou, num_present = t05.compute_macro_class_iou(
            pseudo, gt, class_ids=[1, 2, 3],
        )

        self.assertTrue(math.isnan(macro_iou))
        self.assertEqual(num_present, 0)

    def test_255_pixels_excluded(self):
        """255 pixels in pseudo are excluded from class calculations."""
        gt = np.array([
            [0, 1, 2, 0],
            [1, 1, 2, 0],
        ], dtype=np.uint8)

        # pseudo has 255 where gt has 2 — class 2 IoU should be 0
        pseudo = np.array([
            [0, 1, 255, 0],
            [1, 1, 255, 0],
        ], dtype=np.uint8)

        macro_iou, num_present = t05.compute_macro_class_iou(
            pseudo, gt, class_ids=[1, 2],
        )

        # class 1 IoU = 1.0 (perfect), class 2 IoU = 0.0 (no pred)
        expected = (1.0 + 0.0) / 2.0
        self.assertAlmostEqual(macro_iou, expected)
        self.assertEqual(num_present, 2)

    def test_partial_overlap_multi_class(self):
        """Partial overlap across classes yields fractional IoU."""
        gt = np.array([
            [1, 1, 2],
            [1, 1, 2],
        ], dtype=np.uint8)

        pseudo = np.array([
            [1, 1, 1],
            [1, 2, 2],
        ], dtype=np.uint8)

        macro_iou, num_present = t05.compute_macro_class_iou(
            pseudo, gt, class_ids=[1, 2],
        )

        # class 1: intersection=3 (positions (0,0),(0,1),(1,0)),
        #   union=5 (pred=4+gt=4-inter=3=5); IoU=3/5=0.6
        # class 2: intersection=1 (pos (1,2)),
        #   union=3 (pred=2+gt=2-inter=1=3); IoU=1/3
        expected = (0.6 + 1.0 / 3.0) / 2.0
        self.assertAlmostEqual(macro_iou, expected)
        self.assertEqual(num_present, 2)


# ---------------------------------------------------------------------------
# Test C: compute_slice_metrics
# ---------------------------------------------------------------------------

class TestComputeSliceMetrics(unittest.TestCase):
    """C. compute_slice_metrics returns all expected keys and validates inputs."""

    def _make_mask(self, shape=(4, 6)):
        return np.zeros(shape, dtype=np.uint8)

    def test_returns_all_expected_keys(self):
        """Result dict contains all expected key names."""
        gt = self._make_mask()
        gt[1:3, 1:4] = 1

        pseudo = gt.copy()

        result = t05.compute_slice_metrics(pseudo, gt, class_ids=[1], is_multiclass=False)

        expected_keys = {
            "main_iou", "binary_fg_iou", "macro_class_iou",
            "dice", "intersection_pixels", "pseudo_fg_pixels",
            "gt_fg_pixels", "unknown_pixels", "unknown_ratio",
            "gt_has_foreground", "num_present_classes",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_binary_mode_main_iou_equals_binary_iou(self):
        """In binary mode, main_iou == binary_fg_iou."""
        gt = self._make_mask()
        gt[0, 0] = 1
        pseudo = gt.copy()

        result = t05.compute_slice_metrics(pseudo, gt, class_ids=[1], is_multiclass=False)
        self.assertAlmostEqual(result["main_iou"], result["binary_fg_iou"])

    def test_multiclass_mode_main_iou_equals_macro(self):
        """In multiclass mode with valid macro, main_iou == macro_class_iou."""
        gt = self._make_mask()
        gt[0, 0] = 1
        gt[2, 3] = 2
        pseudo = gt.copy()

        result = t05.compute_slice_metrics(pseudo, gt, class_ids=[1, 2], is_multiclass=True)
        self.assertAlmostEqual(result["main_iou"], result["macro_class_iou"])
        self.assertAlmostEqual(result["main_iou"], 1.0)

    def test_multiclass_but_macro_nan_falls_back_to_binary(self):
        """When macro is NaN (no classes present), main_iou falls back to binary."""
        gt = np.zeros((4, 4), dtype=np.uint8)
        pseudo = np.zeros((4, 4), dtype=np.uint8)

        result = t05.compute_slice_metrics(pseudo, gt, class_ids=[1, 2], is_multiclass=True)

        self.assertTrue(math.isnan(result["macro_class_iou"]))
        self.assertAlmostEqual(result["main_iou"], result["binary_fg_iou"])

    def test_invalid_values_in_pseudo_raise_valueerror(self):
        """Pseudo containing values outside {0, 255, class_ids} raises ValueError."""
        gt = np.array([
            [0, 1, 1],
            [0, 1, 1],
        ], dtype=np.uint8)
        pseudo = np.array([
            [0, 1, 3],  # 3 is invalid
            [0, 1, 1],
        ], dtype=np.uint8)

        with self.assertRaises(ValueError) as ctx:
            t05.compute_slice_metrics(pseudo, gt, class_ids=[1], is_multiclass=False)
        self.assertIn("invalid values", str(ctx.exception))

    def test_shape_mismatch_raises_valueerror(self):
        """Different shapes raise ValueError."""
        gt = np.zeros((4, 4), dtype=np.uint8)
        pseudo = np.zeros((5, 5), dtype=np.uint8)

        with self.assertRaises(ValueError) as ctx:
            t05.compute_slice_metrics(pseudo, gt, class_ids=[1], is_multiclass=False)
        self.assertIn("shape mismatch", str(ctx.exception).lower())

    def test_unknown_pixels_tracked(self):
        """unknown_pixels and unknown_ratio are correctly reported."""
        gt = np.zeros((3, 5), dtype=np.uint8)
        pseudo = np.full((3, 5), t05.UNKNOWN_LABEL, dtype=np.uint8)

        result = t05.compute_slice_metrics(pseudo, gt, class_ids=[1], is_multiclass=False)

        total = pseudo.size  # 15
        self.assertEqual(result["unknown_pixels"], total)
        self.assertAlmostEqual(result["unknown_ratio"], 1.0)

    def test_gt_has_foreground_flag(self):
        """gt_has_foreground reflects GT content."""
        gt_fg = np.array([
            [0, 1],
            [0, 0],
        ], dtype=np.uint8)
        gt_empty = np.zeros((2, 2), dtype=np.uint8)

        result_fg = t05.compute_slice_metrics(gt_fg.copy(), gt_fg, class_ids=[1], is_multiclass=False)
        self.assertTrue(result_fg["gt_has_foreground"])

        result_empty = t05.compute_slice_metrics(gt_empty.copy(), gt_empty, class_ids=[1], is_multiclass=False)
        self.assertFalse(result_empty["gt_has_foreground"])

    def test_intersection_pixels_type(self):
        """intersection_pixels is an int."""
        gt = np.ones((3, 3), dtype=np.uint8)
        pseudo = gt.copy()

        result = t05.compute_slice_metrics(pseudo, gt, class_ids=[1], is_multiclass=False)
        self.assertIsInstance(result["intersection_pixels"], int)
        self.assertEqual(result["intersection_pixels"], 9)

    def test_pseudo_with_unknown_and_valid_mixed(self):
        """Mixed pseudo with 0, 255, and class_ids works correctly."""
        gt = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ], dtype=np.uint8)

        pseudo = np.array([
            [0, 255, 1, 0],
            [0, 1, 255, 0],
        ], dtype=np.uint8)

        # Should not raise
        result = t05.compute_slice_metrics(pseudo, gt, class_ids=[1], is_multiclass=False)
        self.assertEqual(result["unknown_pixels"], 2)  # two 255 pixels
        # pred_fg pixels: two 1s (at (0,2) and (1,1))
        self.assertEqual(result["pseudo_fg_pixels"], 2)


# ---------------------------------------------------------------------------
# Test D: worst20_mean
# ---------------------------------------------------------------------------

class TestWorst20Mean(unittest.TestCase):
    """D. worst20_mean behaviour."""

    def test_10_values_worst_2(self):
        """10 values -> mean of worst 2 (20 %)."""
        values = [0.9, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.7, 0.6, 1.0]
        result = t05.worst20_mean(values)
        # Worst 2: 0.1, 0.2 -> mean = 0.15
        self.assertAlmostEqual(result, 0.15)

    def test_single_value(self):
        """Single value -> that value."""
        result = t05.worst20_mean([0.42])
        self.assertAlmostEqual(result, 0.42)

    def test_empty_list(self):
        """Empty list -> nan."""
        result = t05.worst20_mean([])
        self.assertTrue(math.isnan(result))

    def test_all_equal(self):
        """All equal values -> same value."""
        values = [0.55] * 7
        result = t05.worst20_mean(values)
        # 20% of 7 = ceil(1.4) = 2, mean of two 0.55s = 0.55
        self.assertAlmostEqual(result, 0.55)

    def test_5_values_worst_1(self):
        """5 values, 20% = ceil(1) = 1 worst."""
        values = [0.9, 0.8, 0.3, 0.7, 1.0]
        result = t05.worst20_mean(values)
        # worst 1 = 0.3
        self.assertAlmostEqual(result, 0.3)

    def test_6_values_worst_2(self):
        """6 values, 20% = ceil(1.2) = 2 worst."""
        values = [0.9, 0.8, 0.3, 0.7, 1.0, 0.6]
        result = t05.worst20_mean(values)
        # Worst 2: 0.3, 0.6 -> mean = 0.45
        self.assertAlmostEqual(result, 0.45)

    def test_negative_values(self):
        """Works with negative values."""
        values = [-0.5, -1.0, 0.5, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        # worst 2: -1.0, -0.5 -> mean = -0.75
        result = t05.worst20_mean(values)
        self.assertAlmostEqual(result, -0.75)


# ---------------------------------------------------------------------------
# Test E: 2D selection logic
# ---------------------------------------------------------------------------

class Test2DSelectionLogic(unittest.TestCase):
    """E. 2D selection logic with synthetic rows."""

    def _make_rows(self, n=20):
        """Create n synthetic slice metric rows with varying main_iou."""
        np.random.seed(42)
        rows: list[dict] = []
        for i in range(n):
            main_iou = 0.1 + 0.9 * (i / n)  # ascending: 0.1 to ~0.955
            dice = main_iou + np.random.uniform(-0.05, 0.05)
            rows.append({
                "slice_name": f"slice_{i:04d}",
                "case_id": "unknown_case",
                "label_mode": "box",
                "main_iou": float(main_iou),
                "dice": float(max(0, min(1, dice))),
                "binary_fg_iou": float(main_iou),
                "macro_class_iou": float(main_iou),
                "gt_has_foreground": True,
                "num_present_classes": 1,
                "gt_fg_pixels": 100,
                "pseudo_fg_pixels": 90 + i,
                "unknown_pixels": 0,
                "unknown_ratio": 0.0,
                "intersection_pixels": 50,
            })
        return rows

    def test_select_lowest_5_from_20(self):
        """Select lowest 5 IoU images from 20 box samples."""
        rows = self._make_rows(20)
        select_count = 5

        # Sort same way as the 2D branch
        rows.sort(
            key=lambda row: (
                float(row["main_iou"]),
                float(row["dice"]),
                str(row["slice_name"]),
            )
        )

        selected = {
            str(row["slice_name"])
            for row in rows[: select_count]
        }

        self.assertEqual(len(selected), select_count)
        # Verify the selected are actually the lowest IoU
        selected_rows = [row for row in rows if row["slice_name"] in selected]
        all_ious = sorted(float(r["main_iou"]) for r in rows)
        selected_ious = sorted(float(r["main_iou"]) for r in selected_rows)
        self.assertEqual(selected_ious, all_ious[:select_count])

    def test_selected_slices_are_unique(self):
        """Selected slices contain no duplicates."""
        rows = self._make_rows(20)
        rows.sort(
            key=lambda row: (
                float(row["main_iou"]),
                float(row["dice"]),
                str(row["slice_name"]),
            )
        )
        selected = [row["slice_name"] for row in rows[:6]]
        self.assertEqual(len(selected), len(set(selected)))

    def test_no_overlap_with_full_slices(self):
        """Selected box slices do not overlap with current full slices."""
        rows = self._make_rows(20)
        # Mark first 3 as 'full' (already annotated)
        full_slice_names = {"slice_0000", "slice_0001", "slice_0002"}

        box_slices = [
            row for row in rows
            if row["slice_name"] not in full_slice_names
        ]

        # Sort box slices
        box_slices.sort(
            key=lambda row: (
                float(row["main_iou"]),
                float(row["dice"]),
                str(row["slice_name"]),
            )
        )

        select_count = 4
        selected_names = {
            row["slice_name"] for row in box_slices[:select_count]
        }

        # No overlap with full
        self.assertTrue(selected_names.isdisjoint(full_slice_names))
        self.assertEqual(len(selected_names), select_count)

    def test_slice_level_metrics_retrievable(self):
        """Each row used for selection has the required metric fields."""
        rows = self._make_rows(10)
        required_keys = {"slice_name", "main_iou", "dice"}
        for row in rows:
            self.assertTrue(required_keys.issubset(row.keys()))


# ---------------------------------------------------------------------------
# Test F: 3D case aggregation
# ---------------------------------------------------------------------------

class Test3DCaseAggregation(unittest.TestCase):
    """F. 3D case aggregation from synthetic data."""

    def _make_slice_row(self, slice_name, case_id, main_iou, has_fg=True):
        return {
            "slice_name": slice_name,
            "case_id": case_id,
            "main_iou": float(main_iou),
            "gt_has_foreground": has_fg,
            "dice": float(main_iou),
            "gt_fg_pixels": 100 if has_fg else 0,
        }

    def test_case_level_mean_iou_correct(self):
        """Case-level mean IoU is computed from valid foreground slices only."""
        rows = [
            self._make_slice_row("s01", "cA", 0.6),
            self._make_slice_row("s02", "cA", 0.8),
            self._make_slice_row("s03", "cA", 0.4),
            self._make_slice_row("s04", "cB", 0.9),
            self._make_slice_row("s05", "cB", 0.3),
        ]

        # Group by case_id
        from collections import defaultdict
        case_to_rows: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            case_to_rows[str(row["case_id"])].append(row)

        case_mean_ious = {}
        for case_id, case_rows in case_to_rows.items():
            valid = [r for r in case_rows if bool(r["gt_has_foreground"])]
            scores = [float(r["main_iou"]) for r in valid]
            case_mean_ious[case_id] = float(np.mean(scores))

        self.assertAlmostEqual(case_mean_ious["cA"], (0.6 + 0.8 + 0.4) / 3.0)
        self.assertAlmostEqual(case_mean_ious["cB"], (0.9 + 0.3) / 2.0)

    def test_only_valid_foreground_slices_used(self):
        """Empty slices (no foreground) are excluded from case scoring."""
        rows = [
            self._make_slice_row("s01", "cX", 0.9),
            self._make_slice_row("s02", "cX", 0.1, has_fg=False),  # empty
            self._make_slice_row("s03", "cX", 0.5),
        ]

        from collections import defaultdict
        case_to_rows: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            case_to_rows[str(row["case_id"])].append(row)

        case_rows = case_to_rows["cX"]
        valid_rows = [r for r in case_rows if bool(r["gt_has_foreground"])]
        scores = [float(r["main_iou"]) for r in valid_rows]

        self.assertEqual(len(valid_rows), 2)
        self.assertAlmostEqual(float(np.mean(scores)), 0.7)

    def test_empty_slices_excluded_from_mean(self):
        """Cases with all empty slices get nan mean and valid_for_selection=False."""
        rows = [
            self._make_slice_row("s01", "cEmpty", 0.0, has_fg=False),
            self._make_slice_row("s02", "cEmpty", 0.0, has_fg=False),
        ]

        from collections import defaultdict
        case_to_rows: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            case_to_rows[str(row["case_id"])].append(row)

        case_rows = case_to_rows["cEmpty"]
        valid_rows = [r for r in case_rows if bool(r["gt_has_foreground"])]
        scores = [float(r["main_iou"]) for r in valid_rows]

        self.assertEqual(len(valid_rows), 0)
        self.assertEqual(len(scores), 0)
        self.assertFalse(bool(scores))

    def test_case_selection_uses_iou_ranking(self):
        """Cases are selected by ascending mean_iou."""
        rows = [
            self._make_slice_row("s01", "cLow", 0.2),
            self._make_slice_row("s02", "cLow", 0.3),
            self._make_slice_row("s03", "cMid", 0.5),
            self._make_slice_row("s04", "cMid", 0.5),
            self._make_slice_row("s05", "cHigh", 0.9),
            self._make_slice_row("s06", "cHigh", 0.9),
        ]

        from collections import defaultdict
        case_to_rows: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            case_to_rows[str(row["case_id"])].append(row)

        case_summaries = []
        for case_id, case_rows in case_to_rows.items():
            valid = [r for r in case_rows if bool(r["gt_has_foreground"])]
            scores = [float(r["main_iou"]) for r in valid]
            case_summaries.append({
                "case_id": case_id,
                "mean_iou": float(np.mean(scores)),
                "valid": bool(scores),
            })

        valid_cases = [c for c in case_summaries if c["valid"]]
        valid_cases.sort(key=lambda c: (c["mean_iou"], c["case_id"]))

        select_count = 2
        selected = {c["case_id"] for c in valid_cases[:select_count]}

        # Lowest mean_iou cases selected
        self.assertIn("cLow", selected)
        self.assertIn("cMid", selected)
        self.assertNotIn("cHigh", selected)

    def test_worst20_mean_in_case_context(self):
        """worst20_mean correctly summarizes a case with variable slice IoUs."""
        # Simulate a case with 10 slices, some very poor
        ious = [0.95] * 8 + [0.1, 0.05]
        w20 = t05.worst20_mean(ious)
        # 20% of 10 = 2 worst: 0.05, 0.1 -> mean = 0.075
        self.assertAlmostEqual(w20, 0.075)


# ---------------------------------------------------------------------------
# Test G: load_mask
# ---------------------------------------------------------------------------

class TestLoadMask(unittest.TestCase):
    """G. load_mask handles various dimensionalities."""

    def _temp_npy(self, array: np.ndarray) -> Path:
        """Write array to a temporary .npy file and return its Path."""
        # Use a temporary file but keep it alive until test teardown
        tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        np.save(tmp.name, array)
        tmp.close()
        self._temp_files.append(Path(tmp.name))
        return Path(tmp.name)

    def setUp(self):
        self._temp_files: list[Path] = []

    def tearDown(self):
        for p in self._temp_files:
            if p.exists():
                p.unlink()

    def test_2d_mask_returns_2d_uint8(self):
        """2D mask returns 2D uint8 array."""
        arr = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        path = self._temp_npy(arr)

        result = t05.load_mask(path)
        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.dtype, np.uint8)
        np.testing.assert_array_equal(result, arr)

    def test_3d_h_w_1_squeezed_to_h_w(self):
        """(H, W, 1) mask squeezed to (H, W)."""
        arr = np.array([
            [[0], [1]],
            [[1], [0]],
        ], dtype=np.uint8)

        path = self._temp_npy(arr)
        result = t05.load_mask(path)

        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result.dtype, np.uint8)

        expected = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        np.testing.assert_array_equal(result, expected)

    def test_3d_1_h_w_squeezed_to_h_w(self):
        """(1, H, W) mask squeezed to (H, W)."""
        arr = np.array([
            [[0, 1, 0], [1, 0, 1]],
        ], dtype=np.uint8)  # shape (1, 2, 3)

        path = self._temp_npy(arr)
        result = t05.load_mask(path)

        self.assertEqual(result.ndim, 2)
        self.assertEqual(result.shape, (2, 3))
        self.assertEqual(result.dtype, np.uint8)

    def test_3d_multi_channel_raises(self):
        """3D mask with (H, W, 3) raises ValueError."""
        arr = np.zeros((2, 2, 3), dtype=np.uint8)

        path = self._temp_npy(arr)
        with self.assertRaises(ValueError) as ctx:
            t05.load_mask(path)
        self.assertIn("Unexpected mask shape", str(ctx.exception))

    def test_non_2d_after_squeeze_raises(self):
        """Mask that remains 3D after squeeze raises ValueError."""
        arr = np.zeros((2, 3, 4), dtype=np.uint8)

        path = self._temp_npy(arr)
        with self.assertRaises(ValueError) as ctx:
            t05.load_mask(path)
        self.assertIn("Unexpected mask shape", str(ctx.exception))

    def test_file_not_found(self):
        """Non-existent path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            t05.load_mask(Path("/nonexistent/path.npy"))


# ---------------------------------------------------------------------------
# Test H: CLI parser
# ---------------------------------------------------------------------------

class TestCLIParser(unittest.TestCase):
    """H. CLI parser accepts expected arguments."""

    def test_round_tag_and_round_id_accepted(self):
        """--round_tag and --round_id are accepted."""
        parser = t05.build_parser()
        args = parser.parse_args([
            "--processed_root", "/tmp/processed",
            "--datasets", "demo",
            "--round_tag", "r01_full5",
            "--round_id", "1",
            "--select_count", "3",
        ])
        self.assertEqual(args.round_tag, "r01_full5")
        self.assertEqual(args.round_id, 1)

    def test_select_count_accepted(self):
        """--select_count is accepted."""
        parser = t05.build_parser()
        args = parser.parse_args([
            "--processed_root", "/tmp/processed",
            "--datasets", "demo",
            "--round_tag", "r00_case1",
            "--round_id", "0",
            "--select_count", "5",
        ])
        self.assertEqual(args.select_count, 5)

    def test_overwrite_flag(self):
        """--overwrite flag is accepted."""
        parser = t05.build_parser()
        args = parser.parse_args([
            "--processed_root", "/tmp/processed",
            "--datasets", "demo",
            "--round_tag", "r00_full5",
            "--round_id", "0",
            "--select_count", "2",
            "--overwrite",
        ])
        self.assertTrue(args.overwrite)

    def test_overwrite_default_false(self):
        """--overwrite defaults to False when not given."""
        parser = t05.build_parser()
        args = parser.parse_args([
            "--processed_root", "/tmp/processed",
            "--datasets", "demo",
            "--round_tag", "r00_full5",
            "--round_id", "0",
            "--select_count", "2",
        ])
        self.assertFalse(args.overwrite)

    def test_method_default(self):
        """--method has a default value."""
        parser = t05.build_parser()
        args = parser.parse_args([
            "--processed_root", "/tmp/processed",
            "--datasets", "demo",
            "--round_tag", "r00_full5",
            "--round_id", "0",
            "--select_count", "2",
        ])
        self.assertIsNotNone(args.method)

    def test_process_datasets_comma_separated(self):
        """--datasets handles comma-separated names."""
        parser = t05.build_parser()
        args = parser.parse_args([
            "--processed_root", "/tmp/processed",
            "--datasets", "ds1,ds2,ds3",
            "--round_tag", "r00_full5",
            "--round_id", "0",
            "--select_count", "4",
        ])
        datasets = [n.strip() for n in args.datasets.split(",") if n.strip()]
        self.assertEqual(datasets, ["ds1", "ds2", "ds3"])

    def test_fold_default(self):
        """--fold defaults to 'fold_0'."""
        parser = t05.build_parser()
        args = parser.parse_args([
            "--processed_root", "/tmp/processed",
            "--datasets", "demo",
            "--round_tag", "r00_full5",
            "--round_id", "0",
            "--select_count", "2",
        ])
        self.assertEqual(args.fold, "fold_0")


# ---------------------------------------------------------------------------
# Test I: No test leakage
# ---------------------------------------------------------------------------

class TestNoLeakage(unittest.TestCase):
    """I. parse_class_ids excludes UNKNOWN_LABEL; get_case_id required=True."""

    def test_parse_class_ids_excludes_255(self):
        """parse_class_ids excludes UNKNOWN_LABEL (255)."""
        label_meta = {"unique_labels": [0, 1, 2, 255]}
        class_ids = t05.parse_class_ids(label_meta)
        self.assertNotIn(255, class_ids)
        self.assertIn(1, class_ids)
        self.assertIn(2, class_ids)
        self.assertNotIn(0, class_ids)  # 0 excluded by 0 < class_id

    def test_parse_class_ids_from_labels_dict(self):
        """parse_class_ids reads from 'labels' dict keys."""
        label_meta = {"labels": {"1": "liver", "2": "kidney", "255": "unknown"}}
        class_ids = t05.parse_class_ids(label_meta)
        self.assertNotIn(255, class_ids)
        self.assertEqual(set(class_ids), {1, 2})

    def test_parse_class_ids_from_class_ids_key(self):
        """parse_class_ids reads from 'class_ids' key."""
        label_meta = {"class_ids": [1, 2, 3]}
        class_ids = t05.parse_class_ids(label_meta)
        self.assertEqual(set(class_ids), {1, 2, 3})

    def test_parse_class_ids_uses_num_classes_fallback(self):
        """When no explicit class_ids, uses num_classes."""
        label_meta = {"num_classes": 4}
        class_ids = t05.parse_class_ids(label_meta)
        # range(1, max(4, 2)) = [1, 2, 3]
        self.assertEqual(class_ids, [1, 2, 3])

    def test_parse_class_ids_non_dict_returns_1(self):
        """Non-dict label_meta returns [1]."""
        class_ids = t05.parse_class_ids("not_a_dict")
        self.assertEqual(class_ids, [1])

    def test_parse_class_ids_handles_string_values(self):
        """String class IDs are converted to int."""
        label_meta = {"unique_labels": ["1", "3"]}
        class_ids = t05.parse_class_ids(label_meta)
        self.assertEqual(class_ids, [1, 3])

    def test_get_case_id_required_raises_for_missing(self):
        """get_case_id raises KeyError when required=True and case_id missing."""
        item_empty = {"slice_name": "test.npy", "case_id": None}
        with self.assertRaises(KeyError):
            t05.get_case_id(item_empty, required=True)

        item_blank = {"slice_name": "test.npy", "case_id": ""}
        with self.assertRaises(KeyError):
            t05.get_case_id(item_blank, required=True)

        item_missing = {"slice_name": "test.npy"}
        with self.assertRaises(KeyError):
            t05.get_case_id(item_missing, required=True)

    def test_get_case_id_not_required_returns_unknown(self):
        """get_case_id with required=False returns 'unknown_case' for missing."""
        item = {"slice_name": "test.npy"}
        result = t05.get_case_id(item, required=False)
        self.assertEqual(result, "unknown_case")

    def test_get_case_id_returns_string(self):
        """get_case_id returns string for valid case_id."""
        item = {"case_id": "case_001"}
        result = t05.get_case_id(item)
        self.assertEqual(result, "case_001")

    def test_unknown_label_constant_correct(self):
        """UNKNOWN_LABEL is exactly 255."""
        self.assertEqual(t05.UNKNOWN_LABEL, 255)


# ---------------------------------------------------------------------------
# Additional utility tests
# ---------------------------------------------------------------------------

class TestGetSliceName(unittest.TestCase):
    """get_slice_name extracts names from item dict."""

    def test_direct_slice_name(self):
        item = {"slice_name": "img001.npy"}
        self.assertEqual(t05.get_slice_name(item), "img001.npy")

    def test_from_teacher_img(self):
        item = {"teacher_img": "path/to/image.npy"}
        self.assertEqual(t05.get_slice_name(item), "image.npy")

    def test_from_student_img(self):
        item = {"student_img": "/a/b/slice.png"}
        self.assertEqual(t05.get_slice_name(item), "slice.png")

    def test_from_teacher_gt(self):
        item = {"teacher_gt": "gt/gt003.nii.gz"}
        self.assertEqual(t05.get_slice_name(item), "gt003.nii.gz")

    def test_from_student_gt(self):
        item = {"student_gt": "data/mask.npy"}
        self.assertEqual(t05.get_slice_name(item), "mask.npy")

    def test_missing_raises(self):
        item = {"foo": "bar"}
        with self.assertRaises(KeyError):
            t05.get_slice_name(item)


class TestResolvePath(unittest.TestCase):
    """resolve_path resolves relative paths against fold_root."""

    def setUp(self):
        self.fold_root = Path(tempfile.mkdtemp())
        self._tmp_dirs: list[Path] = [self.fold_root]

    def tearDown(self):
        import shutil
        for d in self._tmp_dirs:
            if d.exists():
                shutil.rmtree(d, ignore_errors=True)

    def test_relative_path_resolved(self):
        item = {"teacher_gt": "gt/mask.npy"}
        (self.fold_root / "gt").mkdir(parents=True, exist_ok=True)
        (self.fold_root / "gt" / "mask.npy").touch()

        result = t05.resolve_path(self.fold_root, item, "teacher_gt")
        self.assertEqual(result, self.fold_root / "gt" / "mask.npy")

    def test_absolute_path_preserved(self):
        abs_path = Path(tempfile.mkdtemp()) / "abs_mask.npy"
        abs_path.touch()
        self._tmp_dirs.append(abs_path.parent)
        item = {"teacher_gt": str(abs_path)}

        result = t05.resolve_path(self.fold_root, item, "teacher_gt")
        self.assertEqual(result, abs_path)

    def test_missing_key_raises(self):
        item = {"other_key": "val"}
        with self.assertRaises(KeyError):
            t05.resolve_path(self.fold_root, item, "teacher_gt")

    def test_file_not_found_raises(self):
        item = {"teacher_gt": "nonexistent.npy"}
        with self.assertRaises(FileNotFoundError):
            t05.resolve_path(self.fold_root, item, "teacher_gt")


class TestBuildUniqueIndex(unittest.TestCase):
    """build_unique_index deduplicates records by slice_name."""

    def test_builds_index(self):
        records = [
            {"slice_name": "a.npy", "val": 1},
            {"slice_name": "b.npy", "val": 2},
        ]
        index = t05.build_unique_index(records, "test_source")
        self.assertEqual(len(index), 2)
        self.assertEqual(index["a.npy"]["val"], 1)
        self.assertEqual(index["b.npy"]["val"], 2)

    def test_duplicate_raises(self):
        records = [
            {"slice_name": "dup.npy"},
            {"slice_name": "dup.npy"},
        ]
        with self.assertRaises(RuntimeError):
            t05.build_unique_index(records, "test_source")

    def test_non_dict_raises(self):
        records: list = ["not_a_dict"]  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            t05.build_unique_index(records, "test_source")


class TestCsvFloat(unittest.TestCase):
    """csv_float returns empty string for NaN."""

    def test_normal_float(self):
        self.assertEqual(t05.csv_float(0.5), 0.5)

    def test_nan_returns_empty_string(self):
        self.assertEqual(t05.csv_float(float("nan")), "")


class TestCheckOutputPaths(unittest.TestCase):
    """check_output_paths validates output paths."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil
        if self.tmp.exists():
            shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_existing_no_error(self):
        paths = [self.tmp / "a.json", self.tmp / "b.json"]
        # Should not raise
        t05.check_output_paths(paths, overwrite=False)

    def test_existing_overwrite_false_raises(self):
        f1 = self.tmp / "exists.json"
        f1.write_text("content")
        with self.assertRaises(FileExistsError):
            t05.check_output_paths([f1], overwrite=False)

    def test_existing_overwrite_true_no_error(self):
        f1 = self.tmp / "exists2.json"
        f1.write_text("content")
        # Should not raise
        t05.check_output_paths([f1], overwrite=True)


class TestValidateArgs(unittest.TestCase):
    """validate_args checks constraints."""

    def test_negative_round_id_raises(self):
        args = argparse.Namespace(
            round_id=-1,
            select_count=3,
        )
        with self.assertRaises(ValueError):
            t05.validate_args(args)

    def test_zero_round_id_ok(self):
        args = argparse.Namespace(
            round_id=0,
            select_count=5,
        )
        # Should not raise
        t05.validate_args(args)

    def test_zero_select_count_raises(self):
        args = argparse.Namespace(
            round_id=1,
            select_count=0,
        )
        with self.assertRaises(ValueError):
            t05.validate_args(args)

    def test_negative_select_count_raises(self):
        args = argparse.Namespace(
            round_id=1,
            select_count=-5,
        )
        with self.assertRaises(ValueError):
            t05.validate_args(args)

    def test_positive_select_count_ok(self):
        args = argparse.Namespace(
            round_id=1,
            select_count=10,
        )
        # Should not raise
        t05.validate_args(args)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
