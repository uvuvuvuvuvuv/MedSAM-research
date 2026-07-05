"""Unit tests for instance_utils module.

Covers bbox_iou, bbox_from_binary, load_gt_array, match_instance_mask,
and load_instance_target.  Uses only standard-library unittest and numpy/torch
(already project dependencies).
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Repo-root import
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from idea1_iterclean_bank_adaptshape import instance_utils as iu


# ====================================================================
# A. bbox_iou
# ====================================================================

class TestBboxIou(unittest.TestCase):
    """Tests for bbox_iou."""

    def test_identical_boxes(self) -> None:
        """Two identical boxes yield IoU = 1.0."""
        box: list[float] = [0.0, 0.0, 10.0, 10.0]
        result = iu.bbox_iou(box, box)
        self.assertAlmostEqual(
            result, 1.0,
            msg="Identical boxes must give IoU = 1.0",
        )

    def test_non_overlapping_boxes(self) -> None:
        """Two boxes that do not overlap yield IoU = 0.0."""
        a: list[float] = [0.0, 0.0, 5.0, 5.0]
        b: list[float] = [10.0, 10.0, 15.0, 15.0]
        result = iu.bbox_iou(a, b)
        self.assertEqual(
            result, 0.0,
            msg="Non-overlapping boxes must give IoU = 0.0",
        )

    def test_partial_overlap(self) -> None:
        """Partially overlapping 10x10 boxes shifted by 5 in x."""
        # a: [0,0,10,10]  area = 100
        # b: [5,0,15,10]  area = 100
        # inter: (5,0)-(10,10) -> 5*10 = 50
        # union: 100 + 100 - 50 = 150
        # IoU = 50 / 150 = 1/3
        a: list[float] = [0.0, 0.0, 10.0, 10.0]
        b: list[float] = [5.0, 0.0, 15.0, 10.0]
        expected = 50.0 / 150.0
        result = iu.bbox_iou(a, b)
        self.assertAlmostEqual(result, expected, msg="Partial overlap IoU mismatch")

    def test_containment(self) -> None:
        """A small box fully inside a large box yields small IoU."""
        outer: list[float] = [0.0, 0.0, 20.0, 20.0]  # area = 400
        inner: list[float] = [5.0, 5.0, 10.0, 10.0]  # area = 25
        # inter = 25, union = 400, IoU = 25 / 400 = 0.0625
        expected = 25.0 / 400.0
        result = iu.bbox_iou(outer, inner)
        self.assertAlmostEqual(result, expected, msg="Containment IoU mismatch")

    def test_zero_area_box(self) -> None:
        """A zero-area box (width=0, height=0) yields IoU = 0.0."""
        zero: list[float] = [5.0, 5.0, 5.0, 5.0]
        normal: list[float] = [0.0, 0.0, 10.0, 10.0]
        result = iu.bbox_iou(zero, normal)
        self.assertEqual(
            result, 0.0,
            msg="Zero-area box must give IoU = 0.0",
        )


# ====================================================================
# B. bbox_from_binary
# ====================================================================

class TestBboxFromBinary(unittest.TestCase):
    """Tests for bbox_from_binary."""

    def test_simple_rectangle(self) -> None:
        """A rectangular True region returns (x1, y1, x2, y2)."""
        mask = np.zeros((50, 60), dtype=bool)
        mask[10:30, 15:40] = True
        result = iu.bbox_from_binary(mask)
        self.assertIsNotNone(result, msg="Non-empty mask must not return None")
        self.assertEqual(
            result, (15, 10, 40, 30),
            msg="Rectangle bbox should be (x1=15, y1=10, x2=40, y2=30)",
        )

    def test_empty_mask(self) -> None:
        """An all-False mask returns None."""
        mask = np.zeros((50, 50), dtype=bool)
        result = iu.bbox_from_binary(mask)
        self.assertIsNone(result, msg="Empty mask must return None")

    def test_single_pixel(self) -> None:
        """A single True pixel produces a 1x1 bounding box."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3, 7] = True
        result = iu.bbox_from_binary(mask)
        self.assertEqual(
            result, (7, 3, 8, 4),
            msg="Single-pixel bbox must be (x1=col, y1=row, x2=col+1, y2=row+1)",
        )

    def test_offset_mask(self) -> None:
        """A blob not touching an image border still gives correct coords."""
        mask = np.zeros((100, 100), dtype=bool)
        mask[50:60, 30:70] = True
        result = iu.bbox_from_binary(mask)
        self.assertEqual(
            result, (30, 50, 70, 60),
            msg="Offset mask bbox mismatch",
        )


# ====================================================================
# C. load_gt_array
# ====================================================================

class TestLoadGtArray(unittest.TestCase):
    """Tests for load_gt_array using temporary .npy files."""

    def setUp(self) -> None:
        self._tmpfiles: list[Path] = []

    def tearDown(self) -> None:
        for p in self._tmpfiles:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    def _save_temp(self, array: np.ndarray) -> Path:
        """Save an array to a temp .npy file and track for cleanup."""
        tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        tmp.close()
        path = Path(tmp.name)
        self._tmpfiles.append(path)
        np.save(str(path), array)
        return path

    def test_2d_int32_array_becomes_int64(self) -> None:
        """A 2D int32 array loads as int64 with same values."""
        arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
        path = self._save_temp(arr)
        result = iu.load_gt_array(path)
        self.assertEqual(result.shape, (2, 2), msg="Shape mismatch for 2D array")
        self.assertEqual(result.dtype, np.int64, msg="Output dtype must be int64")
        np.testing.assert_array_equal(
            result, arr.astype(np.int64),
            err_msg="Array values changed during load",
        )

    def test_3d_channel_first_squeezed(self) -> None:
        """(1, H, W) array is squeezed to (H, W)."""
        arr = np.array([[[1, 2], [3, 4]]], dtype=np.int32)  # shape (1, 2, 2)
        path = self._save_temp(arr)
        result = iu.load_gt_array(path)
        self.assertEqual(
            result.shape, (2, 2),
            msg="(1, H, W) shape must be squeezed to (H, W)",
        )

    def test_3d_channel_last_squeezed(self) -> None:
        """(H, W, 1) array is squeezed to (H, W)."""
        arr = np.array([[[1], [2]], [[3], [4]]], dtype=np.int32)  # shape (2, 2, 1)
        path = self._save_temp(arr)
        result = iu.load_gt_array(path)
        self.assertEqual(
            result.shape, (2, 2),
            msg="(H, W, 1) shape must be squeezed to (H, W)",
        )


# ====================================================================
# D. match_instance_mask
# ====================================================================

class TestMatchInstanceMask(unittest.TestCase):
    """Tests for match_instance_mask with synthetic multi-component GT."""

    @staticmethod
    def _make_two_component_gt(
        size: int = 100,
        label_id: int = 1,
    ) -> np.ndarray:
        """Return a GT array with two disconnected squares of *label_id*.

        Component 1 (first in OpenCV scan order): rows 10-20, cols 10-20.
        Component 2: rows 50-65, cols 50-70.

        Both share the same *label_id* so connectedComponentsWithStats
        separates them.
        """
        gt = np.zeros((size, size), dtype=np.int64)
        gt[10:20, 10:20] = label_id   # comp #1 (smaller, top-left)
        gt[50:65, 50:70] = label_id   # comp #2 (larger, lower-right)
        return gt

    # ---- component_id explicit -----------------------------------------------

    def test_component_id_1_selects_first(self) -> None:
        """component_id=1 returns only the first connected component."""
        gt = self._make_two_component_gt(label_id=1)
        mask = iu.match_instance_mask(
            gt, label_id=1,
            bbox=[0.0, 0.0, 1.0, 1.0],  # ignored when component_id given
            component_id=1,
        )
        self.assertTrue(mask[15, 15], "Pixel inside component 1 must be True")
        self.assertFalse(mask[55, 55], "Pixel inside component 2 must be False")
        self.assertTrue(mask.any(), "Result mask must not be empty")

    def test_component_id_2_selects_second(self) -> None:
        """component_id=2 returns only the second connected component."""
        gt = self._make_two_component_gt(label_id=1)
        mask = iu.match_instance_mask(
            gt, label_id=1,
            bbox=[0.0, 0.0, 1.0, 1.0],
            component_id=2,
        )
        self.assertTrue(mask[55, 55], "Pixel inside component 2 must be True")
        self.assertFalse(mask[15, 15], "Pixel inside component 1 must be False")

    def test_component_id_zero_raises_valueerror(self) -> None:
        """component_id=0 (non-positive) raises ValueError."""
        gt = self._make_two_component_gt(label_id=1)
        with self.assertRaises(ValueError):
            iu.match_instance_mask(
                gt, label_id=1, bbox=[0.0, 0.0, 1.0, 1.0], component_id=0,
            )

    def test_component_id_out_of_range_raises(self) -> None:
        """component_id > num_labels-1 raises ValueError."""
        gt = self._make_two_component_gt(label_id=1)
        with self.assertRaises(ValueError):
            iu.match_instance_mask(
                gt, label_id=1, bbox=[0.0, 0.0, 1.0, 1.0], component_id=999,
            )

    # ---- component_id=None (bbox IoU matching) -------------------------------

    def test_none_component_id_selects_by_iou(self) -> None:
        """component_id=None picks the component with best bbox IoU."""
        gt = self._make_two_component_gt(label_id=1)
        # bbox that perfectly matches component 2: rows 50-65, cols 50-70
        bbox: list[float] = [50.0, 50.0, 70.0, 65.0]
        mask = iu.match_instance_mask(
            gt, label_id=1, bbox=bbox, component_id=None,
        )
        self.assertTrue(mask[55, 55], "Component 2 pixel must be True")
        self.assertFalse(mask[15, 15], "Component 1 pixel must be False")

    def test_none_component_id_no_overlap_raises(self) -> None:
        """component_id=None with a bbox overlapping nothing raises ValueError."""
        gt = self._make_two_component_gt(label_id=1)
        bbox: list[float] = [200.0, 200.0, 210.0, 210.0]
        with self.assertRaises(ValueError):
            iu.match_instance_mask(
                gt, label_id=1, bbox=bbox, component_id=None,
            )

    # ---- label_id errors -----------------------------------------------------

    def test_label_id_not_present_raises(self) -> None:
        """Requesting a label that does not exist in GT raises ValueError."""
        gt = self._make_two_component_gt(label_id=1)
        with self.assertRaises(ValueError):
            iu.match_instance_mask(
                gt, label_id=42, bbox=[0.0, 0.0, 1.0, 1.0], component_id=1,
            )


# ====================================================================
# E. load_instance_target
# ====================================================================

class TestLoadInstanceTarget(unittest.TestCase):
    """Tests for load_instance_target (end-to-end: load + match)."""

    def setUp(self) -> None:
        self._tmpfiles: list[Path] = []

    def tearDown(self) -> None:
        for p in self._tmpfiles:
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass

    def _save_temp_gt(self, array: np.ndarray) -> Path:
        tmp = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
        tmp.close()
        path = Path(tmp.name)
        self._tmpfiles.append(path)
        np.save(str(path), array)
        return path

    @staticmethod
    def _make_simple_gt(label_id: int = 1) -> np.ndarray:
        """Return a 50x50 GT with one square component at (10:20, 10:20)."""
        gt = np.zeros((50, 50), dtype=np.int64)
        gt[10:20, 10:20] = label_id
        return gt

    def test_output_shape(self) -> None:
        """Returns a tensor of shape (1, 1, H, W)."""
        gt = self._make_simple_gt(label_id=1)
        path = self._save_temp_gt(gt)
        device = torch.device("cpu")
        bbox: list[float] = [10.0, 10.0, 20.0, 20.0]
        tensor = iu.load_instance_target(
            path, device, label_id=1, bbox=bbox, component_id=1,
        )
        self.assertEqual(
            tensor.shape, (1, 1, 50, 50),
            msg="Output must be shape (1, 1, H, W) = (1, 1, 50, 50)",
        )

    def test_values_are_float32_and_binary(self) -> None:
        """All tensor values are 0.0 or 1.0 in float32 dtype."""
        gt = self._make_simple_gt(label_id=1)
        path = self._save_temp_gt(gt)
        device = torch.device("cpu")
        bbox: list[float] = [10.0, 10.0, 20.0, 20.0]
        tensor = iu.load_instance_target(
            path, device, label_id=1, bbox=bbox, component_id=1,
        )
        self.assertEqual(
            tensor.dtype, torch.float32,
            msg="Output dtype must be float32",
        )
        unique = tensor.unique()
        all_binary = torch.all((unique == 0.0) | (unique == 1.0))
        self.assertTrue(
            all_binary.item(),
            msg="Output values must be exclusively 0.0 and 1.0",
        )

    def test_matched_component_pixels_are_one(self) -> None:
        """Pixels inside the matched component are 1.0; background is 0.0."""
        gt = self._make_simple_gt(label_id=1)
        path = self._save_temp_gt(gt)
        device = torch.device("cpu")
        bbox: list[float] = [10.0, 10.0, 20.0, 20.0]
        tensor = iu.load_instance_target(
            path, device, label_id=1, bbox=bbox, component_id=1,
        )
        arr = tensor.squeeze().numpy()
        self.assertEqual(arr[15, 15], 1.0, msg="Matched component pixel must be 1.0")
        self.assertEqual(arr[0, 0], 0.0, msg="Background pixel must be 0.0")


# ====================================================================

if __name__ == "__main__":
    unittest.main()
