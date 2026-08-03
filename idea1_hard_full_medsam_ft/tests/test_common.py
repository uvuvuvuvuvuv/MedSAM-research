from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from idea1_common import (
    binary_iou,
    compute_3d_budget,
    map_native_mask_to_target,
    map_teacher_mask_to_native,
    seeded_sample,
    validate_partition,
)


class CommonTests(unittest.TestCase):
    def test_budget(self):
        self.assertEqual(compute_3d_budget(18), 1)
        self.assertEqual(compute_3d_budget(30), 2)
        self.assertEqual(compute_3d_budget(60), 3)
        self.assertEqual(compute_3d_budget(80), 4)
        self.assertEqual(compute_3d_budget(100), 5)
        self.assertEqual(compute_3d_budget(1000), 5)

    def test_seeded_sample_reproducible(self):
        values = [f"s{i}" for i in range(20)]
        self.assertEqual(seeded_sample(values, 5, 2026), seeded_sample(values, 5, 2026))
        self.assertNotEqual(seeded_sample(values, 5, 2026), seeded_sample(values, 5, 2027))

    def test_partition(self):
        validate_partition({"a", "b", "c"}, {"a"}, {"b", "c"})
        with self.assertRaises(Exception):
            validate_partition({"a", "b"}, {"a"}, {"a", "b"})

    def test_geometry_roundtrip_no_padding(self):
        mask = np.zeros((10, 20), dtype=np.uint8)
        mask[2:8, 5:15] = 1
        geom = {
            "native_to_teacher": {
                "orig_h": 10,
                "orig_w": 20,
                "target_h": 20,
                "target_w": 40,
                "scale_x": 2.0,
                "scale_y": 2.0,
                "offset_x": 0,
                "offset_y": 0,
                "resized_h": 20,
                "resized_w": 40,
            }
        }
        teacher = map_native_mask_to_target(mask, geom, "native_to_teacher")
        restored = map_teacher_mask_to_native(teacher, geom)
        self.assertEqual(binary_iou(restored, mask), 1.0)


if __name__ == "__main__":
    unittest.main()
