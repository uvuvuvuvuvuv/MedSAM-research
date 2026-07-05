"""Unit tests for 02_build_support_template module.

A. Coverage calculation (via resize_float)
B. FG/BG screening
C. Feature bank L2 normalization
D. Shape clustering (spherical_kmeans, _adaptive_shape_clustering)
E. NPZ contract (synthetic NPZ with np.savez_compressed)
F. CLI parser (build_parser, validate_args)
G. Shape crop and resize
H. Ring background (_compute_ring_mask)
I. 3D per-case balancing (sample_rows + capping logic)
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

t02 = importlib.import_module(
    "idea1_iterclean_bank_adaptshape.02_build_support_template"
)


# ============================================================================
# Helpers
# ============================================================================


def _bbox_from_binary(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-6) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


# ============================================================================
# A: resize_float + coverage calculation
# ============================================================================


class TestResizeFloat(unittest.TestCase):
    def test_all_ones_stays_one(self):
        result = t02.resize_float(np.ones((100, 100), dtype=np.float32), (20, 20))
        self.assertEqual(result.shape, (20, 20))
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_all_zeros_stays_zero(self):
        result = t02.resize_float(np.zeros((100, 100), dtype=np.float32), (20, 20))
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_square_to_smaller(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1.0
        result = t02.resize_float(mask, (10, 10))
        self.assertGreaterEqual(float(result[5, 5]), 0.9)
        for r, c in [(0, 0), (0, 9), (9, 0), (9, 9)]:
            self.assertLessEqual(float(result[r, c]), 0.1)

    def test_rectangular_input(self):
        result = t02.resize_float(np.ones((80, 120), dtype=np.float32), (40, 60))
        self.assertEqual(result.shape, (40, 60))
        np.testing.assert_allclose(result, 1.0, atol=1e-5)


class TestCoverageCalculation(unittest.TestCase):
    """A. Coverage semantics of bilinearly-resized binary mask."""

    def test_fg_token_in_box_high_coverage(self):
        h, w = 200, 200
        mask = np.zeros((h, w), dtype=np.float32)
        mask[50:150, 50:150] = 1.0
        coverage = t02.resize_float(mask, (20, 20))
        self.assertGreaterEqual(float(coverage[10, 10]), 0.95)

    def test_token_outside_box_zero_coverage(self):
        h, w = 200, 200
        mask = np.zeros((h, w), dtype=np.float32)
        mask[50:150, 50:150] = 1.0
        coverage = t02.resize_float(mask, (20, 20))
        for r, c in [(0, 0), (0, 19), (19, 0), (19, 19)]:
            self.assertLessEqual(float(coverage[r, c]), 0.05)

    def test_bg_token_in_ring_band_low_coverage(self):
        h, w = 200, 200
        mask = np.zeros((h, w), dtype=np.float32)
        mask[50:150, 50:150] = 1.0
        coverage = t02.resize_float(mask, (20, 20))
        bbox_teacher = [50.0, 50.0, 150.0, 150.0]
        ring = t02._compute_ring_mask(20, 20, bbox_teacher, h, w, ring_width=5)
        rr, cc = np.nonzero(ring)
        if len(rr) > 0:
            self.assertLess(float(coverage[rr, cc].max()), 0.15)

    def test_coverage_values_in_range(self):
        rng = np.random.default_rng(42)
        mask = rng.integers(0, 2, size=(128, 128)).astype(np.float32)
        coverage = t02.resize_float(mask, (32, 32))
        self.assertTrue(np.all(coverage >= 0.0))
        self.assertTrue(np.all(coverage <= 1.0))


# ============================================================================
# B: FG/BG screening
# ============================================================================


class TestFgBgScreening(unittest.TestCase):
    """B. Token classification from coverage thresholds."""

    def test_high_coverage_foreground(self):
        coverage = np.array([[0.95, 0.91, 0.50],
                             [0.99, 0.88, 0.02],
                             [0.15, 0.45, 0.03]], dtype=np.float32)
        fg = coverage >= 0.90
        expected = np.array([[True, True, False],
                             [True, False, False],
                             [False, False, False]])
        np.testing.assert_array_equal(fg, expected)

    def test_low_coverage_background(self):
        coverage = np.array([[0.95, 0.91, 0.50],
                             [0.99, 0.88, 0.02],
                             [0.15, 0.45, 0.03]], dtype=np.float32)
        bg = coverage <= 0.10
        expected = np.array([[False, False, False],
                             [False, False, True],
                             [False, False, True]])
        np.testing.assert_array_equal(bg, expected)

    def test_medium_coverage_depends_on_threshold(self):
        coverage = np.array([[0.50, 0.60]], dtype=np.float32)
        self.assertFalse((coverage >= 0.90).any())
        self.assertFalse((coverage <= 0.10).any())
        self.assertTrue((coverage >= 0.50).all())


# ============================================================================
# C: Feature L2 normalization
# ============================================================================


class TestFeatureL2Normalization(unittest.TestCase):
    """C. Per-token L2 normalization (as in extract_encoder_feature)."""

    def test_normalized_features_have_unit_norm(self):
        rng = np.random.default_rng(42)
        features = _l2_normalize(rng.standard_normal((100, 256), dtype=np.float32))
        np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-5)

    def test_zero_norm_features_handled(self):
        normalized = _l2_normalize(np.zeros((10, 64), dtype=np.float32))
        self.assertFalse(np.any(np.isnan(normalized)))
        self.assertFalse(np.any(np.isinf(normalized)))
        np.testing.assert_allclose(normalized, 0.0, atol=1e-7)

    def test_batch_hwc_normalization(self):
        rng = np.random.default_rng(42)
        features = _l2_normalize(
            rng.standard_normal((16, 16, 256), dtype=np.float32), axis=-1,
        )
        np.testing.assert_allclose(np.linalg.norm(features, axis=-1), 1.0, atol=1e-5)
        self.assertEqual(features.shape, (16, 16, 256))

    def test_single_vector(self):
        n = _l2_normalize(np.array([3.0, 4.0], dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(n)), 1.0, places=5)
        self.assertAlmostEqual(float(n[0]), 0.6, places=5)
        self.assertAlmostEqual(float(n[1]), 0.8, places=5)


# ============================================================================
# D: Shape clustering
# ============================================================================


class TestSphericalKmeans(unittest.TestCase):
    """D. spherical_kmeans core clustering."""

    def test_deterministic(self):
        rng = np.random.default_rng(123)
        f = _l2_normalize(rng.standard_normal((50, 64), dtype=np.float32))
        r1 = t02.spherical_kmeans(f, n_clusters=3, seed=42, num_iter=20)
        r2 = t02.spherical_kmeans(f, n_clusters=3, seed=42, num_iter=20)
        np.testing.assert_array_equal(r1, r2)

    def test_k1_degenerate(self):
        rng = np.random.default_rng(123)
        f = _l2_normalize(rng.standard_normal((30, 64), dtype=np.float32))
        centers = t02.spherical_kmeans(f, n_clusters=1, seed=42, num_iter=20)
        self.assertEqual(centers.shape, (1, 64))
        self.assertAlmostEqual(float(np.linalg.norm(centers[0])), 1.0, places=4)

    def test_k2_separates_two_groups(self):
        dim = 64
        rng = np.random.default_rng(99)
        a = (rng.standard_normal((25, dim)) * 0.08).astype(np.float32)
        a[:, 0] += 1.0
        b = (rng.standard_normal((25, dim)) * 0.08).astype(np.float32)
        b[:, 1] += 1.0
        f = _l2_normalize(np.concatenate([a, b], axis=0))
        centers = t02.spherical_kmeans(f, n_clusters=2, seed=42, num_iter=30)
        self.assertEqual(centers.shape, (2, dim))
        labels = (f @ centers.T).argmax(axis=1)
        ma = int(np.bincount(labels[:25]).argmax())
        mb = int(np.bincount(labels[25:]).argmax())
        self.assertNotEqual(ma, mb)
        self.assertGreaterEqual((labels[:25] == ma).mean(), 0.75)
        self.assertGreaterEqual((labels[25:] == mb).mean(), 0.75)

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            t02.spherical_kmeans(np.empty((0, 64), dtype=np.float32),
                                 n_clusters=3, seed=42, num_iter=10)

    def test_k_larger_than_n_capped(self):
        f = _l2_normalize(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        centers = t02.spherical_kmeans(f, n_clusters=5, seed=42, num_iter=10)
        self.assertLessEqual(centers.shape[0], 2)

    def test_centroids_are_unit_norm(self):
        rng = np.random.default_rng(99)
        f = _l2_normalize(rng.standard_normal((60, 32), dtype=np.float32))
        for k in [1, 2, 4]:
            centers = t02.spherical_kmeans(f, n_clusters=k, seed=42, num_iter=20)
            for c in range(centers.shape[0]):
                self.assertAlmostEqual(float(np.linalg.norm(centers[c])), 1.0, places=4)


class TestAdaptiveShapeClustering(unittest.TestCase):
    """D. _adaptive_shape_clustering returns (K, labels, centers)."""

    @staticmethod
    def _mk(rng, n, dim, d, noise=0.05):
        s = (rng.standard_normal((n, dim)) * noise).astype(np.float32)
        s[:, d] += 1.0
        return _l2_normalize(s)

    def test_insufficient_support_fallback_k1(self):
        dim = 64
        n = 4
        rng = np.random.default_rng(2026)
        s = np.concatenate([
            self._mk(rng, 2, dim, 0, 0.01),
            self._mk(rng, 2, dim, 5, 0.01),
        ], axis=0)
        u = np.array(["u0"] * n, dtype=str)
        K, labels, centers = t02._adaptive_shape_clustering(
            s, u, Kmax=3, min_support=2, seed=42, num_iter=20,
        )
        self.assertEqual(K, 1)
        np.testing.assert_array_equal(labels, np.zeros(n, dtype=np.int64))
        self.assertEqual(centers.shape, (1, dim))

    def test_sufficient_support_produces_clusters(self):
        dim = 64
        n = 5
        rng = np.random.default_rng(2026)
        s = np.concatenate([
            self._mk(rng, n, dim, 0, 0.01),
            self._mk(rng, n, dim, 10, 0.01),
        ], axis=0)
        u = np.array([f"a_{i}" for i in range(n)] + [f"b_{i}" for i in range(n)],
                     dtype=str)
        K, labels, centers = t02._adaptive_shape_clustering(
            s, u, Kmax=3, min_support=2, seed=42, num_iter=30,
        )
        self.assertGreaterEqual(K, 1)
        self.assertEqual(centers.shape[0], K)
        self.assertEqual(len(labels), 2 * n)
        for k in range(K):
            self.assertAlmostEqual(float(np.linalg.norm(centers[k])), 1.0, places=4)
        if K >= 2:
            ma = int(np.bincount(labels[:n]).argmax())
            mb = int(np.bincount(labels[n:]).argmax())
            self.assertNotEqual(ma, mb)

    def test_empty_shapes_raises(self):
        with self.assertRaises(ValueError):
            t02._adaptive_shape_clustering(
                np.empty((0, 64), dtype=np.float32),
                np.array([], dtype=str),
                Kmax=3, min_support=2, seed=42, num_iter=10,
            )


class TestBuildClusterTemplates(unittest.TestCase):
    """D. _build_cluster_templates -> list of (S,S) arrays."""

    def test_normal_clusters(self):
        rng = np.random.default_rng(42)
        S = 8
        shapes = rng.random((10, S * S)).astype(np.float32)
        labels = np.array([0] * 5 + [1] * 5, dtype=np.int64)
        tmpl = t02._build_cluster_templates(shapes, labels, 2, S)
        self.assertEqual(len(tmpl), 2)
        for t in tmpl:
            self.assertEqual(t.shape, (S, S))
            self.assertTrue(np.all((t >= 0.0) & (t <= 1.0)))

    def test_empty_cluster_zeros(self):
        rng = np.random.default_rng(42)
        S = 4
        shapes = rng.random((5, S * S)).astype(np.float32)
        tmpl = t02._build_cluster_templates(shapes, np.zeros(5, dtype=np.int64), 2, S)
        self.assertEqual(len(tmpl), 2)
        np.testing.assert_allclose(tmpl[1], 0.0, atol=1e-7)

    def test_single_cluster(self):
        rng = np.random.default_rng(42)
        S = 16
        shapes = rng.random((8, S * S)).astype(np.float32)
        tmpl = t02._build_cluster_templates(shapes, np.zeros(8, dtype=np.int64), 1, S)
        self.assertEqual(len(tmpl), 1)
        self.assertEqual(tmpl[0].shape, (S, S))


class TestComputeSemanticCenters(unittest.TestCase):
    """D. _compute_semantic_centers: per-cluster FG token average."""

    def test_basic(self):
        dim = 32
        rng = np.random.default_rng(42)
        fg = [
            rng.standard_normal((5, dim)).astype(np.float32) * 0.1
            + np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32),
            rng.standard_normal((5, dim)).astype(np.float32) * 0.1
            + np.array([1.0] + [0.0] * (dim - 1), dtype=np.float32),
            rng.standard_normal((5, dim)).astype(np.float32) * 0.1
            + np.array([0.0] + [1.0] + [0.0] * (dim - 2), dtype=np.float32),
            rng.standard_normal((5, dim)).astype(np.float32) * 0.1
            + np.array([0.0] + [1.0] + [0.0] * (dim - 2), dtype=np.float32),
        ]
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        centers = t02._compute_semantic_centers(fg, labels, 2, dim)
        self.assertEqual(centers.shape, (2, dim))
        for k in range(2):
            self.assertAlmostEqual(float(np.linalg.norm(centers[k])), 1.0, places=4)
        self.assertLess(float(np.dot(centers[0], centers[1])), 0.99)


# ============================================================================
# E: NPZ contract
# ============================================================================


class TestNpzContract(unittest.TestCase):
    """E. Synthetic NPZ schema validation."""

    REQUIRED_META = {"class_ids", "feature_dim", "shape_size",
                      "method", "round_tag", "run_id"}
    REQUIRED_CLASS_KINDS = ("bank_fg_c", "bank_bg_c",
                            "shape_templates_c", "shape_semantic_centers_c")
    DEPRECATED = ("proto_fg_c", "proto_bg_c", "shape_A_c", "shape_R_c")

    def _build(self, path):
        dim, S, cids = 128, 64, [1, 3]
        fg = np.random.default_rng(1).standard_normal((200, dim)).astype(np.float32)
        bg = np.random.default_rng(2).standard_normal((150, dim)).astype(np.float32)
        tmpl = np.random.default_rng(3).random((2, S, S)).astype(np.float32)
        centers = _l2_normalize(
            np.random.default_rng(4).standard_normal((2, dim)).astype(np.float32))

        d = {
            "class_ids": np.array(cids, dtype=np.int64),
            "feature_dim": np.array(dim, dtype=np.int64),
            "shape_size": np.array([S, S], dtype=np.int64),
            "method": np.array("test_method"),
            "round_tag": np.array("r00_full5"),
            "run_id": np.array("run_abc123"),
        }
        for c in cids:
            d[f"bank_fg_c{c}"] = fg
            d[f"bank_bg_c{c}"] = bg
            d[f"shape_templates_c{c}"] = tmpl
            d[f"shape_semantic_centers_c{c}"] = centers
        np.savez_compressed(path, **d)

    def test_allow_pickle_false(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.npz")
            self._build(p)
            loaded = np.load(p, allow_pickle=False)
            self.assertIsInstance(loaded, np.lib.npyio.NpzFile)
            loaded.close()

    def test_required_meta_keys(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.npz")
            self._build(p)
            loaded = np.load(p, allow_pickle=False)
            missing = self.REQUIRED_META - set(loaded.keys())
            self.assertFalse(missing, f"Missing: {missing}")
            loaded.close()

    def test_per_class_keys(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.npz")
            self._build(p)
            loaded = np.load(p, allow_pickle=False)
            keys = set(loaded.keys())
            for c in [1, 3]:
                for kind in self.REQUIRED_CLASS_KINDS:
                    self.assertIn(f"{kind}{c}", keys)
            loaded.close()

    def test_deprecated_keys_absent(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.npz")
            self._build(p)
            loaded = np.load(p, allow_pickle=False)
            keys = set(loaded.keys())
            for prefix in self.DEPRECATED:
                self.assertFalse([k for k in keys if k.startswith(prefix)])
            loaded.close()

    def test_shape_size_2d(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.npz")
            self._build(p)
            loaded = np.load(p, allow_pickle=False)
            ss = loaded["shape_size"]
            self.assertEqual(ss.shape, (2,))
            self.assertTrue(np.all(ss > 0))
            loaded.close()

    def test_class_ids_range(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.npz")
            self._build(p)
            loaded = np.load(p, allow_pickle=False)
            cids = loaded["class_ids"]
            self.assertTrue(np.all((cids > 0) & (cids < 255)))
            loaded.close()

    def test_feature_dim_scalar(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.npz")
            self._build(p)
            loaded = np.load(p, allow_pickle=False)
            fd = loaded["feature_dim"]
            self.assertEqual(fd.shape, ())
            self.assertGreater(int(fd), 0)
            loaded.close()

    def test_per_class_array_ndim(self):
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "s.npz")
            self._build(p)
            loaded = np.load(p, allow_pickle=False)
            for c in [1, 3]:
                self.assertEqual(loaded[f"bank_fg_c{c}"].ndim, 2)
                self.assertEqual(loaded[f"bank_fg_c{c}"].shape[1], 128)
                self.assertEqual(loaded[f"shape_templates_c{c}"].ndim, 3)
                self.assertEqual(loaded[f"shape_templates_c{c}"].shape[1], 64)
            loaded.close()


# ============================================================================
# F: CLI parser
# ============================================================================


class TestCliParser(unittest.TestCase):
    """F. build_parser."""

    _B = ["--processed_root", "/tmp/pr", "--checkpoint", "/tmp/c.pth",
          "--datasets", "demo", "--method", "idea1_iter_gt_iou_s2026_r0_full5"]

    def test_kmax_shape(self):
        ns = t02.build_parser().parse_args(self._B + ["--kmax_shape", "3"])
        self.assertEqual(ns.kmax_shape, 3)

    def test_shape_size(self):
        ns = t02.build_parser().parse_args(self._B + ["--shape_size", "128"])
        self.assertEqual(ns.shape_size, 128)

    def test_cluster_max_iter(self):
        ns = t02.build_parser().parse_args(self._B + ["--cluster_max_iter", "50"])
        self.assertEqual(ns.cluster_max_iter, 50)

    def test_ring_expand_ratio(self):
        ns = t02.build_parser().parse_args(self._B + ["--ring_expand_ratio", "8"])
        self.assertEqual(ns.ring_expand_ratio, 8)

    def test_old_k_fg_rejected(self):
        with self.assertRaises(SystemExit):
            t02.build_parser().parse_args(self._B + ["--k_fg", "5"])

    def test_old_k_bg_rejected(self):
        with self.assertRaises(SystemExit):
            t02.build_parser().parse_args(self._B + ["--k_bg", "3"])

    def test_defaults(self):
        ns = t02.build_parser().parse_args(self._B)
        self.assertEqual(ns.fold, "fold_0")
        self.assertEqual(ns.seed, 2026)
        self.assertEqual(ns.device, "cuda")
        self.assertEqual(ns.fg_coverage_threshold, 0.90)
        self.assertEqual(ns.bg_coverage_threshold, 0.10)
        self.assertEqual(ns.shape_size, 64)
        self.assertEqual(ns.kmax_shape, 5)
        self.assertEqual(ns.cluster_max_iter, 25)
        self.assertEqual(ns.ring_expand_ratio, 5)
        self.assertIsNone(ns.round_tag)
        self.assertFalse(ns.overwrite)

    def test_overwrite(self):
        ns = t02.build_parser().parse_args(self._B + ["--overwrite"])
        self.assertTrue(ns.overwrite)

    def test_round_tag(self):
        ns = t02.build_parser().parse_args(self._B + ["--round_tag", "r01_full10"])
        self.assertEqual(ns.round_tag, "r01_full10")


class TestValidateArgs(unittest.TestCase):
    """F. validate_args."""

    @staticmethod
    def _ns(**kw):
        d = dict(
            processed_root=Path("/t"), checkpoint=Path("/c"),
            datasets="d", method="m", fold="f0", round_tag=None,
            device="cpu", seed=2026, overwrite=False,
            fg_coverage_threshold=0.9, bg_coverage_threshold=0.1,
            max_fg_per_instance=128, max_bg_per_instance=128,
            max_fg_per_class=4096, max_bg_per_class=4096,
            ring_expand_ratio=5, shape_size=64, kmax_shape=5,
            cluster_max_iter=25,
        )
        d.update(kw)
        return argparse.Namespace(**d)

    def test_valid(self):
        t02.validate_args(self._ns())

    def test_negative_seed(self):
        with self.assertRaises(ValueError):
            t02.validate_args(self._ns(seed=-1))

    def test_fg_threshold_oob(self):
        for v in [1.5, -0.1]:
            with self.subTest(v=v):
                with self.assertRaises(ValueError):
                    t02.validate_args(self._ns(fg_coverage_threshold=v))

    def test_bg_threshold_oob(self):
        for v in [1.5, -0.1]:
            with self.subTest(v=v):
                with self.assertRaises(ValueError):
                    t02.validate_args(self._ns(bg_coverage_threshold=v))

    def test_zero_positive_params(self):
        for name in ("max_fg_per_instance", "max_bg_per_instance",
                     "max_fg_per_class", "max_bg_per_class",
                     "ring_expand_ratio", "shape_size",
                     "kmax_shape", "cluster_max_iter"):
            with self.subTest(p=name):
                with self.assertRaises(ValueError):
                    t02.validate_args(self._ns(**{name: 0}))

    def test_boundary_thresholds_valid(self):
        for thr, val in [("fg_coverage_threshold", 0.0),
                         ("fg_coverage_threshold", 1.0),
                         ("bg_coverage_threshold", 0.0),
                         ("bg_coverage_threshold", 1.0)]:
            with self.subTest(thr=thr, val=val):
                t02.validate_args(self._ns(**{thr: val}))


# ============================================================================
# G: Shape crop + resize
# ============================================================================


class TestShapeCropResize(unittest.TestCase):
    """G. Shape extraction pipeline."""

    def test_crop_correct_shape(self):
        mask = np.zeros((64, 64), dtype=np.int64)
        mask[20:44, 16:48] = 1  # 24x32
        bbox = _bbox_from_binary(mask)
        self.assertIsNotNone(bbox)
        crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(np.float32)
        self.assertEqual(crop.shape, (24, 32))
        self.assertTrue(np.all(crop == 1.0))

    def test_resize_to_shape_size(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[10:50, 10:50] = 1.0
        bbox = _bbox_from_binary(mask)
        crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(np.float32)
        for S in [32, 64, 128]:
            r = np.clip(t02.resize_float(crop, (S, S)), 0.0, 1.0)
            self.assertEqual(r.shape, (S, S))
            self.assertTrue(np.all((r >= 0.0) & (r <= 1.0)))

    def test_non_square_to_square(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 8:32] = 1.0  # 32x24 HxW
        bbox = _bbox_from_binary(mask)
        crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(np.float32)
        self.assertEqual(crop.shape, (32, 24))
        r = np.clip(t02.resize_float(crop, (64, 64)), 0.0, 1.0)
        self.assertEqual(r.shape, (64, 64))

    def test_empty_mask_bbox_none(self):
        self.assertIsNone(_bbox_from_binary(np.zeros((64, 64), dtype=np.int64)))

    def test_resized_preserves_content(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        bbox = _bbox_from_binary(mask)
        self.assertIsNotNone(bbox)
        crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(np.float32)
        r = np.clip(t02.resize_float(crop, (64, 64)), 0.0, 1.0)
        self.assertGreaterEqual(float(r[32, 32]), 0.9)


# ============================================================================
# H: Ring background
# ============================================================================


class TestComputeRingMask(unittest.TestCase):
    """H. _compute_ring_mask."""

    def test_ring_region(self):
        fh, fw = 32, 32
        bbox = [10.0, 10.0, 22.0, 22.0]
        gh, gw = 64, 64
        ring = t02._compute_ring_mask(fh, fw, bbox, gh, gw, ring_width=4)
        self.assertEqual(ring.shape, (32, 32))
        self.assertEqual(ring.dtype, bool)
        # interior False
        sc = 0.5
        ix1, iy1 = int(10 * sc), int(10 * sc)
        ix2 = min(32, int(np.ceil(22 * sc)))
        iy2 = min(32, int(np.ceil(22 * sc)))
        if ix1 < ix2 and iy1 < iy2:
            self.assertFalse(ring[iy1:iy2, ix1:ix2].any())
        self.assertTrue(ring.any())

    def test_ring_width_proportional(self):
        fh, fw = 32, 32
        bbox = [20.0, 20.0, 44.0, 44.0]
        gh, gw = 64, 64
        c1 = int(t02._compute_ring_mask(fh, fw, bbox, gh, gw, ring_width=3).sum())
        c2 = int(t02._compute_ring_mask(fh, fw, bbox, gh, gw, ring_width=9).sum())
        self.assertGreaterEqual(c2, c1)

    def test_boundary_clamped(self):
        ring = t02._compute_ring_mask(32, 32, [0, 0, 10, 10], 64, 64, ring_width=15)
        self.assertEqual(ring.shape, (32, 32))

    def test_far_pixels_excluded(self):
        ring = t02._compute_ring_mask(32, 32, [16, 16, 24, 24], 64, 64, ring_width=2)
        for r, c in [(0, 0), (0, 31), (31, 0), (31, 31)]:
            self.assertFalse(ring[r, c])

    def test_binary(self):
        ring = t02._compute_ring_mask(32, 32, [8, 8, 24, 24], 128, 128, ring_width=3)
        self.assertTrue(set(np.unique(ring)).issubset({False, True}))


# ============================================================================
# I: sample_rows + 3D per-case balancing
# ============================================================================


class TestSampleRows(unittest.TestCase):
    """sample_rows(array, max_count, rng)."""

    def test_below_max(self):
        arr = np.arange(20).reshape(10, 2).astype(np.float32)
        r = t02.sample_rows(arr, 15, np.random.default_rng(42))
        self.assertEqual(r.shape, (10, 2))

    def test_above_max(self):
        arr = np.arange(200).reshape(100, 2).astype(np.float32)
        r = t02.sample_rows(arr, 30, np.random.default_rng(42))
        self.assertEqual(r.shape, (30, 2))
        self.assertTrue({tuple(x) for x in r}.issubset({tuple(x) for x in arr}))

    def test_exact_max(self):
        arr = np.arange(100).reshape(50, 2).astype(np.float32)
        r = t02.sample_rows(arr, 50, np.random.default_rng(42))
        self.assertEqual(r.shape, (50, 2))

    def test_empty(self):
        r = t02.sample_rows(np.empty((0, 5), dtype=np.float32), 10,
                            np.random.default_rng(42))
        self.assertEqual(r.shape[0], 0)


class TestResizeMaskNearest(unittest.TestCase):
    def test_labels_preserved(self):
        mask = np.zeros((64, 64), dtype=np.int64)
        mask[16:48, 16:48] = 2
        mask[32:48, 32:48] = 5
        r = t02.resize_mask_nearest(mask, (32, 32))
        self.assertEqual(r.shape, (32, 32))
        self.assertEqual(r.dtype, np.int64)
        self.assertTrue(set(np.unique(r)).issubset({0, 2, 5}))


class Test3DPerCaseBalancing(unittest.TestCase):
    """I. Per-case capping logic."""

    @staticmethod
    def _cap(parts, lid, max_fg, rng):
        nc = len(parts)
        if nc == 0:
            return []
        cap = max(128, int(np.ceil(max_fg / nc)))
        out = []
        for cd in parts.values():
            ps = cd.get(lid, [])
            if not ps:
                continue
            ca = np.concatenate(ps, axis=0)
            c = t02.sample_rows(ca, cap, rng)
            if c.shape[0] > 0:
                out.append(c)
        return out

    def test_multiple_cases(self):
        dim = 64
        rng = np.random.default_rng(2026)
        parts = {
            "A": {1: [rng.standard_normal((100, dim), dtype=np.float32)]},
            "B": {1: [rng.standard_normal((300, dim), dtype=np.float32)]},
            "C": {1: [rng.standard_normal((200, dim), dtype=np.float32)]},
        }
        capped = self._cap(parts, 1, 500, rng)
        self.assertGreater(len(capped), 0)
        for p in capped:
            self.assertLessEqual(p.shape[0], 167)  # max(128, ceil(500/3))=167

    def test_single_case(self):
        dim = 64
        rng = np.random.default_rng(2026)
        parts = {"only": {1: [rng.standard_normal((50, dim), dtype=np.float32)]}}
        capped = self._cap(parts, 1, 2000, rng)
        self.assertEqual(len(capped), 1)
        self.assertEqual(capped[0].shape[0], 50)

    def test_small_max_fg(self):
        dim = 64
        rng = np.random.default_rng(2026)
        parts = {
            f"c{i}": {1: [rng.standard_normal((500, dim), dtype=np.float32)]}
            for i in range(10)
        }
        capped = self._cap(parts, 1, 300, rng)
        self.assertEqual(len(capped), 10)
        for p in capped:
            self.assertLessEqual(p.shape[0], 128)  # max(128, ceil(300/10))=128


# ============================================================================
# Helper function tests
# ============================================================================


class TestLoadJson(unittest.TestCase):
    def test_dict(self):
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False) as f:
            json.dump({"k": [1, 2, 3]}, f)
            f.flush()
            fp = Path(f.name)
        try:
            self.assertEqual(t02.load_json(fp), {"k": [1, 2, 3]})
        finally:
            os.unlink(fp)

    def test_list(self):
        with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False) as f:
            json.dump([1, 2, 3], f)
            f.flush()
            fp = Path(f.name)
        try:
            self.assertEqual(t02.load_json(fp), [1, 2, 3])
        finally:
            os.unlink(fp)


class TestLoadImageNpy(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.mkdtemp(prefix="t02_img_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._td, ignore_errors=True)

    def test_2d_to_3channel(self):
        p = Path(self._td) / "i.npy"
        np.save(str(p), np.random.rand(256, 256).astype(np.float32))
        r = t02.load_image_npy(p)
        self.assertEqual(r.ndim, 3)
        self.assertEqual(r.shape[-1], 3)

    def test_uint8_normalized(self):
        p = Path(self._td) / "i.npy"
        np.save(str(p), (np.ones((256, 256), dtype=np.uint8) * 128))
        self.assertTrue(t02.load_image_npy(p).max() <= 1.0)

    def test_channels_last(self):
        p = Path(self._td) / "i.npy"
        np.save(str(p), np.random.rand(3, 256, 256).astype(np.float32))
        r = t02.load_image_npy(p)
        self.assertEqual(r.ndim, 3)
        self.assertEqual(r.shape[-1], 3)


class TestLoadMaskNpy(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.mkdtemp(prefix="t02_mask_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._td, ignore_errors=True)

    def test_squeeze_hw1(self):
        p = Path(self._td) / "m.npy"
        np.save(str(p), np.ones((128, 128, 1), dtype=np.int32))
        self.assertEqual(t02.load_mask_npy(p).shape, (128, 128))

    def test_already_2d(self):
        p = Path(self._td) / "m.npy"
        np.save(str(p), np.ones((128, 128), dtype=np.int32))
        self.assertEqual(t02.load_mask_npy(p).shape, (128, 128))


class TestResolvePath(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.mkdtemp(prefix="t02_resolve_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._td, ignore_errors=True)

    def test_absolute(self):
        abs_path = os.path.join(self._td, "real.npy")
        Path(abs_path).touch()
        r = t02.resolve_path(Path(self._td), {"k": abs_path, "slice_name": "x"},
                             keys=("k",), fallback_dirs=())
        self.assertEqual(str(r), abs_path)

    def test_relative(self):
        (Path(self._td) / "sub").mkdir()
        (Path(self._td) / "sub" / "f.npy").touch()
        r = t02.resolve_path(Path(self._td), {"k": "sub/f.npy"},
                             keys=("k",), fallback_dirs=())
        self.assertTrue(r.is_absolute())

    def test_not_found(self):
        with self.assertRaises(FileNotFoundError):
            t02.resolve_path(Path(self._td), {"slice_name": "x.npy"},
                             keys=("k",), fallback_dirs=("sub",))


class TestGetSliceName(unittest.TestCase):
    def test_explicit(self):
        self.assertEqual(t02.get_slice_name({"slice_name": "a.png"}), "a.png")

    def test_teacher_img_fallback(self):
        self.assertEqual(t02.get_slice_name({"teacher_img": "/p/img.npy"}), "img.npy")

    def test_student_img_fallback(self):
        self.assertEqual(t02.get_slice_name({"student_img": "/p/s.npy"}), "s.npy")

    def test_missing_raises(self):
        with self.assertRaises(KeyError):
            t02.get_slice_name({})


class TestBuildUniqueManifestIndex(unittest.TestCase):
    def test_index(self):
        idx = t02.build_unique_manifest_index([
            {"slice_name": "a.npy", "s": "train"},
            {"slice_name": "b.npy", "s": "train"},
        ])
        self.assertEqual(len(idx), 2)
        self.assertEqual(idx["a.npy"]["s"], "train")

    def test_duplicate(self):
        with self.assertRaises(RuntimeError):
            t02.build_unique_manifest_index([
                {"slice_name": "a.npy"}, {"slice_name": "a.npy"},
            ])

    def test_non_dict(self):
        with self.assertRaises(TypeError):
            t02.build_unique_manifest_index(["x"])  # type: ignore[arg-type]


class TestAppendPositiveInts(unittest.TestCase):
    def test_valid(self):
        t: list[int] = []
        t02.append_positive_ints(t, [1, 2, 200, 254])
        self.assertEqual(t, [1, 2, 200, 254])

    def test_excludes_zero_neg(self):
        t: list[int] = []
        t02.append_positive_ints(t, [0, -1, 1, 2])
        self.assertEqual(t, [1, 2])

    def test_excludes_255_plus(self):
        t: list[int] = []
        t02.append_positive_ints(t, [1, 255, 256, 2])
        self.assertEqual(t, [1, 2])

    def test_non_list_ignored(self):
        t: list[int] = [10]
        t02.append_positive_ints(t, "str")
        self.assertEqual(t, [10])

    def test_non_int_skipped(self):
        t: list[int] = []
        t02.append_positive_ints(t, [1, "two", None, 3])
        self.assertEqual(t, [1, 3])


class TestReadClassIds(unittest.TestCase):
    def test_fallback(self):
        with tempfile.TemporaryDirectory() as td:
            r = t02.read_class_ids(Path(td), "m",
                                    [{"class_ids": [1, 3, 5]}, {"class_ids": [3, 7]}])
            self.assertEqual(r, [1, 3, 5, 7])

    def test_default(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(t02.read_class_ids(Path(td), "m", []), [1])


class TestInferRoundTag(unittest.TestCase):
    def test_standard(self):
        self.assertEqual(
            t02.infer_round_tag_from_method("idea1_iter_gt_iou_s2026_r0_full5"),
            "r00_full5")

    def test_two_digit(self):
        self.assertEqual(
            t02.infer_round_tag_from_method("idea1_iter_gt_iou_s2026_r12_full10"),
            "r12_full10")

    def test_unknown(self):
        self.assertEqual(
            t02.infer_round_tag_from_method("some_method"), "unknown_round")

    def test_no_match(self):
        self.assertEqual(
            t02.infer_round_tag_from_method("idea1_baseline"), "unknown_round")


# ============================================================================
if __name__ == "__main__":
    unittest.main()
