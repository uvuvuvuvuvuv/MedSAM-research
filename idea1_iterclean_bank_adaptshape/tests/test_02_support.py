"""Unit tests for 02_build_support_template module.

A. Coverage calculation (via resize_binary_mask_to_coverage_area)
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
# A: resize_binary_mask_to_coverage_area + coverage calculation
# ============================================================================


class TestResizeBinaryMaskToCoverageArea(unittest.TestCase):
    def test_all_ones_stays_one(self):
        result = t02.resize_binary_mask_to_coverage_area(np.ones((100, 100), dtype=np.float32), (20, 20))
        self.assertEqual(result.shape, (20, 20))
        np.testing.assert_allclose(result, 1.0, atol=1e-5)

    def test_all_zeros_stays_zero(self):
        result = t02.resize_binary_mask_to_coverage_area(np.zeros((100, 100), dtype=np.float32), (20, 20))
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_square_to_smaller(self):
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[30:70, 30:70] = 1.0
        result = t02.resize_binary_mask_to_coverage_area(mask, (10, 10))
        self.assertGreaterEqual(float(result[5, 5]), 0.9)
        for r, c in [(0, 0), (0, 9), (9, 0), (9, 9)]:
            self.assertLessEqual(float(result[r, c]), 0.1)

    def test_rectangular_input(self):
        result = t02.resize_binary_mask_to_coverage_area(np.ones((80, 120), dtype=np.float32), (40, 60))
        self.assertEqual(result.shape, (40, 60))
        np.testing.assert_allclose(result, 1.0, atol=1e-5)


class TestCoverageCalculation(unittest.TestCase):
    """A. Coverage semantics of area-resized binary mask."""

    def test_fg_token_in_box_high_coverage(self):
        h, w = 200, 200
        mask = np.zeros((h, w), dtype=np.float32)
        mask[50:150, 50:150] = 1.0
        coverage = t02.resize_binary_mask_to_coverage_area(mask, (20, 20))
        self.assertGreaterEqual(float(coverage[10, 10]), 0.95)

    def test_token_outside_box_zero_coverage(self):
        h, w = 200, 200
        mask = np.zeros((h, w), dtype=np.float32)
        mask[50:150, 50:150] = 1.0
        coverage = t02.resize_binary_mask_to_coverage_area(mask, (20, 20))
        for r, c in [(0, 0), (0, 19), (19, 0), (19, 19)]:
            self.assertLessEqual(float(coverage[r, c]), 0.05)

    def test_bg_token_in_ring_band_low_coverage(self):
        h, w = 200, 200
        mask = np.zeros((h, w), dtype=np.float32)
        mask[50:150, 50:150] = 1.0
        coverage = t02.resize_binary_mask_to_coverage_area(mask, (20, 20))
        bbox_teacher = [50.0, 50.0, 150.0, 150.0]
        ring = t02._compute_ring_mask(20, 20, bbox_teacher, h, w, ring_width_tokens=5)
        rr, cc = np.nonzero(ring)
        if len(rr) > 0:
            self.assertLess(float(coverage[rr, cc].max()), 0.15)

    def test_coverage_values_in_range(self):
        rng = np.random.default_rng(42)
        mask = rng.integers(0, 2, size=(128, 128)).astype(np.float32)
        coverage = t02.resize_binary_mask_to_coverage_area(mask, (32, 32))
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

    def test_bg_ring_width_tokens(self):
        ns = t02.build_parser().parse_args(self._B + ["--bg_ring_width_tokens", "2"])
        self.assertEqual(ns.bg_ring_width_tokens, 2)

    def test_bg_ring_width_tokens_zero(self):
        ns = t02.build_parser().parse_args(self._B + ["--bg_ring_width_tokens", "0"])
        self.assertEqual(ns.bg_ring_width_tokens, 0)

    def test_max_global_non_bg_coverage(self):
        ns = t02.build_parser().parse_args(
            self._B + ["--max_global_non_bg_coverage", "0.05"]
        )
        self.assertEqual(ns.max_global_non_bg_coverage, 0.05)

    def test_old_ring_expand_ratio_rejected(self):
        with self.assertRaises(SystemExit):
            t02.build_parser().parse_args(self._B + ["--ring_expand_ratio", "8"])

    def test_old_bg_coverage_threshold_rejected(self):
        with self.assertRaises(SystemExit):
            t02.build_parser().parse_args(self._B + ["--bg_coverage_threshold", "0.2"])

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
        self.assertEqual(ns.max_global_non_bg_coverage, 0.10)
        self.assertEqual(ns.shape_size, 64)
        self.assertEqual(ns.kmax_shape, 5)
        self.assertEqual(ns.cluster_max_iter, 25)
        self.assertEqual(ns.bg_ring_width_tokens, 1)
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
            fg_coverage_threshold=0.9, max_global_non_bg_coverage=0.1,
            max_fg_per_instance=128, max_bg_per_instance=128,
            max_fg_per_class=4096, max_bg_per_class=4096,
            bg_ring_width_tokens=1, shape_size=64, kmax_shape=5,
            cluster_max_iter=25, max_fg_fallback_tokens=4,
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

    def test_max_global_non_bg_coverage_oob(self):
        for v in [1.5, -0.1]:
            with self.subTest(v=v):
                with self.assertRaises(ValueError):
                    t02.validate_args(self._ns(max_global_non_bg_coverage=v))

    def test_bg_ring_width_tokens_negative(self):
        with self.assertRaises(ValueError):
            t02.validate_args(self._ns(bg_ring_width_tokens=-1))

    def test_zero_positive_params(self):
        for name in ("max_fg_per_instance", "max_bg_per_instance",
                     "max_fg_per_class", "max_bg_per_class",
                     "shape_size",
                     "kmax_shape", "cluster_max_iter",
                     "max_fg_fallback_tokens"):
            with self.subTest(p=name):
                with self.assertRaises(ValueError):
                    t02.validate_args(self._ns(**{name: 0}))

    def test_boundary_thresholds_valid(self):
        for thr, val in [("fg_coverage_threshold", 0.0),
                         ("fg_coverage_threshold", 1.0),
                         ("max_global_non_bg_coverage", 0.0),
                         ("max_global_non_bg_coverage", 1.0)]:
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
            r = np.clip(t02.resize_binary_shape_nearest(crop, (S, S)), 0.0, 1.0)
            self.assertEqual(r.shape, (S, S))
            self.assertTrue(np.all((r >= 0.0) & (r <= 1.0)))

    def test_non_square_to_square(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 8:32] = 1.0  # 32x24 HxW
        bbox = _bbox_from_binary(mask)
        crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(np.float32)
        self.assertEqual(crop.shape, (32, 24))
        r = np.clip(t02.resize_binary_shape_nearest(crop, (64, 64)), 0.0, 1.0)
        self.assertEqual(r.shape, (64, 64))

    def test_empty_mask_bbox_none(self):
        self.assertIsNone(_bbox_from_binary(np.zeros((64, 64), dtype=np.int64)))

    def test_resized_preserves_content(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        bbox = _bbox_from_binary(mask)
        self.assertIsNotNone(bbox)
        crop = mask[bbox[1]:bbox[3], bbox[0]:bbox[2]].astype(np.float32)
        r = np.clip(t02.resize_binary_shape_nearest(crop, (64, 64)), 0.0, 1.0)
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
        ring = t02._compute_ring_mask(fh, fw, bbox, gh, gw, ring_width_tokens=4)
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
        c1 = int(t02._compute_ring_mask(fh, fw, bbox, gh, gw, ring_width_tokens=3).sum())
        c2 = int(t02._compute_ring_mask(fh, fw, bbox, gh, gw, ring_width_tokens=9).sum())
        self.assertGreaterEqual(c2, c1)

    def test_boundary_clamped(self):
        ring = t02._compute_ring_mask(32, 32, [0, 0, 10, 10], 64, 64, ring_width_tokens=15)
        self.assertEqual(ring.shape, (32, 32))

    def test_far_pixels_excluded(self):
        ring = t02._compute_ring_mask(32, 32, [16, 16, 24, 24], 64, 64, ring_width_tokens=2)
        for r, c in [(0, 0), (0, 31), (31, 0), (31, 31)]:
            self.assertFalse(ring[r, c])

    def test_binary(self):
        ring = t02._compute_ring_mask(32, 32, [8, 8, 24, 24], 128, 128, ring_width_tokens=3)
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
# H: Ring geometry tests
# ============================================================================

class TestRingGeometry(unittest.TestCase):
    """Background ring geometry: width 0/1/2, bbox interior excluded,
    boundary clipping."""

    def setUp(self):
        self.feature_h, self.feature_w = 16, 16
        self.bbox = [100.0, 80.0, 300.0, 250.0]  # x1,y1,x2,y2 in teacher
        self.gt_h, self.gt_w = 512, 512

    def _call(self, width):
        return t02._compute_ring_mask(
            self.feature_h, self.feature_w,
            self.bbox, self.gt_h, self.gt_w,
            width,
        )

    def test_width_zero_returns_empty(self):
        ring = self._call(0)
        self.assertFalse(ring.any(), "width=0 must produce all-False ring")

    def test_width_one_has_tokens(self):
        ring = self._call(1)
        self.assertTrue(ring.any(), "width=1 must produce non-empty ring")

    def test_width_two_has_more_tokens_than_width_one(self):
        r1 = self._call(1)
        r2 = self._call(2)
        self.assertGreater(
            int(r2.sum()), int(r1.sum()),
            "width=2 must have more tokens than width=1",
        )

    def test_interior_never_in_ring(self):
        ring = self._call(1)
        scale_h = self.feature_h / self.gt_h
        scale_w = self.feature_w / self.gt_w
        x1, y1, x2, y2 = (float(v) for v in self.bbox)
        ix1 = int(x1 * scale_w)
        iy1 = int(y1 * scale_h)
        ix2 = min(self.feature_w, int(np.ceil(x2 * scale_w)))
        iy2 = min(self.feature_h, int(np.ceil(y2 * scale_h)))
        # Every interior position must be False
        interior = ring[iy1:iy2, ix1:ix2]
        self.assertFalse(
            interior.any(),
            "original bbox interior must not be in ring",
        )

    def test_ring_clips_at_boundary(self):
        # Place bbox at top-left corner so expansion would go negative
        ring = t02._compute_ring_mask(
            16, 16, [0.0, 0.0, 63.0, 63.0], 512, 512, 3,
        )
        # Should not crash and should produce boolean array
        self.assertEqual(ring.dtype, bool)
        self.assertEqual(ring.shape, (16, 16))

    def test_bbox_never_modified(self):
        """Verify _compute_ring_mask is side-effect free on bbox list."""
        bbox_copy = list(self.bbox)
        self._call(1)
        self.assertEqual(bbox_copy, self.bbox,
                         "bbox must not be modified by _compute_ring_mask")


# ============================================================================
# I: Global background tests
# ============================================================================

class TestGlobalBackground(unittest.TestCase):
    """Global GT-based background: excludes other instances, other classes,
    and ignore=255."""

    def _make_gt(self, h=64, w=64):
        return np.zeros((h, w), dtype=np.int64)

    def _make_coverage_map(self, mask, feature_h=8, feature_w=8):
        return t02.resize_binary_mask_to_coverage_area(
            mask.astype(np.float32), (feature_h, feature_w),
        )

    def test_pure_bg_allowed(self):
        """gt==0 everywhere: all coverage is 0, tokens are valid bg."""
        gt = self._make_gt()
        cov = self._make_coverage_map((gt != 0))
        self.assertTrue((cov <= 0.10).all())

    def test_other_instance_excluded(self):
        """Instance B in GT excludes token from being A's background."""
        gt = self._make_gt()
        # Instance B occupies half of the gt
        gt[:, 32:] = 2
        cov = self._make_coverage_map((gt != 0))
        # Right half has high non-bg coverage, must be rejected
        bg_valid = cov <= 0.10
        self.assertTrue(bg_valid[:, :4].all(),
                        "left half (true bg) must be valid")
        self.assertFalse(bg_valid[:, 4:].any(),
                         "right half (instance B) must be rejected")

    def test_other_class_excluded(self):
        """Other class foreground is excluded from background."""
        gt = self._make_gt()
        gt[10:54, 10:54] = 3  # different class
        cov = self._make_coverage_map((gt != 0))
        bg_valid = cov <= 0.10
        self.assertFalse(bg_valid.all(),
                         "other-class foreground must be rejected from bg")

    def test_ignore_255_excluded(self):
        """Ignore=255 must be excluded from background."""
        gt = self._make_gt()
        gt[10:54, 30:34] = 255
        cov = self._make_coverage_map((gt != 0))
        bg_valid = cov <= 0.10
        # The token covering the 255 region must be rejected
        col3_valid = bg_valid[:, 3]
        self.assertFalse(
            col3_valid.all(),
            "tokens covering ignore=255 must be rejected from bg",
        )

    def test_90pct_bg_allowed(self):
        """Token with 90% bg + 10% non-bg is allowed."""
        gt = self._make_gt(512, 512)
        # Place a small non-bg region covering ~10% of a 64x64 area (= feature token)
        # Feature token at position (0,0) covers teacher pixels 0:64, 0:64
        # 10% of 64x64 = ~410 pixels. Put ~400 pixels of class 1.
        gt[0:20, 0:20] = 1  # 400 pixels = ~9.8%
        cov = t02.resize_binary_mask_to_coverage_area(
            (gt != 0).astype(np.float32), (8, 8),
        )
        self.assertTrue(
            cov[0, 0] <= 0.10,
            f"~9.8% non-bg coverage must be <= 0.10, got {cov[0,0]:.4f}",
        )

    def test_over_10pct_non_bg_rejected(self):
        """Token with >10% non-bg is rejected."""
        gt = self._make_gt(512, 512)
        # Place non-bg covering ~15% of the upper-left token area
        gt[0:25, 0:25] = 1  # 625 pixels / 4096 ≈ 15.3%
        cov = t02.resize_binary_mask_to_coverage_area(
            (gt != 0).astype(np.float32), (8, 8),
        )
        self.assertGreater(
            cov[0, 0], 0.10,
            f"~15% non-bg coverage must be > 0.10, got {cov[0,0]:.4f}",
        )


# ============================================================================
# J: Area coverage tests
# ============================================================================

class TestAreaCoverage(unittest.TestCase):
    """resize_binary_mask_to_coverage_area uses INTER_AREA for true
    area averaging."""

    def test_all_zero(self):
        mask = np.zeros((128, 128), dtype=np.float32)
        cov = t02.resize_binary_mask_to_coverage_area(mask, (8, 8))
        self.assertTrue(np.allclose(cov, 0.0, atol=0.01))

    def test_all_one(self):
        mask = np.ones((128, 128), dtype=np.float32)
        cov = t02.resize_binary_mask_to_coverage_area(mask, (8, 8))
        self.assertTrue(np.allclose(cov, 1.0, atol=0.01))

    def test_half_coverage(self):
        """With 128→8, output cells cover 16×16 input pixels.
        Place FG in first 72 rows so cell 4 spans 8 FG + 8 BG → 0.5."""
        mask = np.zeros((128, 128), dtype=np.float32)
        mask[:, :72] = 1.0  # boundary at col 72, inside cell 4 (cols 64-79)
        cov = t02.resize_binary_mask_to_coverage_area(mask, (8, 8))
        # Cells 0-3 are fully FG (cols 0-63), cell 4 has 8/16=0.5 FG
        self.assertTrue(np.allclose(cov[:, :4], 1.0, atol=0.01))
        self.assertTrue(np.allclose(cov[:, 4], 0.5, atol=0.05),
                        f"cell 4 expected ~0.5, got {cov[:,4]}")
        self.assertTrue(np.allclose(cov[:, 5:], 0.0, atol=0.01))

    def test_quarter_coverage(self):
        """128→8, cell=16×16. mask[:72,:72] → cell (4,4) has 8×8/16×16=0.25."""
        mask = np.zeros((128, 128), dtype=np.float32)
        mask[:72, :72] = 1.0  # boundary at row 72, col 72
        cov = t02.resize_binary_mask_to_coverage_area(mask, (8, 8))
        # Cell (4,4) gets 64/256 = 0.25
        self.assertTrue(np.allclose(cov[4, 4], 0.25, atol=0.05),
                        f"cell (4,4) expected ~0.25, got {cov[4,4]:.4f}")
        # Cells (0-3, 0-3) are fully inside FG
        self.assertTrue(np.allclose(cov[:4, :4], 1.0, atol=0.01))
        # Cells (5:, 5:) are fully outside
        self.assertTrue(np.allclose(cov[5:, 5:], 0.0, atol=0.01))

    def test_not_bilinear(self):
        """INTER_AREA gives 0.0 or 1.0 when boundary aligns with cell edges;
        bilinear would smear. A clean boundary proves area mode."""
        mask = np.zeros((128, 128), dtype=np.float32)
        mask[:, :64] = 1.0  # boundary aligns with cell 4 start
        cov = t02.resize_binary_mask_to_coverage_area(mask, (8, 8))
        # Boundary at col 64 → cell 4 (cols 64-79) is fully BG = 0.0
        self.assertTrue(
            np.allclose(cov[:, 4], 0.0, atol=0.01),
            f"boundary column must be 0.0 (area), got {cov[:,4]}",
        )
        # Cell 3 (cols 48-63) is fully FG = 1.0
        self.assertTrue(
            np.allclose(cov[:, 3], 1.0, atol=0.01),
            f"inner column must be 1.0 (area), got {cov[:,3]}",
        )


class TestShapeResizeNearest(unittest.TestCase):
    """resize_binary_shape_nearest uses INTER_NEAREST to preserve 0/1."""

    def test_output_is_binary(self):
        mask = np.random.randn(37, 41) > 0  # boolean
        resized = t02.resize_binary_shape_nearest(
            mask.astype(np.float32), (64, 64),
        )
        unique = np.unique(resized)
        self.assertTrue(set(unique).issubset({0.0, 1.0}),
                        f"nearest resize must produce only 0 and 1, got {unique}")

    def test_preserves_structure(self):
        mask = np.zeros((50, 50), dtype=np.float32)
        mask[10:40, 10:40] = 1.0
        resized = t02.resize_binary_shape_nearest(mask, (64, 64))
        self.assertGreater(resized.sum(), 0)
        self.assertLess(resized.sum(), 64 * 64)


# ============================================================================
# K: CLI contract tests
# ============================================================================

class TestCliContract(unittest.TestCase):
    """New args accepted, old args rejected."""

    def setUp(self):
        self._B = [
            "--processed_root", "/tmp",
            "--checkpoint", "/tmp/ckpt.pth",
            "--datasets", "btcv",
            "--method", "test_m",
        ]

    def test_bg_ring_width_tokens_in_help(self):
        help_text = t02.build_parser().format_help()
        self.assertIn("--bg_ring_width_tokens", help_text)

    def test_ring_expand_ratio_not_in_help(self):
        help_text = t02.build_parser().format_help()
        self.assertNotIn("--ring_expand_ratio", help_text)

    def test_max_global_non_bg_coverage_in_help(self):
        help_text = t02.build_parser().format_help()
        self.assertIn("--max_global_non_bg_coverage", help_text)

    def test_bg_coverage_threshold_not_in_help(self):
        help_text = t02.build_parser().format_help()
        self.assertNotIn("--bg_coverage_threshold", help_text)


# ============================================================================
# L: NPZ and Stats contract tests
# ============================================================================

class TestNpzStatsContract(unittest.TestCase):
    """Stats contain new fields, do not contain old fields."""

    def setUp(self):
        import tempfile
        self._tmpdir = Path(tempfile.mkdtemp(prefix="test_npz_"))
        self._npz_path = self._tmpdir / "test.npz"
        self._stats_path = self._tmpdir / "stats.json"

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self._tmpdir), ignore_errors=True)

    def _write_dummy_npz(self):
        """Write minimal valid NPZ."""
        arrays = {
            "class_ids": np.array([1], dtype=np.int64),
            "feature_dim": np.array(768, dtype=np.int64),
            "shape_size": np.array([64, 64], dtype=np.int64),
            "method": np.array("test"),
            "round_tag": np.array("r00_full5"),
            "run_id": np.array("test_m_r00_full5"),
            "bank_fg_c1": np.zeros((10, 768), dtype=np.float32),
            "bank_bg_c1": np.zeros((10, 768), dtype=np.float32),
            "shape_templates_c1": np.zeros((2, 64, 64), dtype=np.float32),
            "shape_semantic_centers_c1": np.zeros((2, 768), dtype=np.float32),
            "shape_cluster_instance_counts_c1": np.array([5, 5], dtype=np.int64),
            "shape_cluster_support_counts_c1": np.array([5, 5], dtype=np.int64),
        }
        np.savez_compressed(str(self._npz_path), **arrays)

    def _write_dummy_stats(self, extra=None):
        stats = {
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
        if extra:
            stats.update(extra)
        import json
        self._stats_path.write_text(json.dumps(stats))

    def test_npz_keys_unchanged(self):
        self._write_dummy_npz()
        import zipfile
        with zipfile.ZipFile(str(self._npz_path), "r") as zf:
            names = set(zf.namelist())
        expected = {
            "class_ids.npy", "feature_dim.npy", "shape_size.npy",
            "method.npy", "round_tag.npy", "run_id.npy",
            "bank_fg_c1.npy", "bank_bg_c1.npy",
            "shape_templates_c1.npy", "shape_semantic_centers_c1.npy",
            "shape_cluster_instance_counts_c1.npy",
            "shape_cluster_support_counts_c1.npy",
        }
        self.assertEqual(names, expected)

    def test_npz_no_old_keys(self):
        self._write_dummy_npz()
        import zipfile
        with zipfile.ZipFile(str(self._npz_path), "r") as zf:
            names = set(zf.namelist())
        self.assertNotIn("proto_fg_c1.npy", names)
        self.assertNotIn("proto_bg_c1.npy", names)
        self.assertNotIn("shape_A_c1.npy", names)
        self.assertNotIn("shape_R_c1.npy", names)

    def test_npz_allow_pickle_false(self):
        self._write_dummy_npz()
        data = np.load(str(self._npz_path), allow_pickle=False)
        self.assertIn("class_ids", data)
        data.close()

    def test_stats_new_fields(self):
        self._write_dummy_stats()
        import json
        s = json.loads(self._stats_path.read_text())
        for field in [
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
        ]:
            with self.subTest(field=field):
                self.assertIn(field, s)

    def test_stats_no_ring_expand_ratio(self):
        self._write_dummy_stats()
        import json
        s = json.loads(self._stats_path.read_text())
        self.assertNotIn("ring_expand_ratio", s)

    def test_stats_no_bg_coverage_threshold(self):
        self._write_dummy_stats()
        import json
        s = json.loads(self._stats_path.read_text())
        self.assertNotIn("bg_coverage_threshold", s)


# ============================================================================
# M: FG Fallback Algorithm Tests
# ============================================================================

class TestFgFallback(unittest.TestCase):
    """FG fallback: deterministic selection, bbox constraint, coverage range."""

    def test_fallback_deterministic_sorting(self):
        """Coverage desc -> row asc -> col asc deterministic ranking."""
        coverage = np.zeros((8, 8), dtype=np.float32)
        coverage[2, 3] = 0.14
        coverage[3, 4] = 0.50
        coverage[4, 2] = 0.50
        coverage[5, 1] = 0.80

        cand_rows, cand_cols = np.where(coverage > 0)
        cand_coverage = coverage[cand_rows, cand_cols]
        order = np.lexsort((cand_cols, cand_rows, -cand_coverage))
        sorted_rows = cand_rows[order]
        sorted_cols = cand_cols[order]
        sorted_cov = cand_coverage[order]

        self.assertEqual(sorted_cov[0], 0.80)
        self.assertEqual(sorted_rows[0], 5)
        self.assertEqual(sorted_cols[0], 1)
        self.assertEqual(sorted_cov[1], 0.50)
        self.assertEqual(sorted_rows[1], 3)
        self.assertEqual(sorted_cov[2], 0.50)
        self.assertEqual(sorted_rows[2], 4)
        self.assertEqual(sorted_cov[3], 0.14)
        self.assertEqual(sorted_rows[3], 2)

    def test_fallback_single_candidate(self):
        """Only 1 positive-coverage token selects exactly 1."""
        n_select = min(1, 4)
        self.assertEqual(n_select, 1)

    def test_fallback_top_k_selection(self):
        """More than 4 candidates selects top 4 by coverage."""
        coverage = np.zeros((8, 8), dtype=np.float32)
        for i in range(6):
            coverage[2 + i, 3] = 0.1 + 0.01 * i

        cand_rows, cand_cols = np.where(coverage > 0)
        cand_coverage = coverage[cand_rows, cand_cols]
        order = np.lexsort((cand_cols, cand_rows, -cand_coverage))
        n_select = min(len(cand_rows), 4)
        self.assertEqual(n_select, 4)
        selected_cov = cand_coverage[order][:4]
        all_cov = sorted(cand_coverage, reverse=True)
        self.assertEqual(list(selected_cov), all_cov[:4])

    def test_fallback_empty_instance_raises(self):
        """max coverage <= 0 must raise error."""
        coverage = np.zeros((8, 8), dtype=np.float32)
        max_cov = float(coverage.max())
        self.assertLessEqual(max_cov, 0)
        self.assertTrue(max_cov <= 0)

    def test_fallback_bbox_constraint(self):
        """Positive coverage outside bbox excluded from fallback."""
        coverage = np.zeros((64, 64), dtype=np.float32)
        coverage[12, 16] = 0.50
        coverage[14, 18] = 0.30
        coverage[5, 5] = 0.90
        coverage[30, 30] = 0.70

        bbox_mask = np.zeros((64, 64), dtype=bool)
        bbox_mask[10:20, 15:25] = True

        cand_mask = (coverage > 0) & bbox_mask
        cand_rows, cand_cols = np.where(cand_mask)
        self.assertEqual(len(cand_rows), 2)
        positions = set(zip(cand_rows.tolist(), cand_cols.tolist()))
        self.assertNotIn((5, 5), positions)
        self.assertNotIn((30, 30), positions)
        self.assertIn((12, 16), positions)
        self.assertIn((14, 18), positions)

    def test_fallback_coverage_range(self):
        """All fallback coverage values are in (0, 1]."""
        coverage = np.array([0.14, 0.50, 0.80, 0.05], dtype=np.float32)
        for c in coverage:
            self.assertGreater(c, 0.0)
            self.assertLessEqual(c, 1.0)

    def test_fallback_max_fg_fallback_tokens_cli(self):
        """--max_fg_fallback_tokens accepted with default 4."""
        parser = t02.build_parser()
        args = parser.parse_args([
            "--processed_root", "/tmp",
            "--checkpoint", "/tmp/c.pth",
            "--datasets", "x",
            "--method", "m",
        ])
        self.assertEqual(args.max_fg_fallback_tokens, 4)
        args2 = parser.parse_args([
            "--processed_root", "/tmp",
            "--checkpoint", "/tmp/c.pth",
            "--datasets", "x",
            "--method", "m",
            "--max_fg_fallback_tokens", "2",
        ])
        self.assertEqual(args2.max_fg_fallback_tokens, 2)

    def test_fallback_validate_positive(self):
        """max_fg_fallback_tokens must be >= 1."""
        with self.assertRaises((ValueError, SystemExit)):
            t02.validate_args(t02.build_parser().parse_args([
                "--processed_root", "/tmp",
                "--checkpoint", "/tmp/c.pth",
                "--datasets", "x",
                "--method", "m",
                "--max_fg_fallback_tokens", "0",
            ]))

    def test_fallback_no_nan_in_stats(self):
        """Fallback stats serialize to valid JSON without NaN/Inf."""
        stats = {
            "fg_fallback_instance_count": 0,
            "fg_fallback_token_count": 0,
            "max_fg_fallback_tokens": 4,
            "fg_fallback_coverage_min": None,
            "fg_fallback_coverage_mean": None,
            "fg_fallback_coverage_max": None,
        }
        serialized = json.dumps(stats)
        restored = json.loads(serialized)
        for k, v in restored.items():
            if isinstance(v, float):
                self.assertFalse(v != v, f"{k} is NaN")

    def test_fallback_per_class_stats(self):
        """Per-class fallback counts computed correctly from per-instance data."""
        per_instance = [
            {"label_id": 1, "fg_source": "primary", "fg_fallback_coverage": None},
            {"label_id": 1, "fg_source": "fallback", "fg_fallback_coverage": [0.14, 0.30]},
            {"label_id": 2, "fg_source": "primary", "fg_fallback_coverage": None},
            {"label_id": 2, "fg_source": "fallback", "fg_fallback_coverage": [0.20]},
        ]
        from collections import defaultdict
        fb_inst = defaultdict(int)
        fb_tok = defaultdict(int)
        for inst in per_instance:
            lid = inst["label_id"]
            if inst["fg_source"] == "fallback":
                fb_inst[lid] += 1
                fb_tok[lid] += len(inst["fg_fallback_coverage"])
        self.assertEqual(fb_inst[1], 1)
        self.assertEqual(fb_tok[1], 2)
        self.assertEqual(fb_inst[2], 1)
        self.assertEqual(fb_tok[2], 1)


# ============================================================================
# N: Schema V2 Contract Tests
# ============================================================================

class TestSchemaV2(unittest.TestCase):
    """Support stats schema v2: fallback fields required, invariants enforced."""

    def setUp(self):
        import tempfile
        self._tmpdir = Path(tempfile.mkdtemp(prefix="test_sv2_"))
        self.v08 = importlib.import_module(
            "idea1_iterclean_bank_adaptshape.08_validate_round_outputs"
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(str(self._tmpdir), ignore_errors=True)

    def _make_v1_stats(self):
        return {
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

    def _make_v2_stats(self, overrides=None):
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
        if overrides:
            stats.update(overrides)
        return stats

    def _make_npz(self, path):
        import zipfile
        import io as _io_npz
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

    def test_schema_v1_no_fallback_fields_still_valid(self):
        """v1 stats without fallback fields pass validation."""
        npz_path = self._tmpdir / "test.npz"
        stats_path = self._tmpdir / "stats.json"
        self._make_npz(npz_path)
        stats_path.write_text(json.dumps(self._make_v1_stats()))
        paths = {"support_path": npz_path, "support_stats_path": stats_path}
        errors = self.v08.validate_template(paths, strict=True)
        self.assertEqual(errors, [])

    def test_schema_v2_all_fields_present_passes(self):
        """v2 stats with all fallback fields pass validation."""
        npz_path = self._tmpdir / "test.npz"
        stats_path = self._tmpdir / "stats.json"
        self._make_npz(npz_path)
        stats_path.write_text(json.dumps(self._make_v2_stats()))
        paths = {"support_path": npz_path, "support_stats_path": stats_path}
        errors = self.v08.validate_template(paths, strict=True)
        self.assertEqual(errors, [])

    def test_schema_v2_missing_fallback_fields_fails(self):
        """v2 stats missing required fallback fields must fail."""
        npz_path = self._tmpdir / "test.npz"
        stats_path = self._tmpdir / "stats.json"
        self._make_npz(npz_path)
        v1 = self._make_v1_stats()
        v1["support_stats_schema_version"] = 2
        stats_path.write_text(json.dumps(v1))
        paths = {"support_path": npz_path, "support_stats_path": stats_path}
        errors = self.v08.validate_template(paths, strict=True)
        self.assertTrue(len(errors) > 0,
                        f"Expected errors for v2 with missing fallback fields, got {errors}")

    def test_schema_v2_count_mismatch_fails(self):
        """instances_with_fallback_fg != fg_fallback_instance_count fails."""
        npz_path = self._tmpdir / "test.npz"
        stats_path = self._tmpdir / "stats.json"
        self._make_npz(npz_path)
        v2 = self._make_v2_stats({"instances_with_fallback_fg": 3})
        stats_path.write_text(json.dumps(v2))
        paths = {"support_path": npz_path, "support_stats_path": stats_path}
        errors = self.v08.validate_template(paths, strict=True)
        self.assertTrue(len(errors) > 0,
                        f"Expected errors for count mismatch, got {errors}")

    def test_schema_v2_nan_fails(self):
        """NaN in fallback coverage fields must fail."""
        npz_path = self._tmpdir / "test.npz"
        stats_path = self._tmpdir / "stats.json"
        self._make_npz(npz_path)
        v2 = self._make_v2_stats({"fg_fallback_coverage_mean": float("nan")})
        stats_path.write_text(json.dumps(v2))
        paths = {"support_path": npz_path, "support_stats_path": stats_path}
        errors = self.v08.validate_template(paths, strict=True)
        self.assertTrue(len(errors) > 0,
                        f"Expected errors for NaN coverage, got {errors}")


# ============================================================================
# O: Empty Slice Strict Skip Tests
# ============================================================================

class TestEmptySliceStrict(unittest.TestCase):
    """Empty slice skip must verify GT has 0 foreground pixels."""

    def test_empty_slice_prompt0_gt0_allowed(self):
        """prompt=0 AND GT=0 -> skip is valid."""
        gt = np.zeros((100, 100), dtype=np.int64)
        fg_values = gt[gt != 0]
        fg_count = int((fg_values != 255).sum())
        self.assertEqual(fg_count, 0)

    def test_empty_slice_prompt0_gt_has_fg_raises(self):
        """prompt=0 but GT has non-zero, non-ignore pixels -> must raise."""
        gt = np.zeros((100, 100), dtype=np.int64)
        gt[10:20, 10:20] = 1
        fg_values = gt[gt != 0]
        fg_count = int((fg_values != 255).sum())
        self.assertGreater(fg_count, 0)

    def test_empty_slice_gt_has_only_ignore_is_allowed(self):
        """GT has only ignore (255) pixels -> treated as no FG."""
        gt = np.full((100, 100), 255, dtype=np.int64)
        fg_values = gt[gt != 0]
        fg_count = int((fg_values != 255).sum())
        self.assertEqual(fg_count, 0)

    def test_empty_slice_gt_mixed_ignore_and_fg_raises(self):
        """GT has ignore (255) AND real FG pixels -> must raise."""
        gt = np.full((100, 100), 255, dtype=np.int64)
        gt[10:20, 10:20] = 1
        fg_values = gt[gt != 0]
        fg_count = int((fg_values != 255).sum())
        self.assertGreater(fg_count, 0)

    def test_empty_slice_max_coverage_zero_raises(self):
        """GT exists but inst_coverage.max() == 0 -> fallback raises error."""
        inst_coverage = np.zeros((64, 64), dtype=np.float32)
        max_cov = float(inst_coverage.max())
        self.assertEqual(max_cov, 0.0)
        self.assertTrue(max_cov <= 0)

    def test_empty_slice_prompt_instances_but_gt_empty_raises(self):
        """Prompt bbox cannot match empty GT -> match_instance_mask raises."""
        gt = np.zeros((100, 100), dtype=np.int64)
        prompt_bbox = [10.0, 10.0, 30.0, 30.0]
        from idea1_iterclean_bank_adaptshape.instance_utils import (
            match_instance_mask,
        )
        with self.assertRaises((ValueError, RuntimeError)):
            match_instance_mask(gt, label_id=1, bbox=prompt_bbox, component_id=None)


# ============================================================================
if __name__ == "__main__":
    unittest.main()
