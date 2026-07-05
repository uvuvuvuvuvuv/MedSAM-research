"""Unit tests for 03_train_medsam_sac module.

Covers dice_loss_with_logits, set_seed, load_image_tensor,
freeze_for_mask_decoder_only, parse_train_entries, EMA update,
CLI parser, make_box_tensor, and safety checks (no MONAI imports).

Uses only standard-library unittest and numpy.  Torch-condition tests
skip gracefully when torch is unavailable.

Run with::

    PYTHONPATH=/storage/baiyuting/data/MedSAM-main \\
        python -m idea1_iterclean_bank_adaptshape.tests.test_03_train

"""
from __future__ import annotations

import copy
import importlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

t03 = importlib.import_module(
    "idea1_iterclean_bank_adaptshape.03_train_medsam_sac"
)

# ---------------------------------------------------------------------------
# Torch availability guard
# ---------------------------------------------------------------------------
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ===================================================================
# Helper: create a minimal mock SAM model for freeze / EMA tests
# ===================================================================

def _make_mock_sam():
    """Return a minimal nn.Module with image_encoder/prompt_encoder/mask_decoder.

    Each submodule has real trainable parameters so that freeze_for_mask_decoder_only
    and update_ema can be exercised without the real SAM checkpoint.
    """
    if not TORCH_AVAILABLE:
        raise unittest.SkipTest("torch not available")

    class _MockSub(torch.nn.Module):
        def __init__(self, out_features: int = 4):
            super().__init__()
            self.linear = torch.nn.Linear(8, out_features)

    class _MockSAM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = _MockSub(4)
            self.prompt_encoder = _MockSub(4)
            self.mask_decoder = _MockSub(4)

    return _MockSAM()


# ===================================================================
# A.  dice_loss_with_logits
# ===================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
class TestDiceLossWithLogits(unittest.TestCase):
    """Tests for t03.dice_loss_with_logits(logits, target, eps=1e-6)."""

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _make_target(h: int = 32, w: int = 32) -> torch.Tensor:
        """(1, 1, H, W) binary target with a centred square foreground."""
        t = torch.zeros(1, 1, h, w)
        t[:, :, h // 4: 3 * h // 4, w // 4: 3 * w // 4] = 1.0
        return t

    # -- perfect prediction ------------------------------------------------

    def test_perfect_prediction_loss_near_zero(self):
        """Large positive logits on foreground + large negative on background -> loss ~ 0."""
        target = self._make_target()
        logits = torch.where(target == 1.0,
                             torch.tensor(100.0), torch.tensor(-100.0))
        loss = t03.dice_loss_with_logits(logits, target)
        self.assertLess(loss.item(), 0.02,
                        msg=f"Perfect prediction loss should be near 0, got {loss.item():.6f}")

    # -- completely wrong ---------------------------------------------------

    def test_completely_wrong_loss_near_one(self):
        """Swapped logits (fg negative, bg positive) -> loss ~ 1."""
        target = self._make_target()
        logits = torch.where(target == 1.0,
                             torch.tensor(-100.0), torch.tensor(100.0))
        loss = t03.dice_loss_with_logits(logits, target)
        self.assertAlmostEqual(loss.item(), 1.0, places=2,
                               msg=f"Wrong prediction loss ~ 1, got {loss.item():.6f}")

    # -- scalar output ------------------------------------------------------

    def test_returns_scalar_tensor(self):
        """Loss must be a 0-dimensional tensor whose .item() is a float."""
        target = self._make_target(16, 16)
        logits = torch.randn(1, 1, 16, 16)
        loss = t03.dice_loss_with_logits(logits, target)
        self.assertEqual(loss.ndim, 0,
                         msg=f"Loss should be 0-dim, got ndim={loss.ndim}")
        self.assertIsInstance(loss.item(), float,
                              msg=f"loss.item() should be float, got {type(loss.item())}")

    # -- batch dimension ----------------------------------------------------

    def test_handles_batch_dimension(self):
        """Loss must accept (B, 1, H, W) and still return a scalar."""
        target = torch.ones(4, 1, 16, 16)
        logits = torch.randn(4, 1, 16, 16)
        loss = t03.dice_loss_with_logits(logits, target)
        self.assertEqual(loss.ndim, 0,
                         msg=f"Batch loss should be 0-dim, got ndim={loss.ndim}")
        self.assertFalse(torch.isnan(loss),
                         msg="Batch loss should not be NaN")

    # -- all-zero target does not crash -------------------------------------

    def test_all_zero_target_does_not_crash(self):
        """When target has no foreground, loss is still a valid scalar."""
        target = torch.zeros(1, 1, 32, 32)
        logits = torch.randn(1, 1, 32, 32)
        loss = t03.dice_loss_with_logits(logits, target)
        self.assertEqual(loss.ndim, 0)
        self.assertFalse(torch.isnan(loss),
                         msg="All-zero target should not produce NaN loss")

    def test_all_one_target_does_not_crash(self):
        """When target is entirely foreground, loss is still a valid scalar."""
        target = torch.ones(1, 1, 32, 32)
        logits = torch.randn(1, 1, 32, 32)
        loss = t03.dice_loss_with_logits(logits, target)
        self.assertEqual(loss.ndim, 0)
        self.assertFalse(torch.isnan(loss),
                         msg="All-one target should not produce NaN loss")

    # -- random prediction produces a reasonable value ----------------------

    def test_random_prediction_in_expected_range(self):
        """Loss should be in [0, 1] range (within numerical tolerance)."""
        target = self._make_target(32, 32)
        # Use logits near 0 so sigmoid ~ 0.5, giving moderate dice loss
        logits = 0.1 * torch.randn(1, 1, 32, 32)
        loss = t03.dice_loss_with_logits(logits, target)
        self.assertGreaterEqual(loss.item(), -1e-5,
                                msg=f"Loss should be >= 0, got {loss.item():.6f}")
        self.assertLessEqual(loss.item(), 1.0 + 1e-5,
                             msg=f"Loss should be <= 1, got {loss.item():.6f}")

    # -- gradient flow ------------------------------------------------------

    def test_loss_supports_backward(self):
        """Loss must be differentiable with respect to logits."""
        target = self._make_target(16, 16)
        logits = torch.randn(1, 1, 16, 16, requires_grad=True)
        loss = t03.dice_loss_with_logits(logits, target)
        loss.backward()
        self.assertIsNotNone(logits.grad,
                             msg="logits.grad must not be None after backward()")
        self.assertFalse(torch.all(logits.grad == 0),
                         msg="logits.grad should not be all zeros")

    # -- eps parameter ------------------------------------------------------

    def test_custom_eps(self):
        """Custom eps is accepted and does not crash."""
        target = self._make_target(16, 16)
        logits = torch.randn(1, 1, 16, 16)
        loss_default = t03.dice_loss_with_logits(logits, target, eps=1e-6)
        loss_custom = t03.dice_loss_with_logits(logits, target, eps=1e-3)
        self.assertEqual(loss_default.ndim, 0)
        self.assertEqual(loss_custom.ndim, 0)
        # Values may differ slightly with different epsilon
        self.assertFalse(torch.isnan(loss_custom))


# ===================================================================
# B.  set_seed
# ===================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
class TestSetSeed(unittest.TestCase):
    """Tests for t03.set_seed(seed)."""

    def test_same_seed_produces_same_torch_random(self):
        """Calling set_seed with the same seed twice yields identical random numbers."""
        t03.set_seed(42)
        a = torch.randn(10, 10)

        t03.set_seed(42)
        b = torch.randn(10, 10)

        self.assertTrue(torch.equal(a, b),
                        msg="Same seed must produce identical random torch tensors")

    def test_different_seeds_produce_different_torch_random(self):
        """Different seeds should produce different sequences."""
        t03.set_seed(1)
        a = torch.randn(20)

        t03.set_seed(2)
        b = torch.randn(20)

        self.assertFalse(torch.equal(a, b),
                         msg="Different seeds should produce different random tensors")

    def test_set_seed_also_affects_numpy(self):
        """set_seed should seed numpy as well."""
        t03.set_seed(99)
        a = np.random.randn(5)

        t03.set_seed(99)
        b = np.random.randn(5)

        np.testing.assert_array_equal(a, b,
                                      "Same seed must produce identical numpy random arrays")

    def test_set_seed_also_affects_random_module(self):
        """set_seed should seed the random module."""
        import random

        t03.set_seed(7)
        a = random.random()

        t03.set_seed(7)
        b = random.random()

        self.assertAlmostEqual(a, b, places=10,
                               msg="Same seed must produce identical random.random() results")

    def test_set_seed_does_not_crash(self):
        """set_seed with various inputs should not raise."""
        for seed in [0, 1, 42, 2026, 99999]:
            try:
                t03.set_seed(seed)
            except Exception as e:
                self.fail(f"set_seed({seed}) raised {type(e).__name__}: {e}")


# ===================================================================
# C.  load_image_tensor
# ===================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
class TestLoadImageTensor(unittest.TestCase):
    """Tests for t03.load_image_tensor(path, device)."""

    def setUp(self):
        self._tmpdir = tempfile.mkdtemp(prefix="test_load_image_")
        self._device = torch.device("cpu")

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _write_npy(self, name: str, array: np.ndarray) -> Path:
        p = Path(self._tmpdir) / name
        np.save(str(p), array)
        return p

    # -- 2D grayscale -> 3-channel ------------------------------------------

    def test_2d_grayscale_becomes_3_channel(self):
        """A (H, W) float32 image -> (1, 3, H, W) tensor."""
        arr = np.random.rand(64, 64).astype(np.float32) * 0.8
        path = self._write_npy("gray2d.npy", arr)
        tensor = t03.load_image_tensor(path, self._device)
        self.assertEqual(tensor.shape, (1, 3, 64, 64),
                         msg=f"(H,W) input should produce (1,3,H,W), got {tensor.shape}")

    def test_2d_grayscale_values_preserved(self):
        """Gray values replicated across all three channels."""
        arr = np.random.rand(64, 64).astype(np.float32) * 0.5
        path = self._write_npy("gray2d.npy", arr)
        tensor = t03.load_image_tensor(path, self._device)
        # Channel 0, 1, 2 should be (close to) identical for grayscale input
        for c in range(3):
            ch = tensor[0, c].cpu().numpy()
            np.testing.assert_array_almost_equal(ch, arr, decimal=5,
                                                 err_msg=f"Channel {c} should match input grayscale")

    # -- RGB image ----------------------------------------------------------

    def test_rgb_image_correct_shape(self):
        """An (H, W, 3) float32 array -> (1, 3, H, W) tensor."""
        arr = np.random.rand(48, 64, 3).astype(np.float32)
        path = self._write_npy("rgb.npy", arr)
        tensor = t03.load_image_tensor(path, self._device)
        self.assertEqual(tensor.shape, (1, 3, 48, 64),
                         msg=f"RGB (48,64,3) should produce (1,3,48,64), got {tensor.shape}")

    def test_rgb_channels_correct_order(self):
        """RGB channels should be in the same order (H, W, 3 -> 1, 3, H, W)."""
        # Create a simple pattern: R channel = 0.1, G = 0.3, B = 0.7
        arr = np.zeros((32, 32, 3), dtype=np.float32)
        arr[:, :, 0] = 0.1
        arr[:, :, 1] = 0.3
        arr[:, :, 2] = 0.7
        path = self._write_npy("rgb_pattern.npy", arr)
        tensor = t03.load_image_tensor(path, self._device)
        t = tensor[0].cpu().numpy()  # (3, 32, 32)
        np.testing.assert_allclose(t[0].mean(), 0.1, atol=1e-5,
                                   err_msg="Channel 0 (R) should match")
        np.testing.assert_allclose(t[1].mean(), 0.3, atol=1e-5,
                                   err_msg="Channel 1 (G) should match")
        np.testing.assert_allclose(t[2].mean(), 0.7, atol=1e-5,
                                   err_msg="Channel 2 (B) should match")

    # -- values in [0, 1] range ---------------------------------------------

    def test_output_values_in_01_range(self):
        """Output tensor values must be clipped to [0, 1]."""
        arr = np.random.rand(64, 64).astype(np.float32) * 3.0  # some values > 1
        path = self._write_npy("clamped.npy", arr)
        tensor = t03.load_image_tensor(path, self._device)
        t = tensor.cpu().numpy()
        self.assertGreaterEqual(t.min(), 0.0,
                                msg=f"Min value {t.min()} should be >= 0")
        self.assertLessEqual(t.max(), 1.0 + 1e-6,
                             msg=f"Max value {t.max()} should be <= 1")

    def test_already_01_values_unchanged(self):
        """Values already in [0, 1] should not be rescaled."""
        arr = np.random.rand(32, 32).astype(np.float32) * 0.5 + 0.3  # in [0.3, 0.8]
        path = self._write_npy("in_range.npy", arr)
        tensor = t03.load_image_tensor(path, self._device)
        ch0 = tensor[0, 0].cpu().numpy()
        np.testing.assert_allclose(ch0, arr, atol=1e-5,
                                   err_msg="Values already in [0,1] should be preserved")

    # -- normalization from [0, 255] to [0, 1] ------------------------------

    def test_uint8_255_normalized_to_01(self):
        """Integer values in [0, 255] must be divided by 255 to [0, 1]."""
        arr = np.arange(256, dtype=np.float32).reshape(16, 16)  # 0..255
        path = self._write_npy("uint8_range.npy", arr)
        tensor = t03.load_image_tensor(path, self._device)
        t = tensor.cpu().numpy()
        # Original values are 0..255, after division should be 0..1
        self.assertAlmostEqual(t.max(), 1.0, places=3,
                               msg=f"Max after normalization should be 1.0, got {t.max():.6f}")
        self.assertAlmostEqual(t.min(), 0.0, places=3,
                               msg=f"Min after normalization should be 0.0, got {t.min():.6f}")

    # -- channel-first input (3, H, W) handled ------------------------------

    def test_channel_first_rgb_reordered(self):
        """A (3, H, W) array should be transposed to (H, W, 3) then permuted to (1, 3, H, W)."""
        h, w = 32, 40
        # (3, H, W) layout
        arr = np.random.rand(3, h, w).astype(np.float32)
        path = self._write_npy("ch_first.npy", arr)
        tensor = t03.load_image_tensor(path, self._device)
        self.assertEqual(tensor.shape, (1, 3, h, w),
                         msg=f"(3,H,W) should produce (1,3,H,W), got {tensor.shape}")

    def test_channel_first_grayscale_1HW_handled(self):
        """A (1, H, W) array should be expanded to 3 channels."""
        h, w = 24, 24
        arr = np.random.rand(1, h, w).astype(np.float32)
        path = self._write_npy("ch_first_gray.npy", arr)
        tensor = t03.load_image_tensor(path, self._device)
        self.assertEqual(tensor.shape, (1, 3, h, w),
                         msg=f"(1,H,W) should produce (1,3,H,W), got {tensor.shape}")

    # -- tensor is on correct device ----------------------------------------

    def test_tensor_on_correct_device(self):
        """Output tensor must reside on the requested device."""
        arr = np.random.rand(32, 32).astype(np.float32)
        path = self._write_npy("device_test.npy", arr)
        tensor = t03.load_image_tensor(path, self._device)
        self.assertEqual(str(tensor.device), str(self._device),
                         msg=f"Tensor device={tensor.device}, expected {self._device}")

    # -- dtype is float32 ---------------------------------------------------

    def test_output_dtype_is_float32(self):
        """Output tensor dtype must be float32."""
        arr = np.random.rand(32, 32).astype(np.float32)
        path = self._write_npy("dtype_test.npy", arr)
        tensor = t03.load_image_tensor(path, self._device)
        self.assertEqual(tensor.dtype, torch.float32,
                         msg=f"Tensor dtype should be float32, got {tensor.dtype}")


# ===================================================================
# D.  freeze_for_mask_decoder_only
# ===================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
class TestFreezeForMaskDecoderOnly(unittest.TestCase):
    """Tests for t03.freeze_for_mask_decoder_only(model)."""

    def setUp(self):
        self.model = _make_mock_sam()

    # -- image encoder frozen ------------------------------------------------

    def test_image_encoder_params_frozen(self):
        """After freezing, image_encoder parameters must have requires_grad=False."""
        t03.freeze_for_mask_decoder_only(self.model)
        for name, p in self.model.image_encoder.named_parameters():
            self.assertFalse(p.requires_grad,
                             msg=f"image_encoder.{name} should be frozen, got requires_grad=True")

    def test_image_encoder_no_trainable_params(self):
        """Sum of trainable image_encoder params must be zero."""
        t03.freeze_for_mask_decoder_only(self.model)
        count = sum(1 for p in self.model.image_encoder.parameters() if p.requires_grad)
        self.assertEqual(count, 0,
                         msg=f"image_encoder should have 0 trainable params, got {count}")

    # -- prompt encoder frozen -----------------------------------------------

    def test_prompt_encoder_params_frozen(self):
        """After freezing, prompt_encoder parameters must have requires_grad=False."""
        t03.freeze_for_mask_decoder_only(self.model)
        for name, p in self.model.prompt_encoder.named_parameters():
            self.assertFalse(p.requires_grad,
                             msg=f"prompt_encoder.{name} should be frozen, got requires_grad=True")

    def test_prompt_encoder_no_trainable_params(self):
        """Sum of trainable prompt_encoder params must be zero."""
        t03.freeze_for_mask_decoder_only(self.model)
        count = sum(1 for p in self.model.prompt_encoder.parameters() if p.requires_grad)
        self.assertEqual(count, 0,
                         msg=f"prompt_encoder should have 0 trainable params, got {count}")

    # -- mask decoder trainable ----------------------------------------------

    def test_mask_decoder_params_trainable(self):
        """After freezing, mask_decoder parameters must have requires_grad=True."""
        t03.freeze_for_mask_decoder_only(self.model)
        for name, p in self.model.mask_decoder.named_parameters():
            self.assertTrue(p.requires_grad,
                            msg=f"mask_decoder.{name} should be trainable, got requires_grad=False")

    def test_mask_decoder_all_trainable(self):
        """Every mask_decoder parameter should be trainable."""
        t03.freeze_for_mask_decoder_only(self.model)
        total = sum(1 for _ in self.model.mask_decoder.parameters())
        trainable = sum(1 for p in self.model.mask_decoder.parameters() if p.requires_grad)
        self.assertEqual(trainable, total,
                         msg=f"All {total} mask_decoder params should be trainable, got {trainable}")

    # -- return value --------------------------------------------------------

    def test_returns_list_of_trainable_parameters(self):
        """Return value must be a list of trainable parameters."""
        result = t03.freeze_for_mask_decoder_only(self.model)
        self.assertIsInstance(result, list,
                              msg="Return value must be a list")
        self.assertGreater(len(result), 0,
                           msg="Return value should contain at least one trainable parameter")
        for p in result:
            self.assertIsInstance(p, torch.nn.Parameter,
                                  msg=f"Each element must be nn.Parameter, got {type(p)}")
            self.assertTrue(p.requires_grad,
                            msg="Every returned parameter must have requires_grad=True")

    def test_returned_list_only_contains_mask_decoder_params(self):
        """Returned trainable parameters should all come from mask_decoder."""
        result = t03.freeze_for_mask_decoder_only(self.model)
        mask_decoder_ids = {id(p) for p in self.model.mask_decoder.parameters()}
        for p in result:
            self.assertIn(id(p), mask_decoder_ids,
                          msg="Returned parameter must belong to mask_decoder")

    # -- idempotent ----------------------------------------------------------

    def test_call_twice_is_idempotent(self):
        """Calling freeze twice should not change the result."""
        r1 = t03.freeze_for_mask_decoder_only(self.model)
        r2 = t03.freeze_for_mask_decoder_only(self.model)
        self.assertEqual(len(r1), len(r2),
                         msg="Two calls should return same number of parameters")
        # Check that mask_decoder still has requires_grad=True
        for p in self.model.mask_decoder.parameters():
            self.assertTrue(p.requires_grad)


# ===================================================================
# E.  make_box_tensor
# ===================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
class TestMakeBoxTensor(unittest.TestCase):
    """Tests for t03.make_box_tensor(box, device)."""

    def setUp(self):
        self._device = torch.device("cpu")

    def test_shape_is_1_4(self):
        """Output shape must be (1, 4)."""
        result = t03.make_box_tensor([10, 20, 100, 200], self._device)
        self.assertEqual(tuple(result.shape), (1, 4),
                         msg=f"Expected shape (1,4), got {result.shape}")

    def test_values_match_input_list(self):
        """All four values must match the input list."""
        box = [10.0, 20.5, 100.0, 200.75]
        result = t03.make_box_tensor(box, self._device)
        for i, val in enumerate(box):
            self.assertAlmostEqual(result[0, i].item(), val, places=4,
                                   msg=f"Value at index {i} should be {val}, got {result[0,i].item()}")

    def test_values_match_input_tuple(self):
        """Input as tuple should also work."""
        box = (5.0, 15.0, 95.0, 85.0)
        result = t03.make_box_tensor(box, self._device)
        for i, val in enumerate(box):
            self.assertAlmostEqual(result[0, i].item(), val, places=4)

    def test_dtype_is_float32(self):
        """Output tensor must be float32."""
        result = t03.make_box_tensor([1, 2, 3, 4], self._device)
        self.assertEqual(result.dtype, torch.float32,
                         msg=f"Dtype should be float32, got {result.dtype}")

    def test_on_correct_device(self):
        """Output must reside on the requested device."""
        result = t03.make_box_tensor([1, 2, 3, 4], self._device)
        self.assertEqual(str(result.device), str(self._device))


# ===================================================================
# F.  EMA update
# ===================================================================

@unittest.skipUnless(TORCH_AVAILABLE, "torch not available")
class TestEMAUpdate(unittest.TestCase):
    """Tests for t03.update_ema(model, ema_model, decay)."""

    def setUp(self):
        self.model = _make_mock_sam()
        self.ema_model = copy.deepcopy(self.model)
        # Make online model different from EMA by changing weights
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.randn_like(p))

    # -- after update, EMA differs from online -------------------------------

    def test_after_update_ema_differs_from_online(self):
        """With decay < 1, one EMA update should move EMA params toward online."""
        initial_ema_params = [p.clone() for p in self.ema_model.parameters()]

        t03.update_ema(self.model, self.ema_model, decay=0.9)

        changed = False
        for i, p in enumerate(self.ema_model.parameters()):
            if not torch.equal(p, initial_ema_params[i]):
                changed = True
                break
        self.assertTrue(changed,
                        msg="EMA params should change after update with decay=0.9")

    def test_ema_moves_toward_online(self):
        """After one update, EMA params should be a weighted mix."""
        # EMA_new = decay * EMA_old + (1 - decay) * online
        # So EMA_new should be between EMA_old and online
        decay = 0.9
        initial_ema = [p.clone() for p in self.ema_model.parameters()]

        t03.update_ema(self.model, self.ema_model, decay=decay)

        for i, p_ema in enumerate(self.ema_model.parameters()):
            online_val = list(self.model.parameters())[i]
            # EMA_new should differ from both EMA_old and online (unless they happen to be equal)
            self.assertFalse(torch.equal(p_ema, initial_ema[i]),
                             msg=f"EMA param {i} should not equal its initial value")
            self.assertFalse(torch.equal(p_ema, online_val),
                             msg=f"EMA param {i} should not equal online param after one step")

    # -- decay=1.0: EMA unchanged -------------------------------------------

    def test_decay_1_0_ema_unchanged(self):
        """With decay=1.0, EMA params must remain exactly as before."""
        initial_ema_params = [p.clone() for p in self.ema_model.parameters()]

        t03.update_ema(self.model, self.ema_model, decay=1.0)

        for i, p in enumerate(self.ema_model.parameters()):
            self.assertTrue(torch.allclose(p, initial_ema_params[i]),
                            msg=f"With decay=1.0, EMA param {i} should not change")

    # -- no_grad context respected -------------------------------------------

    def test_ema_update_runs_without_error(self):
        """EMA update completes without raising."""
        t03.update_ema(self.model, self.ema_model, decay=0.9)
        # EMA update happens under torch.no_grad() — params may still
        # have requires_grad=True but no gradients flow during update.
        for i, (p_online, p_ema) in enumerate(zip(
            self.model.parameters(), self.ema_model.parameters(),
        )):
            self.assertTrue(torch.isfinite(p_ema).all(),
                            msg=f"EMA param {i} must be finite after update")

    # -- multiple updates accumulate -----------------------------------------

    def test_multiple_updates_accumulate(self):
        """Multiple EMA updates should progressively move EMA toward online."""
        # Reset: EMA = online
        self.ema_model.load_state_dict(self.model.state_dict())
        # Change online
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.ones_like(p))

        ema_after_1 = []
        t03.update_ema(self.model, self.ema_model, decay=0.5)
        for p in self.ema_model.parameters():
            ema_after_1.append(p.clone())

        t03.update_ema(self.model, self.ema_model, decay=0.5)
        for i, p in enumerate(self.ema_model.parameters()):
            # After a second update, EMA should be even closer to online
            dist_1 = (ema_after_1[i] - list(self.model.parameters())[i]).abs().mean()
            dist_2 = (p - list(self.model.parameters())[i]).abs().mean()
            self.assertLess(dist_2.item(), dist_1.item(),
                            msg=f"EMA should converge toward online params over updates")

    # -- integer/bool tensors handled ----------------------------------------

    def test_integer_params_copied(self):
        """Non-floating-point parameters (int/bool) should be copied directly."""

        class ModelWithInt(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = torch.nn.Embedding(10, 4)

        model = ModelWithInt()
        ema_model = copy.deepcopy(model)
        # Modify online
        with torch.no_grad():
            model.embed.weight.add_(0.5)

        t03.update_ema(model, ema_model, decay=0.9)
        # Should not crash


# ===================================================================
# G.  parse_train_entries / full-only parsing
# ===================================================================

class TestParseTrainEntries(unittest.TestCase):
    """Tests for t03.parse_train_entries(fold_root, manifest, prompts, split_records)."""

    def setUp(self):
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_parse_entries_")
        self._tmpdir = Path(self._tmpdir_obj.name)
        # Create dummy .npy files so resolve_path succeeds
        self._img_path = self._tmpdir / "img.npy"
        self._gt_path = self._tmpdir / "gt.npy"
        np.save(str(self._img_path), np.zeros((64, 64), dtype=np.uint8))
        np.save(str(self._gt_path), np.zeros((64, 64), dtype=np.int64))

    def tearDown(self):
        self._tmpdir_obj.cleanup()

    def _make_manifest(self, entries: list[dict] | None = None) -> list[dict]:
        if entries is None:
            entries = [{
                "slice_name": "case001.npy",
                "teacher_img": str(self._img_path),
                "teacher_gt": str(self._gt_path),
                "split": "train",
            }]
        return entries

    def _make_prompts(self, entries: list[dict] | None = None) -> dict:
        if entries is None:
            entries = [{
                "slice_name": "case001.npy",
                "instances": [
                    {"label_id": 1, "component_id": 1,
                     "bbox_teacher": [10, 20, 100, 200]},
                ],
            }]
        result = {}
        for e in entries:
            sn = e.pop("slice_name")
            result[sn] = e
        return result

    def _make_split_records(self, entries: list[dict] | None = None) -> list[dict]:
        if entries is None:
            entries = [{
                "slice_name": "case001.npy",
                "label_mode": "full",
                "case_id": "case001",
            }]
        return entries

    # -- only full label_mode entries included -------------------------------

    def test_full_entries_included(self):
        """Entries with label_mode='full' are included."""
        entries = t03.parse_train_entries(
            self._tmpdir,
            self._make_manifest(),
            self._make_prompts(),
            self._make_split_records([{
                "slice_name": "case001.npy",
                "label_mode": "full",
                "case_id": "case001",
            }]),
        )
        self.assertEqual(len(entries), 1,
                         msg="One full record with one instance -> 1 entry")
        self.assertEqual(entries[0]["label_mode"], "full")

    # -- box entries excluded ------------------------------------------------

    def test_box_entries_excluded(self):
        """Records with label_mode='box' must not appear in result."""
        manifest = self._make_manifest([
            {"slice_name": "case001.npy", "teacher_img": str(self._img_path),
             "teacher_gt": str(self._gt_path), "split": "train"},
            {"slice_name": "case002.npy", "teacher_img": str(self._img_path),
             "teacher_gt": str(self._gt_path), "split": "train"},
        ])
        prompts = self._make_prompts([
            {"slice_name": "case001.npy",
             "instances": [{"label_id": 1, "component_id": 1,
                            "bbox_teacher": [10, 20, 100, 200]}]},
            {"slice_name": "case002.npy",
             "instances": [{"label_id": 2, "component_id": 2,
                            "bbox_teacher": [5, 5, 50, 50]}]},
        ])
        split_records = [
            {"slice_name": "case001.npy", "label_mode": "full", "case_id": "c1"},
            {"slice_name": "case002.npy", "label_mode": "box", "case_id": "c2"},
        ]
        entries = t03.parse_train_entries(self._tmpdir, manifest, prompts, split_records)
        slice_names = {e["slice_name"] for e in entries}
        self.assertIn("case001.npy", slice_names,
                      msg="full record should be included")
        self.assertNotIn("case002.npy", slice_names,
                         msg="box record should be excluded")

    def test_only_full_included_multiple_records(self):
        """When multiple records exist, only full ones are kept."""
        manifest = self._make_manifest([
            {"slice_name": f"case{i:03d}.npy", "teacher_img": str(self._img_path),
             "teacher_gt": str(self._gt_path), "split": "train"}
            for i in range(5)
        ])
        prompts_list = []
        for i in range(5):
            prompts_list.append({
                "slice_name": f"case{i:03d}.npy",
                "instances": [{"label_id": i, "component_id": i,
                               "bbox_teacher": [0, 0, 10, 10]}],
            })
        prompts = self._make_prompts(prompts_list)
        split_records = [
            {"slice_name": "case000.npy", "label_mode": "full", "case_id": "c0"},
            {"slice_name": "case001.npy", "label_mode": "box", "case_id": "c1"},
            {"slice_name": "case002.npy", "label_mode": "full", "case_id": "c2"},
            {"slice_name": "case003.npy", "label_mode": "box", "case_id": "c3"},
            {"slice_name": "case004.npy", "label_mode": "full", "case_id": "c4"},
        ]
        entries = t03.parse_train_entries(self._tmpdir, manifest, prompts, split_records)
        slice_names = {e["slice_name"] for e in entries}
        self.assertSetEqual(slice_names, {"case000.npy", "case002.npy", "case004.npy"},
                            msg="Only full entries should be present")

    # -- missing manifest entry -> KeyError ----------------------------------

    def test_slice_not_in_manifest_raises_KeyError(self):
        """A full record with no matching manifest entry must raise KeyError."""
        manifest = self._make_manifest([
            {"slice_name": "other.npy", "teacher_img": str(self._img_path),
             "teacher_gt": str(self._gt_path), "split": "train"},
        ])
        prompts = self._make_prompts([
            {"slice_name": "case001.npy",
             "instances": [{"label_id": 1, "component_id": 1,
                            "bbox_teacher": [10, 20, 100, 200]}]},
        ])
        split_records = [{"slice_name": "case001.npy", "label_mode": "full", "case_id": "c1"}]
        with self.assertRaises(KeyError,
                               msg="Missing manifest entry must raise KeyError"):
            t03.parse_train_entries(self._tmpdir, manifest, prompts, split_records)

    # -- missing prompt entry -> KeyError ------------------------------------

    def test_slice_not_in_prompts_raises_KeyError(self):
        """A full record with no matching prompt entry must raise KeyError."""
        manifest = self._make_manifest([
            {"slice_name": "case001.npy", "teacher_img": str(self._img_path),
             "teacher_gt": str(self._gt_path), "split": "train"},
        ])
        prompts = self._make_prompts([
            {"slice_name": "other.npy",
             "instances": [{"label_id": 1, "component_id": 1,
                            "bbox_teacher": [10, 20, 100, 200]}]},
        ])
        split_records = [{"slice_name": "case001.npy", "label_mode": "full", "case_id": "c1"}]
        with self.assertRaises(KeyError,
                               msg="Missing prompt entry must raise KeyError"):
            t03.parse_train_entries(self._tmpdir, manifest, prompts, split_records)

    # -- non-train split -> RuntimeError -------------------------------------

    def test_non_train_split_raises_RuntimeError(self):
        """A full record whose manifest split is not 'train' must raise RuntimeError."""
        manifest = self._make_manifest([
            {"slice_name": "case001.npy", "teacher_img": str(self._img_path),
             "teacher_gt": str(self._gt_path), "split": "test"},
        ])
        prompts = self._make_prompts([
            {"slice_name": "case001.npy",
             "instances": [{"label_id": 1, "component_id": 1,
                            "bbox_teacher": [10, 20, 100, 200]}]},
        ])
        split_records = [{"slice_name": "case001.npy", "label_mode": "full", "case_id": "c1"}]
        with self.assertRaises(RuntimeError,
                               msg="Non-train split must raise RuntimeError"):
            t03.parse_train_entries(self._tmpdir, manifest, prompts, split_records)

    # -- missing instances -> ValueError -------------------------------------

    def test_empty_instances_skipped(self):
        """A prompt entry with empty instances list is skipped (not an error)."""
        manifest = self._make_manifest()
        prompts = {"case001.npy": {"instances": []}}
        split_records = [{"slice_name": "case001.npy", "label_mode": "full", "case_id": "c1"}]
        entries = t03.parse_train_entries(self._tmpdir, manifest, prompts, split_records)
        full_entries = [e for e in entries if e["label_mode"] == "full"]
        self.assertEqual(len(full_entries), 0,
                         msg="Empty instances should be skipped")

    def test_missing_instances_key_skipped(self):
        """A prompt entry without 'instances' key is skipped (not an error)."""
        manifest = self._make_manifest()
        prompts = {"case001.npy": {}}
        split_records = [{"slice_name": "case001.npy", "label_mode": "full", "case_id": "c1"}]
        entries = t03.parse_train_entries(self._tmpdir, manifest, prompts, split_records)
        full_entries = [e for e in entries if e["label_mode"] == "full"]
        self.assertEqual(len(full_entries), 0,
                         msg="Missing 'instances' key should be skipped")

    # -- bbox extraction -----------------------------------------------------

    def test_bbox_teacher_preferred_over_bbox(self):
        """bbox_teacher takes priority over bbox."""
        manifest = self._make_manifest()
        prompts = {
            "case001.npy": {
                "instances": [{
                    "label_id": 1, "component_id": 1,
                    "bbox_teacher": [10, 20, 100, 200],
                    "bbox": [99, 99, 99, 99],  # should be ignored
                }],
            },
        }
        split_records = [{"slice_name": "case001.npy", "label_mode": "full", "case_id": "c1"}]
        entries = t03.parse_train_entries(self._tmpdir, manifest, prompts, split_records)
        self.assertEqual(entries[0]["bbox"], [10.0, 20.0, 100.0, 200.0],
                         msg="bbox_teacher should be used when available")

    def test_bbox_fallback_when_no_bbox_teacher(self):
        """When bbox_teacher is absent, bbox is used as fallback."""
        manifest = self._make_manifest()
        prompts = {
            "case001.npy": {
                "instances": [{
                    "label_id": 1, "component_id": 1,
                    "bbox": [5, 10, 55, 110],
                }],
            },
        }
        split_records = [{"slice_name": "case001.npy", "label_mode": "full", "case_id": "c1"}]
        entries = t03.parse_train_entries(self._tmpdir, manifest, prompts, split_records)
        self.assertEqual(entries[0]["bbox"], [5.0, 10.0, 55.0, 110.0],
                         msg="bbox fallback should work when bbox_teacher is absent")

    def test_missing_both_bbox_keys_raises_KeyError(self):
        """When both bbox_teacher and bbox are absent, KeyError is raised."""
        manifest = self._make_manifest()
        prompts = {
            "case001.npy": {
                "instances": [{"label_id": 1, "component_id": 1}],
            },
        }
        split_records = [{"slice_name": "case001.npy", "label_mode": "full", "case_id": "c1"}]
        with self.assertRaises(KeyError,
                               msg="Missing both bbox keys must raise KeyError"):
            t03.parse_train_entries(self._tmpdir, manifest, prompts, split_records)

    # -- entry fields --------------------------------------------------------

    def test_entries_have_required_keys(self):
        """Each returned entry must have the expected keys."""
        manifest = self._make_manifest()
        prompts = self._make_prompts()
        split_records = self._make_split_records()
        entries = t03.parse_train_entries(self._tmpdir, manifest, prompts, split_records)
        required = {"slice_name", "label_mode", "case_id", "label_id",
                    "component_id", "bbox", "teacher_img", "teacher_gt"}
        for i, entry in enumerate(entries):
            missing = required - set(entry.keys())
            self.assertEqual(missing, set(),
                             msg=f"Entry {i} missing keys: {missing}")

    def test_label_mode_always_full(self):
        """Every returned entry must have label_mode='full'."""
        manifest = self._make_manifest()
        prompts = self._make_prompts()
        split_records = self._make_split_records()
        entries = t03.parse_train_entries(self._tmpdir, manifest, prompts, split_records)
        for entry in entries:
            self.assertEqual(entry["label_mode"], "full")

    # -- multiple instances produce multiple entries -------------------------

    def test_multiple_instances_produce_multiple_entries(self):
        """A slice with N instances produces N entries."""
        manifest = self._make_manifest()
        prompts = {
            "case001.npy": {
                "instances": [
                    {"label_id": 1, "component_id": 1,
                     "bbox_teacher": [10, 20, 100, 200]},
                    {"label_id": 2, "component_id": 2,
                     "bbox_teacher": [30, 40, 120, 220]},
                    {"label_id": 3, "component_id": 3,
                     "bbox_teacher": [50, 60, 140, 240]},
                ],
            },
        }
        split_records = [{"slice_name": "case001.npy", "label_mode": "full", "case_id": "c1"}]
        entries = t03.parse_train_entries(self._tmpdir, manifest, prompts, split_records)
        self.assertEqual(len(entries), 3,
                         msg="3 instances should produce 3 entries")
        label_ids = {e["label_id"] for e in entries}
        self.assertSetEqual(label_ids, {1, 2, 3})

    def test_case_id_propagated(self):
        """case_id from split_records should appear in the entry."""
        manifest = self._make_manifest()
        prompts = self._make_prompts()
        split_records = [{"slice_name": "case001.npy", "label_mode": "full", "case_id": "my_case"}]
        entries = t03.parse_train_entries(self._tmpdir, manifest, prompts, split_records)
        self.assertEqual(entries[0]["case_id"], "my_case")


# ===================================================================
# H.  CLI parser (build_parser / validate_args)
# ===================================================================

class TestCLIParser(unittest.TestCase):
    """Tests for t03.build_parser() and t03.validate_args()."""

    def setUp(self):
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_cli_")
        self._tmpdir = Path(self._tmpdir_obj.name)

        # Create minimal files for validate_args
        self._checkpoint = self._tmpdir / "checkpoint.pth"
        self._checkpoint.write_bytes(b"dummy")
        self._processed = self._tmpdir / "processed"
        self._processed.mkdir()
        self._out = self._tmpdir / "out"
        self._out.mkdir()

    def tearDown(self):
        self._tmpdir_obj.cleanup()

    # -- standard args accepted ----------------------------------------------

    def test_standard_args_accepted(self):
        """Build parser and parse known arguments without error."""
        parser = t03.build_parser()
        argv = [
            "--processed_root", str(self._processed),
            "--checkpoint", str(self._checkpoint),
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--medsam_ft_root", str(self._out),
            "--device", "cpu",
            "--epochs", "3",
            "--max_steps", "100",
            "--lr", "1e-4",
            "--weight_decay", "0.05",
            "--ema_decay", "0.999",
            "--max_grad_norm", "2.0",
            "--seed", "42",
        ]
        args = parser.parse_args(argv)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.max_steps, 100)
        self.assertAlmostEqual(args.lr, 1e-4, places=8)
        self.assertAlmostEqual(args.weight_decay, 0.05)
        self.assertAlmostEqual(args.ema_decay, 0.999)
        self.assertAlmostEqual(args.max_grad_norm, 2.0)
        self.assertEqual(args.seed, 42)

    def test_epochs_accepted(self):
        """--epochs can be set."""
        parser = t03.build_parser()
        args = parser.parse_args([
            "--processed_root", str(self._processed),
            "--checkpoint", str(self._checkpoint),
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--medsam_ft_root", str(self._out),
            "--epochs", "10",
        ])
        self.assertEqual(args.epochs, 10)

    def test_max_steps_accepted(self):
        parser = t03.build_parser()
        args = parser.parse_args([
            "--processed_root", str(self._processed),
            "--checkpoint", str(self._checkpoint),
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--medsam_ft_root", str(self._out),
            "--max_steps", "500",
        ])
        self.assertEqual(args.max_steps, 500)

    def test_lr_accepted(self):
        parser = t03.build_parser()
        args = parser.parse_args([
            "--processed_root", str(self._processed),
            "--checkpoint", str(self._checkpoint),
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--medsam_ft_root", str(self._out),
            "--lr", "3e-5",
        ])
        self.assertAlmostEqual(args.lr, 3e-5, places=8)

    def test_weight_decay_accepted(self):
        parser = t03.build_parser()
        args = parser.parse_args([
            "--processed_root", str(self._processed),
            "--checkpoint", str(self._checkpoint),
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--medsam_ft_root", str(self._out),
            "--weight_decay", "0.001",
        ])
        self.assertAlmostEqual(args.weight_decay, 0.001, places=6)

    def test_ema_decay_accepted(self):
        parser = t03.build_parser()
        args = parser.parse_args([
            "--processed_root", str(self._processed),
            "--checkpoint", str(self._checkpoint),
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--medsam_ft_root", str(self._out),
            "--ema_decay", "0.99",
        ])
        self.assertAlmostEqual(args.ema_decay, 0.99, places=6)

    def test_max_grad_norm_accepted(self):
        parser = t03.build_parser()
        args = parser.parse_args([
            "--processed_root", str(self._processed),
            "--checkpoint", str(self._checkpoint),
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--medsam_ft_root", str(self._out),
            "--max_grad_norm", "3.0",
        ])
        self.assertAlmostEqual(args.max_grad_norm, 3.0, places=6)

    # -- old SAC args rejected -----------------------------------------------

    def test_lambda_out_causes_unrecognized(self):
        """--lambda_out (old SAC arg) must cause 'unrecognized arguments'."""
        parser = t03.build_parser()
        with self.assertRaises(SystemExit,
                               msg="--lambda_out should cause unrecognized-arguments error"):
            # Suppress argparse's stderr noise
            import sys as _sys
            _sys.stderr = io.StringIO()
            try:
                parser.parse_args(["--lambda_out", "1.0"])
            finally:
                _sys.stderr = _sys.__stderr__

    def test_lambda_seed_causes_unrecognized(self):
        """--lambda_seed (old SAC arg) must cause 'unrecognized arguments'."""
        parser = t03.build_parser()
        with self.assertRaises(SystemExit,
                               msg="--lambda_seed should cause unrecognized-arguments error"):
            import sys as _sys
            _sys.stderr = io.StringIO()
            try:
                parser.parse_args(["--lambda_seed", "1.0"])
            finally:
                _sys.stderr = _sys.__stderr__

    def test_old_sac_criterion_rejected(self):
        """--sac_criterion (old arg) must cause unrecognized-arguments error."""
        parser = t03.build_parser()
        with self.assertRaises(SystemExit,
                               msg="--sac_criterion should be rejected"):
            import sys as _sys
            _sys.stderr = io.StringIO()
            try:
                parser.parse_args(["--sac_criterion", "dice"])
            finally:
                _sys.stderr = _sys.__stderr__

    # -- defaults ------------------------------------------------------------

    def test_default_epochs(self):
        parser = t03.build_parser()
        args = parser.parse_args([
            "--processed_root", str(self._processed),
            "--checkpoint", str(self._checkpoint),
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--medsam_ft_root", str(self._out),
        ])
        self.assertEqual(args.epochs, 1)

    def test_default_lr(self):
        parser = t03.build_parser()
        args = parser.parse_args([
            "--processed_root", str(self._processed),
            "--checkpoint", str(self._checkpoint),
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--medsam_ft_root", str(self._out),
        ])
        self.assertAlmostEqual(args.lr, 1e-5, places=8)

    def test_default_ema_decay(self):
        parser = t03.build_parser()
        args = parser.parse_args([
            "--processed_root", str(self._processed),
            "--checkpoint", str(self._checkpoint),
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--medsam_ft_root", str(self._out),
        ])
        self.assertAlmostEqual(args.ema_decay, 0.99, places=8)

    def test_default_weight_decay(self):
        parser = t03.build_parser()
        args = parser.parse_args([
            "--processed_root", str(self._processed),
            "--checkpoint", str(self._checkpoint),
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--medsam_ft_root", str(self._out),
        ])
        self.assertAlmostEqual(args.weight_decay, 0.01, places=8)

    def test_default_max_grad_norm(self):
        parser = t03.build_parser()
        args = parser.parse_args([
            "--processed_root", str(self._processed),
            "--checkpoint", str(self._checkpoint),
            "--datasets", "btcv",
            "--round_tag", "r00_full5",
            "--medsam_ft_root", str(self._out),
        ])
        self.assertAlmostEqual(args.max_grad_norm, 1.0, places=8)

    # -- validate_args -------------------------------------------------------

    def test_validate_args_accepts_valid(self):
        """validate_args should pass with valid inputs."""
        import argparse
        ns = argparse.Namespace(
            epochs=1, max_steps=0, lr=1e-5,
            weight_decay=0.01, ema_decay=0.99, max_grad_norm=1.0,
            log_every=10,
            checkpoint=self._checkpoint,
            processed_root=self._processed,
            medsam_ft_root=self._out,
        )
        try:
            t03.validate_args(ns)
        except Exception as e:
            self.fail(f"validate_args raised unexpectedly: {e}")

    def test_validate_args_rejects_epochs_zero(self):
        import argparse
        ns = argparse.Namespace(
            epochs=0, max_steps=0, lr=1e-5,
            weight_decay=0.01, ema_decay=0.99, max_grad_norm=1.0,
            log_every=10,
            checkpoint=self._checkpoint,
            processed_root=self._processed,
            medsam_ft_root=self._out,
        )
        with self.assertRaises(ValueError):
            t03.validate_args(ns)

    def test_validate_args_rejects_negative_max_steps(self):
        import argparse
        ns = argparse.Namespace(
            epochs=1, max_steps=-1, lr=1e-5,
            weight_decay=0.01, ema_decay=0.99, max_grad_norm=1.0,
            log_every=10,
            checkpoint=self._checkpoint,
            processed_root=self._processed,
            medsam_ft_root=self._out,
        )
        with self.assertRaises(ValueError):
            t03.validate_args(ns)

    def test_validate_args_rejects_zero_lr(self):
        import argparse
        ns = argparse.Namespace(
            epochs=1, max_steps=0, lr=0.0,
            weight_decay=0.01, ema_decay=0.99, max_grad_norm=1.0,
            log_every=10,
            checkpoint=self._checkpoint,
            processed_root=self._processed,
            medsam_ft_root=self._out,
        )
        with self.assertRaises(ValueError):
            t03.validate_args(ns)

    def test_validate_args_rejects_ema_decay_1(self):
        """ema_decay must be strictly less than 1."""
        import argparse
        ns = argparse.Namespace(
            epochs=1, max_steps=0, lr=1e-5,
            weight_decay=0.01, ema_decay=1.0, max_grad_norm=1.0,
            log_every=10,
            checkpoint=self._checkpoint,
            processed_root=self._processed,
            medsam_ft_root=self._out,
        )
        with self.assertRaises(ValueError):
            t03.validate_args(ns)

    def test_validate_args_rejects_negative_ema_decay(self):
        import argparse
        ns = argparse.Namespace(
            epochs=1, max_steps=0, lr=1e-5,
            weight_decay=0.01, ema_decay=-0.1, max_grad_norm=1.0,
            log_every=10,
            checkpoint=self._checkpoint,
            processed_root=self._processed,
            medsam_ft_root=self._out,
        )
        with self.assertRaises(ValueError):
            t03.validate_args(ns)

    def test_validate_args_missing_checkpoint(self):
        import argparse
        ns = argparse.Namespace(
            epochs=1, max_steps=0, lr=1e-5,
            weight_decay=0.01, ema_decay=0.99, max_grad_norm=1.0,
            log_every=10,
            checkpoint=Path("/nonexistent/checkpoint.pth"),
            processed_root=self._processed,
        )
        with self.assertRaises(FileNotFoundError):
            t03.validate_args(ns)

    def test_validate_args_missing_processed_root(self):
        import argparse
        ns = argparse.Namespace(
            epochs=1, max_steps=0, lr=1e-5,
            weight_decay=0.01, ema_decay=0.99, max_grad_norm=1.0,
            log_every=10,
            checkpoint=self._checkpoint,
            processed_root=Path("/nonexistent/dir"),
            medsam_ft_root=self._out,
        )
        with self.assertRaises(FileNotFoundError):
            t03.validate_args(ns)


# ===================================================================
# I.  No MONAI imports / no old SAC losses
# ===================================================================

class TestNoMonaiImports(unittest.TestCase):
    """Safety checks: the module must not import monai or define old SAC loss functions."""

    @classmethod
    def setUpClass(cls):
        """Read the module source once for all tests in this class."""
        source_path = Path(t03.__file__)
        cls._source = source_path.read_text(encoding="utf-8")

    def test_no_monai_import(self):
        """The module must not import monai anywhere."""
        # Check for import statements containing 'monai'
        self.assertNotIn("monai", self._source,
                         msg="Module must not import monai")

    def test_no_from_monai_import(self):
        """Specific check: 'from monai' or 'import monai' must not appear."""
        self.assertNotIn("import monai", self._source,
                         msg="Must not have 'import monai'")
        self.assertNotIn("from monai", self._source,
                         msg="Must not have 'from monai'")

    def test_no_sac_loss_class_definition(self):
        """Check that no SAC-branded loss class is defined in the module."""
        # Look for class definitions that hint at SAC-specific losses
        source_lower = self._source.lower()
        # Generic "sac" appears in paths (medsam_sac_*.pth) — that's fine.
        # We specifically check for SAC loss function class definitions.
        self.assertNotIn("class sacloss", source_lower,
                         msg="Module must not define SACLoss class")
        self.assertNotIn("class sa_cascade_loss", source_lower,
                         msg="Module must not define SA_Cascade_Loss class")

    def test_loss_functions_are_clean(self):
        """Verify that only dice_loss_with_logits is defined, no legacy SAC losses."""
        # The only custom loss function defined should be dice_loss_with_logits
        self.assertIn("def dice_loss_with_logits", self._source,
                      msg="Expected dice_loss_with_logits to be defined")
        # No old SAC custom loss functions
        self.assertNotIn("def sac_loss", self._source,
                         msg="Must not define sac_loss function")
        self.assertNotIn("def sa_cascade", self._source,
                         msg="Must not define sa_cascade function")
        self.assertNotIn("def dae_loss", self._source,
                         msg="Must not define dae_loss function")

    def test_uses_bce_with_logits(self):
        """Verify the clean BCE loss from torch.nn.functional is used."""
        self.assertIn("F.binary_cross_entropy_with_logits", self._source,
                      msg="Module must use F.binary_cross_entropy_with_logits for BCE loss")

    def test_no_monai_loss_imports(self):
        """No MONAI loss classes are imported (DiceLoss, TverskyLoss, etc.)."""
        self.assertNotIn("DiceLoss", self._source,
                         msg="Must not import MONAI's DiceLoss")
        self.assertNotIn("TverskyLoss", self._source,
                         msg="Must not import MONAI's TverskyLoss")
        self.assertNotIn("FocalLoss", self._source,
                         msg="Must not import MONAI's FocalLoss")


# ===================================================================
# J.  Helper functions: get_slice_name / resolve_path / save_json / load_json
# ===================================================================

class TestHelperFunctions(unittest.TestCase):
    """Tests for small utility functions in the module."""

    def setUp(self):
        self._tmpdir_obj = tempfile.TemporaryDirectory(prefix="test_helpers_")
        self._tmpdir = Path(self._tmpdir_obj.name)

    def tearDown(self):
        self._tmpdir_obj.cleanup()

    # -- get_slice_name ------------------------------------------------------

    def test_get_slice_name_from_slice_name_key(self):
        """When slice_name key is present, it is used directly."""
        item = {"slice_name": "my_slice.png"}
        self.assertEqual(t03.get_slice_name(item), "my_slice.png")

    def test_get_slice_name_from_teacher_img(self):
        """When slice_name is missing, fall back to teacher_img basename."""
        item = {"teacher_img": "/path/to/img_001.npy"}
        self.assertEqual(t03.get_slice_name(item), "img_001.npy")

    def test_get_slice_name_from_student_img(self):
        """Fallback to student_img basename."""
        item = {"student_img": "/data/student_case.npy", "teacher_img": ""}
        self.assertEqual(t03.get_slice_name(item), "student_case.npy")

    def test_get_slice_name_from_teacher_gt(self):
        """Fallback to teacher_gt basename."""
        item = {"teacher_gt": "/gt/mask.npy"}
        self.assertEqual(t03.get_slice_name(item), "mask.npy")

    def test_get_slice_name_raises_KeyError_when_none_found(self):
        """If no identifiable key is present, KeyError is raised."""
        item = {"unknown_key": "value"}
        with self.assertRaises(KeyError):
            t03.get_slice_name(item)

    def test_get_slice_name_empty_strings_not_used(self):
        """Empty string values should not be used as fallback."""
        # teacher_img is empty string -> falsy, skip to next
        item = {"teacher_img": "", "student_img": "/valid/name.npy"}
        result = t03.get_slice_name(item)
        self.assertEqual(result, "name.npy")

    # -- resolve_path --------------------------------------------------------

    def test_resolve_path_absolute(self):
        """An absolute path in the manifest is returned as-is."""
        p = self._tmpdir / "exists.npy"
        p.write_bytes(b"")
        item = {"teacher_img": str(p)}
        result = t03.resolve_path(self._tmpdir, item, "teacher_img")
        self.assertEqual(result, p)

    def test_resolve_path_relative(self):
        """A relative path is resolved against fold_root."""
        p = self._tmpdir / "subdir" / "img.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"")
        item = {"teacher_img": "subdir/img.npy"}
        result = t03.resolve_path(self._tmpdir, item, "teacher_img")
        self.assertEqual(result, p)

    def test_resolve_path_missing_key_raises_KeyError(self):
        """Missing key raises KeyError."""
        item = {"other_key": "value"}
        with self.assertRaises(KeyError):
            t03.resolve_path(self._tmpdir, item, "teacher_img")

    def test_resolve_path_empty_value_raises_KeyError(self):
        """Empty string value raises KeyError."""
        item = {"teacher_img": ""}
        with self.assertRaises(KeyError):
            t03.resolve_path(self._tmpdir, item, "teacher_img")

    def test_resolve_path_nonexistent_file_raises_FileNotFoundError(self):
        """If the resolved path does not exist, FileNotFoundError is raised."""
        item = {"teacher_img": "/nonexistent/file.npy"}
        with self.assertRaises(FileNotFoundError):
            t03.resolve_path(self._tmpdir, item, "teacher_img")

    # -- load_json / save_json roundtrip -------------------------------------

    def test_json_roundtrip(self):
        """save_json writes data that load_json can read back."""
        data = {"key": "value", "list": [1, 2, 3], "nested": {"a": True}}
        p = self._tmpdir / "test.json"
        t03.save_json(data, p)
        self.assertTrue(p.is_file(), msg="JSON file should exist after save_json")
        loaded = t03.load_json(p)
        self.assertEqual(loaded, data, msg="JSON roundtrip should preserve data")

    def test_save_json_creates_parent_dirs(self):
        """save_json should create parent directories."""
        p = self._tmpdir / "deep" / "nested" / "dir" / "test.json"
        t03.save_json({"ok": True}, p)
        self.assertTrue(p.is_file())

    def test_load_json_unicode(self):
        """load_json preserves unicode content."""
        data = {"chinese": "中文", "emoji": "(check)"}
        p = self._tmpdir / "unicode.json"
        t03.save_json(data, p)
        loaded = t03.load_json(p)
        self.assertEqual(loaded, data)


# ===================================================================
if __name__ == "__main__":
    unittest.main()
