import unittest

import torch

from lightly.loss import MaskedImageModelingLoss


class TestMaskedImageModelingLoss(unittest.TestCase):
    def test_forward_pass(self) -> None:
        loss_fn = MaskedImageModelingLoss()
        for n_masked in [1, 40, 160]:
            logits = torch.randn(n_masked, 8192)
            targets = torch.randint(0, 8192, (n_masked,))
            loss = loss_fn(logits, targets)
            self.assertEqual(loss.shape, torch.Size([]))
            self.assertTrue(torch.isfinite(loss))

    def test_reduction_mean(self) -> None:
        loss_fn = MaskedImageModelingLoss(reduction="mean")
        logits = torch.randn(80, 8192)
        targets = torch.randint(0, 8192, (80,))
        loss = loss_fn(logits, targets)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_reduction_sum(self) -> None:
        loss_mean = MaskedImageModelingLoss(reduction="mean")
        loss_sum = MaskedImageModelingLoss(reduction="sum")
        logits = torch.randn(80, 8192)
        targets = torch.randint(0, 8192, (80,))
        self.assertAlmostEqual(
            loss_sum(logits, targets).item(),
            loss_mean(logits, targets).item() * 80,
            places=4,
        )

    def test_invalid_reduction_raises(self) -> None:
        with self.assertRaises(ValueError):
            MaskedImageModelingLoss(reduction="none")

    def test_invalid_label_smoothing_raises(self) -> None:
        with self.assertRaises(ValueError):
            MaskedImageModelingLoss(label_smoothing=-0.1)
        with self.assertRaises(ValueError):
            MaskedImageModelingLoss(label_smoothing=1.0)

    def test_shape_mismatch_raises(self) -> None:
        loss_fn = MaskedImageModelingLoss()
        with self.assertRaises(ValueError):
            loss_fn(torch.randn(10, 8192), torch.randint(0, 8192, (9,)))

    def test_label_smoothing(self) -> None:
        loss_plain = MaskedImageModelingLoss(label_smoothing=0.0)
        loss_smooth = MaskedImageModelingLoss(label_smoothing=0.1)
        torch.manual_seed(0)
        logits = torch.randn(80, 8192)
        targets = torch.randint(0, 8192, (80,))
        self.assertNotAlmostEqual(
            loss_plain(logits, targets).item(),
            loss_smooth(logits, targets).item(),
            places=3,
        )

    def test_gradient_flows(self) -> None:
        loss_fn = MaskedImageModelingLoss()
        logits = torch.randn(40, 8192, requires_grad=True)
        targets = torch.randint(0, 8192, (40,))
        loss = loss_fn(logits, targets)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertEqual(logits.grad.shape, logits.shape)

    def test_perfect_prediction_gives_low_loss(self) -> None:
        loss_fn = MaskedImageModelingLoss()
        n, v = 40, 8192
        targets = torch.randint(0, v, (n,))
        logits = torch.full((n, v), -100.0)
        logits[torch.arange(n), targets] = 100.0
        loss = loss_fn(logits, targets)
        self.assertLess(loss.item(), 0.01)

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available")
    def test_forward_pass_cuda(self) -> None:
        loss_fn = MaskedImageModelingLoss()
        logits = torch.randn(80, 8192).cuda()
        targets = torch.randint(0, 8192, (80,)).cuda()
        loss = loss_fn(logits, targets)
        self.assertEqual(loss.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(loss))
