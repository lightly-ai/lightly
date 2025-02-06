import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from lightly.loss import MACLLoss


def _cal_macl_loss_original(
    pos: Tensor, neg: Tensor, tau_0: float, alpha: float, A0: float
) -> Tensor:
    """The original implementation of the MACL loss.

    See: https://github.com/chenhaoxing/MACL/blob/main/macl.py
    Note that the original implementation is not numerically stable, so the error bar here is 1e-6.

    Args:
        pos:
            The positive cosine similarities.
        neg:
            The negative cosine similarities.
        tau_0:
            The initial temperature. Must be greater than 0.
        alpha:
            The alpha parameter. Must be in the range [0, 1].
        A0:
            The initial value of A. Must be in the range [0, 1].

    Returns:
        The loss value.
    """

    A = torch.mean(pos.detach())
    tau = tau_0 * (1.0 + alpha * (A - A0))

    logits = torch.cat([pos, neg], dim=1) / tau
    P = torch.softmax(logits, dim=1)[:, 0]

    # reweighting and loss computation
    V = 1.0 / (1.0 - P)
    loss: Tensor = -V.detach() * torch.log(P)

    return loss.mean()


class TestMACLLoss:
    @pytest.mark.parametrize("temperature", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("alpha", [0, 0.5, 1])
    @pytest.mark.parametrize("A_0", [0, 0.5, 1])
    @pytest.mark.parametrize("emb_dim", [2, 4, 8])
    def test_with_batch_size_2(
        self, temperature: float, alpha: float, A_0: float, emb_dim: int
    ) -> None:
        # Generate two random embeddings with batch size 2
        z0 = torch.tensor(np.random.rand(2, emb_dim))
        z1 = torch.tensor(np.random.rand(2, emb_dim))

        # Calculate the loss in both directions
        loss_function = MACLLoss(temperature=temperature, alpha=alpha, A_0=A_0)
        l1 = float(loss_function(z0, z1))
        l2 = float(loss_function(z1, z0))

        # Calculate the positive and negative cosine similarities
        z0_norm = F.normalize(z0, dim=-1, p=2)
        z1_norm = F.normalize(z1, dim=-1, p=2)

        pos = torch.tensor(
            [
                z0_norm[0] @ z1_norm[0],
                z0_norm[1] @ z1_norm[1],
                z0_norm[0] @ z1_norm[0],
                z0_norm[1] @ z1_norm[1],
            ]
        ).unsqueeze(1)
        neg = torch.stack(
            [
                torch.tensor([z0_norm[0] @ z0_norm[1], z0_norm[0] @ z1_norm[1]]),
                torch.tensor([z0_norm[1] @ z0_norm[0], z0_norm[1] @ z1_norm[0]]),
                torch.tensor([z1_norm[0] @ z0_norm[1], z1_norm[0] @ z1_norm[1]]),
                torch.tensor([z1_norm[1] @ z0_norm[0], z1_norm[1] @ z1_norm[0]]),
            ],
            dim=0,
        )

        # Calculate the loss using the original implementation
        l_original = _cal_macl_loss_original(pos, neg, temperature, alpha, A_0)

        assert l1 == pytest.approx(l2, abs=1e-6)
        assert l1 == pytest.approx(l_original, abs=1e-6)
        assert l2 == pytest.approx(l_original, abs=1e-6)

    def test_forward_pass(self) -> None:
        loss = MACLLoss()
        for bsz in range(2, 20):  # batch_size 1 is not allowed
            batch_1 = torch.randn((bsz, 32))
            batch_2 = torch.randn((bsz, 32))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            assert (l1 - l2).pow(2).item() == pytest.approx(0.0, abs=1e-10)

    def test_forward_pass_1d(self) -> None:
        loss = MACLLoss()
        for bsz in range(2, 20):  # # batch_size 1 is not allowed
            batch_1 = torch.randn((bsz, 1))
            batch_2 = torch.randn((bsz, 1))

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            assert (l1 - l2).pow(2).item() == pytest.approx(0.0, abs=1e-10)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_forward_pass_cuda(self) -> None:
        loss = MACLLoss()
        for bsz in range(2, 20):  # batch_size 1 is not allowed
            batch_1 = torch.randn((bsz, 32)).cuda()
            batch_2 = torch.randn((bsz, 32)).cuda()

            # symmetry
            l1 = loss(batch_1, batch_2)
            l2 = loss(batch_2, batch_1)
            assert (l1 - l2).pow(2).item() == pytest.approx(0.0, abs=1e-10)
