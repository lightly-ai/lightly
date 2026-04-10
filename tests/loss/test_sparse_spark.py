import pytest
import torch
import torch.distributed as dist
from pytest_mock import MockerFixture

pytest.importorskip("timm.models.layers")

from lightly.loss import SparKPatchReconLoss


class TestSparKPatchReconLoss:
    def test__gather_distributed(self, mocker: MockerFixture) -> None:
        mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
        SparKPatchReconLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError):
            SparKPatchReconLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test_forward_pass(self) -> None:
        loss = SparKPatchReconLoss()
        inp_patches = torch.randn((2, 4, 3))
        recon_patches = torch.randn((2, 4, 3))
        mask = torch.randn((2, 1, 2, 2)) > 0

        loss(inp_patches, recon_patches, mask)

    def test_forward_pass_deterministic(self) -> None:
        loss = SparKPatchReconLoss()
        inp_patches = torch.randn((2, 4, 3))
        recon_patches = torch.randn((2, 4, 3))
        mask = torch.randn((2, 1, 2, 2)) > 0

        loss1, _, _ = loss(inp_patches, recon_patches, mask)
        loss2, _, _ = loss(inp_patches, recon_patches, mask)

        assert loss1.item() == pytest.approx(loss2.item())

    def test_forward__compare(self) -> None:
        loss = SparKPatchReconLoss()
        inp_patches = torch.randn((2, 4, 3))
        recon_patches = torch.randn((2, 4, 3))

        mask = torch.randn((2, 1, 2, 2)) > 0

        loss1 = loss(inp_patches, recon_patches, mask)[0]
        loss2 = _reference_loss_implementation(inp_patches, recon_patches, mask)

        assert loss1.item() == pytest.approx(loss2.item(), rel=1e-5)


def _reference_loss_implementation(
    inp: torch.Tensor,
    rec: torch.Tensor,
    active_b1ff: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    This loss implementation was taken from the official implementation of the
    SparK paper, and serves as a reference to ensure that the
    SparKPatchReconLoss implementation in lightly produces the same results.

    https://github.com/keyu-tian/SparK/blob/a63e386f8e5186bc07ad7fce86e06b08f48a61ea/pretrain/spark.py#L112-L120
    """
    mean = inp.mean(dim=-1, keepdim=True)
    var = (inp.var(dim=-1, keepdim=True) + eps) ** 0.5
    inp = (inp - mean) / var
    l2_loss = ((rec - inp) ** 2).mean(
        dim=2, keepdim=False
    )  # (B, L, C) ==mean==> (B, L)

    non_active = (
        active_b1ff.logical_not().int().view(active_b1ff.shape[0], -1)
    )  # (B, 1, f, f) => (B, L)
    recon_loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + eps)
    return recon_loss  # type: ignore[no-any-return]
