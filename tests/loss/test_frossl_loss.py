import pytest
import torch

from lightly.loss import FroSSLLoss


class TestFroSSLLoss:
    def test_forward(self) -> None:
        loss_fn = FroSSLLoss()
        for bsz in range(2, 4):
            z0 = torch.randn((bsz, 32))
            z1 = torch.randn((bsz, 32))
            loss = loss_fn([z0, z1])
            assert loss.dim() == 0

    def test_forward__symmetry(self) -> None:
        loss_fn = FroSSLLoss()
        z0 = torch.randn((8, 32))
        z1 = torch.randn((8, 32))
        l1 = loss_fn([z0, z1])
        l2 = loss_fn([z1, z0])
        assert l1.item() == pytest.approx(l2.item())

    def test_forward__multiview(self) -> None:
        loss_fn = FroSSLLoss()
        z_views = [torch.randn((8, 32)) for _ in range(4)]
        loss = loss_fn(z_views)
        assert loss.dim() == 0

    def test_forward__wide(self) -> None:
        # dim > batch_size exercises the N < D branch (gram matrix).
        loss_fn = FroSSLLoss()
        z_views = [torch.randn((4, 64)) for _ in range(2)]
        loss = loss_fn(z_views)
        assert loss.dim() == 0

    def test_forward__backward(self) -> None:
        loss_fn = FroSSLLoss()
        z0 = torch.randn((8, 32), requires_grad=True)
        z1 = torch.randn((8, 32), requires_grad=True)
        loss = loss_fn([z0, z1])
        loss.backward()
        assert z0.grad is not None
        assert z1.grad is not None

    def test_forward__invariance_weight(self) -> None:
        z0 = torch.randn((8, 32))
        z1 = torch.randn((8, 32))
        default = FroSSLLoss()([z0, z1])
        no_invariance = FroSSLLoss(invariance_weight=0.0)([z0, z1])
        assert default.item() != pytest.approx(no_invariance.item())

    def test_forward__error_single_view(self) -> None:
        loss_fn = FroSSLLoss()
        with pytest.raises(ValueError):
            loss_fn([torch.randn((8, 32))])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
    def test_forward_cuda(self) -> None:
        loss_fn = FroSSLLoss().cuda()
        z0 = torch.randn((8, 32)).cuda()
        z1 = torch.randn((8, 32)).cuda()
        loss = loss_fn([z0, z1])
        assert loss.dim() == 0
