from __future__ import annotations

import pytest
import torch

from lightly.loss import DINOLoss
from lightly.models.modules.heads import (
    BarlowTwinsProjectionHead,
    BYOLPredictionHead,
    BYOLProjectionHead,
    CAPIProjectionHead,
    DenseCLProjectionHead,
    DINOProjectionHead,
    DINOv2ProjectionHead,
    FrancaProjectionHead,
    LeJEPAProjectionHead,
    MMCRProjectionHead,
    MoCoProjectionHead,
    MSNProjectionHead,
    NNCLRPredictionHead,
    NNCLRProjectionHead,
    SimCLRProjectionHead,
    SimSiamPredictionHead,
    SimSiamProjectionHead,
    SwaVProjectionHead,
    SwaVPrototypes,
    TiCoProjectionHead,
    VicRegLLocalProjectionHead,
)


class TestProjectionHeads:
    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        self.n_features = [
            (8, 16, 32),
            (8, 32, 16),
            (16, 8, 32),
            (16, 32, 8),
            (32, 8, 16),
            (32, 16, 8),
        ]
        self.swavProtoypes = [(8, 16, [32, 64, 128])]
        self.heads = [
            BarlowTwinsProjectionHead,
            BYOLProjectionHead,
            BYOLPredictionHead,
            DenseCLProjectionHead,
            DINOProjectionHead,
            LeJEPAProjectionHead,
            MoCoProjectionHead,
            MSNProjectionHead,
            MMCRProjectionHead,
            NNCLRProjectionHead,
            NNCLRPredictionHead,
            SimCLRProjectionHead,
            SimSiamProjectionHead,
            SimSiamPredictionHead,
            SwaVProjectionHead,
            TiCoProjectionHead,
            VicRegLLocalProjectionHead,
        ]

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_single_projection_head(self, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        seed = 0
        for head_cls in self.heads:
            for in_features, hidden_features, out_features in self.n_features:
                torch.manual_seed(seed)
                if head_cls == DINOProjectionHead:
                    bottleneck_features = hidden_features
                    head = head_cls(
                        in_features, hidden_features, bottleneck_features, out_features
                    )
                elif head_cls == SimCLRProjectionHead:
                    head = head_cls(
                        in_features, hidden_features, out_features, batch_norm=False
                    )
                else:
                    head = head_cls(in_features, hidden_features, out_features)
                head = head.eval()
                head = head.to(device)
                for batch_size in [1, 2]:
                    x = torch.torch.rand((batch_size, in_features)).to(device)
                    with torch.no_grad():
                        y = head(x)
                    assert y.shape[0] == batch_size
                    assert y.shape[1] == out_features

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_swav_prototypes(self, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        seed = 0
        for in_features, _, n_prototypes in self.n_features:
            torch.manual_seed(seed)
            prototypes = SwaVPrototypes(in_features, n_prototypes)
            prototypes = prototypes.eval()
            prototypes = prototypes.to(device)
            for batch_size in [1, 2]:
                x = torch.torch.rand((batch_size, in_features)).to(device)
                with torch.no_grad():
                    y = prototypes(x)
                assert y.shape[0] == batch_size
                assert y.shape[1] == n_prototypes

    def test_swav_frozen_prototypes(self) -> None:
        seed = 0
        criterion = torch.nn.L1Loss()
        linear_layer = torch.nn.Linear(8, 8, bias=False)
        prototypes = SwaVPrototypes(
            input_dim=8, n_prototypes=8, n_steps_frozen_prototypes=2
        )
        optimizer = torch.optim.SGD(prototypes.parameters(), lr=0.01)
        torch.manual_seed(seed)
        in_features = torch.rand(4, 8, device="cpu")
        target_features = torch.ones(4, 8, device="cpu")
        for step in range(4):
            out_features = linear_layer(in_features)
            out_features = prototypes.forward(out_features, step)
            loss = criterion(out_features, target_features)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step == 0:
                loss0 = loss
            if step <= 2:
                assert loss == loss0
            if step > 2:
                assert loss != loss0

    def test_swav_multi_prototypes(self) -> None:
        device = "cpu"
        seed = 0
        for in_features, _, n_prototypes in self.swavProtoypes:
            torch.manual_seed(seed)
            prototypes = SwaVPrototypes(in_features, n_prototypes)
            prototypes = prototypes.eval()
            prototypes = prototypes.to(device)
            for batch_size in [1, 2]:
                x = torch.torch.rand((batch_size, in_features)).to(device)
                with torch.no_grad():
                    y = prototypes(x)
                for layerNum, prototypeSize in enumerate(n_prototypes):
                    assert y[layerNum].shape[0] == batch_size
                    assert y[layerNum].shape[1] == prototypeSize

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_dino_projection_head(self, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        seed = 0
        input_dim, hidden_dim, output_dim = self.n_features[0]
        for bottleneck_dim in [8, 16, 32]:
            for batch_norm in [False, True]:
                torch.manual_seed(seed)
                head = DINOProjectionHead(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    bottleneck_dim=bottleneck_dim,
                    batch_norm=batch_norm,
                )
                head = head.eval()
                head = head.to(device)
                for batch_size in [1, 2]:
                    x = torch.torch.rand((batch_size, input_dim)).to(device)
                    with torch.no_grad():
                        y = head(x)
                    assert y.shape[0] == batch_size
                    assert y.shape[1] == output_dim

    def test_dino_projection_head_freeze_last_layer(self) -> None:
        """Test if freeze last layer cancels backprop."""
        seed = 0
        torch.manual_seed(seed)
        for norm_last_layer in [False, True]:
            for freeze_last_layer in range(-1, 3):
                head = DINOProjectionHead(
                    input_dim=4,
                    hidden_dim=4,
                    output_dim=4,
                    bottleneck_dim=4,
                    freeze_last_layer=freeze_last_layer,
                    norm_last_layer=norm_last_layer,
                )
                optimizer = torch.optim.SGD(head.parameters(), lr=5)
                criterion = DINOLoss(output_dim=4)
                # Store initial weights of last layer
                initial_data = [
                    param.data.detach().clone()
                    for param in head.last_layer.parameters()
                ]
                for epoch in range(5):
                    views = [torch.rand((3, 4)) for _ in range(2)]
                    teacher_out = [head(view) for view in views]
                    student_out = [head(view) for view in views]
                    loss = criterion(teacher_out, student_out)
                    optimizer.zero_grad()
                    loss.backward()
                    head.cancel_last_layer_gradients(current_epoch=epoch)
                    optimizer.step()
                    params = head.last_layer.parameters()
                    # Verify that weights have (not) changed depending on epoch.
                    for param, init_data in zip(params, initial_data):
                        if param.requires_grad:
                            are_same = torch.allclose(param.data, init_data)
                            if epoch >= freeze_last_layer:
                                assert not are_same
                            else:
                                assert are_same

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_dinov2_projection_head(self, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        seed = 0
        input_dim, hidden_dim, output_dim = self.n_features[0]
        for bottleneck_dim in [8, 16, 32]:
            for batch_norm in [False, True]:
                torch.manual_seed(seed)
                head = DINOv2ProjectionHead(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    output_dim=output_dim,
                    bottleneck_dim=bottleneck_dim,
                    batch_norm=batch_norm,
                )
                head = head.eval()
                head = head.to(device)
                for batch_size in [1, 2]:
                    x = torch.torch.rand((batch_size, input_dim)).to(device)
                    with torch.no_grad():
                        y = head(x)
                    assert y.shape[0] == batch_size
                    assert y.shape[1] == output_dim

    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_franca_projection_head(self, device: str) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        seed = 0
        input_dim, hidden_dim = 32, 16
        nesting_dims = [8, 16, 32]
        output_dim = 24
        last_dim = nesting_dims[-1]
        for bottleneck_dim in [8, 16]:
            for batch_norm in [False, True]:
                torch.manual_seed(seed)
                head = FrancaProjectionHead(
                    input_dim=input_dim,
                    nesting_dims=nesting_dims,
                    hidden_dim=hidden_dim,
                    bottleneck_dim=bottleneck_dim,
                    output_dim=output_dim,
                    batch_norm=batch_norm,
                )
                head = head.eval()
                head = head.to(device)
                for batch_size in [1, 2]:
                    x = torch.rand((batch_size, input_dim)).to(device)
                    with torch.no_grad():
                        outputs = head(x)
                    assert isinstance(outputs, tuple)
                    assert len(outputs) == len(nesting_dims)
                    expected_dims = [
                        int(output_dim * dim / last_dim) for dim in nesting_dims
                    ]
                    assert head.output_dims == expected_dims
                    for out, expected in zip(outputs, expected_dims):
                        assert out.shape[0] == batch_size
                        assert out.shape[1] == expected

    def test_franca_projection_head_backward(self) -> None:
        torch.manual_seed(0)
        head = FrancaProjectionHead(
            input_dim=32,
            nesting_dims=[8, 16, 32],
            hidden_dim=16,
            bottleneck_dim=8,
            output_dim=24,
        )
        x = torch.rand((4, 32), requires_grad=True)
        outputs = head(x)
        loss = sum(out.sum() for out in outputs)
        loss.backward()
        assert x.grad is not None
        for param in head.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_franca_projection_head_uses_only_nested_prefix(self) -> None:
        """Each level's output depends only on the prefix ``x[..., :nesting_dim]``."""
        torch.manual_seed(0)
        head = FrancaProjectionHead(
            input_dim=32,
            nesting_dims=[8, 16, 32],
            hidden_dim=16,
            bottleneck_dim=8,
            output_dim=24,
        )
        head = head.eval()
        x = torch.rand((2, 32))
        x_perturbed = x.clone()
        # Change only the tail beyond the smallest nesting dim (8).
        x_perturbed[..., 8:] += 1.0
        with torch.no_grad():
            out = head(x)
            out_perturbed = head(x_perturbed)
        # The smallest level reads only x[..., :8], so it must be unchanged.
        assert torch.equal(out[0], out_perturbed[0])
        # A larger level reads a longer prefix that the perturbation touched, so it changes.
        assert not torch.equal(out[-1], out_perturbed[-1])

    @pytest.mark.parametrize(
        "nesting_dims",
        [[], [16, 16], [32, 16], [-1, 16], [16, 64]],
        ids=["empty", "not-increasing", "decreasing", "non-positive", "exceeds-input"],
    )
    def test_franca_projection_head_invalid_nesting(
        self, nesting_dims: list[int]
    ) -> None:
        with pytest.raises(ValueError):
            FrancaProjectionHead(input_dim=32, nesting_dims=nesting_dims)

    def test_franca_projection_head_output_dim_too_small(self) -> None:
        """An output_dim that rounds a level to zero prototypes is rejected."""
        with pytest.raises(ValueError):
            # Level 8/32 would give int(3 * 8 / 32) = 0 prototypes.
            FrancaProjectionHead(input_dim=32, nesting_dims=[8, 32], output_dim=3)

    def test_simclr_projection_head_multiple_layers(self) -> None:
        device = "cpu"
        seed = 0
        for in_features, hidden_features, out_features in self.n_features:
            for num_layers in range(2, 5):
                for batch_norm in [True, False]:
                    torch.manual_seed(seed)
                    head = SimCLRProjectionHead(
                        in_features,
                        hidden_features,
                        out_features,
                        num_layers,
                        batch_norm,
                    )
                    head = head.eval()
                    head = head.to(device)
                    for batch_size in [1, 2]:
                        x = torch.torch.rand((batch_size, in_features)).to(device)
                        with torch.no_grad():
                            y = head(x)
                        assert y.shape[0] == batch_size
                        assert y.shape[1] == out_features

    def test_moco_projection_head_multiple_layers(self) -> None:
        device = "cpu"
        seed = 0
        for in_features, hidden_features, out_features in self.n_features:
            for num_layers in range(2, 5):
                for batch_norm in [True, False]:
                    torch.manual_seed(seed)
                    head = MoCoProjectionHead(
                        in_features,
                        hidden_features,
                        out_features,
                        num_layers,
                        batch_norm,
                    )
                    head = head.eval()
                    head = head.to(device)
                    for batch_size in [1, 2]:
                        x = torch.torch.rand((batch_size, in_features)).to(device)
                        with torch.no_grad():
                            y = head(x)
                        assert y.shape[0] == batch_size
                        assert y.shape[1] == out_features

    def test_lejepa_projection_head_backward(self) -> None:
        device = "cpu"
        seed = 0
        # batch_size must be at least 2
        # because the head uses BatchNorm1d in training mode.
        batch_size = 2
        for in_features, hidden_features, out_features in self.n_features:
            torch.manual_seed(seed)
            head = LeJEPAProjectionHead(
                input_dim=in_features,
                hidden_dim=hidden_features,
                output_dim=out_features,
            ).to(device)
            x = torch.randn(
                (batch_size, in_features), requires_grad=True, device=device
            )
            y = head(x)
            assert y.shape == (batch_size, out_features)

            y.sum().backward()
            x_grad = x.grad
            assert x_grad is not None
            assert x_grad.shape == x.shape
            for name, param in head.named_parameters():
                param_grad = param.grad
                assert param_grad is not None
                assert param_grad.shape == param.shape

    def test_lejepa_projection_head_requires_at_least_two_layers(self) -> None:
        with pytest.raises(ValueError):
            LeJEPAProjectionHead(
                input_dim=8,
                hidden_dim=16,
                output_dim=4,
                num_layers=1,
            )
        with pytest.raises(ValueError):
            LeJEPAProjectionHead(
                input_dim=8,
                hidden_dim=16,
                output_dim=4,
                num_layers=0,
            )


def test_capi_projection_head() -> None:
    head = CAPIProjectionHead(input_dim=16, num_clusters=32)
    features = torch.randn(2, 5, 16)
    logits = head(features)
    assert logits.shape == (2, 5, 32)
    # The input is L2-normalized, so the head is invariant to input scaling.
    assert torch.allclose(logits, head(features * 3.0), atol=1e-5)
    logits.sum().backward()
    assert head.layer.weight.grad is not None


def test_capi_projection_head__weight_norm() -> None:
    head = CAPIProjectionHead(input_dim=16, num_clusters=32, weight_norm=True)
    features = torch.randn(2, 5, 16)
    logits = head(features)
    assert logits.shape == (2, 5, 32)
    # The prototypes are weight-normalized to unit norm.
    prototype_norms = head.layer.weight.norm(dim=1)
    assert torch.allclose(prototype_norms, torch.ones_like(prototype_norms), atol=1e-5)
    # Still invariant to input scaling and differentiable.
    assert torch.allclose(logits, head(features * 3.0), atol=1e-5)
    logits.sum().backward()
    assert any(p.grad is not None for p in head.parameters())
