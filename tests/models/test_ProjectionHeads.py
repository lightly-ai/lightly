import unittest

import torch

from lightly.loss import DINOLoss
from lightly.models.modules.heads import (
    BarlowTwinsProjectionHead,
    BYOLPredictionHead,
    BYOLProjectionHead,
    DenseCLProjectionHead,
    DINOProjectionHead,
    DINOv2ProjectionHead,
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


class TestProjectionHeads(unittest.TestCase):
    def setUp(self) -> None:
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

    def test_single_projection_head(self, device: str = "cpu", seed: int = 0) -> None:
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
                    msg = (
                        f"head: {head_cls}"
                        + f"d_in, d_h, d_out = "
                        + f"{in_features}x{hidden_features}x{out_features}"
                    )
                    with self.subTest(msg=msg):
                        x = torch.torch.rand((batch_size, in_features)).to(device)
                        with torch.no_grad():
                            y = head(x)
                        self.assertEqual(y.shape[0], batch_size)
                        self.assertEqual(y.shape[1], out_features)

    @unittest.skipUnless(torch.cuda.is_available(), "skip")
    def test_single_projection_head_cuda(self, seed: int = 0) -> None:
        self.test_single_projection_head(device="cuda", seed=seed)

    def test_swav_prototypes(self, device: str = "cpu", seed: int = 0) -> None:
        for in_features, _, n_prototypes in self.n_features:
            torch.manual_seed(seed)
            prototypes = SwaVPrototypes(in_features, n_prototypes)
            prototypes = prototypes.eval()
            prototypes = prototypes.to(device)
            for batch_size in [1, 2]:
                msg = (
                    "prototypes d_in, n_prototypes = "
                    + f"{in_features} x {n_prototypes}"
                )
                with self.subTest(msg=msg):
                    x = torch.torch.rand((batch_size, in_features)).to(device)
                    with torch.no_grad():
                        y = prototypes(x)
                    self.assertEqual(y.shape[0], batch_size)
                    self.assertEqual(y.shape[1], n_prototypes)

    def test_swav_frozen_prototypes(self, seed: int = 0) -> None:
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
                self.assertEqual(loss, loss0)
            if step > 2:
                self.assertNotEqual(loss, loss0)

    def test_swav_multi_prototypes(self, device: str = "cpu", seed: int = 0) -> None:
        for in_features, _, n_prototypes in self.swavProtoypes:
            torch.manual_seed(seed)
            prototypes = SwaVPrototypes(in_features, n_prototypes)
            prototypes = prototypes.eval()
            prototypes = prototypes.to(device)
            for batch_size in [1, 2]:
                msg = (
                    "prototypes d_in, n_prototypes = "
                    + f"{in_features} x {n_prototypes}"
                )
                with self.subTest(msg=msg):
                    x = torch.torch.rand((batch_size, in_features)).to(device)
                    with torch.no_grad():
                        y = prototypes(x)
                    for layerNum, prototypeSize in enumerate(n_prototypes):
                        self.assertEqual(y[layerNum].shape[0], batch_size)
                        self.assertEqual(y[layerNum].shape[1], prototypeSize)

    @unittest.skipUnless(torch.cuda.is_available(), "skip")
    def test_swav_prototypes_cuda(self, device: str = "cuda", seed: int = 0) -> None:
        self.test_swav_prototypes(device=device, seed=seed)

    def test_dino_projection_head(self, device: str = "cpu", seed: int = 0) -> None:
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
                    msg = (
                        f"bottleneck_dim={bottleneck_dim}, " f"batch_norm={batch_norm}"
                    )
                    with self.subTest(msg=msg):
                        x = torch.torch.rand((batch_size, input_dim)).to(device)
                        with torch.no_grad():
                            y = head(x)
                        self.assertEqual(y.shape[0], batch_size)
                        self.assertEqual(y.shape[1], output_dim)

    @unittest.skipUnless(torch.cuda.is_available(), "skip")
    def test_dino_projection_head_cuda(
        self, device: str = "cuda", seed: int = 0
    ) -> None:
        self.test_dino_projection_head(device=device, seed=seed)

    def test_dino_projection_head_freeze_last_layer(self, seed: int = 0) -> None:
        """Test if freeze last layer cancels backprop."""
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
                    with self.subTest(
                        f"norm_last_layer={norm_last_layer}, "
                        f"freeze_last_layer={freeze_last_layer}, "
                        f"epoch={epoch}"
                    ):
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
                                    self.assertFalse(are_same)
                                else:
                                    self.assertTrue(are_same)

    def test_dinov2_projection_head(self, device: str = "cpu", seed: int = 0) -> None:
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
                    msg = (
                        f"bottleneck_dim={bottleneck_dim}, " f"batch_norm={batch_norm}"
                    )
                    with self.subTest(msg=msg):
                        x = torch.torch.rand((batch_size, input_dim)).to(device)
                        with torch.no_grad():
                            y = head(x)
                        self.assertEqual(y.shape[0], batch_size)
                        self.assertEqual(y.shape[1], output_dim)

    @unittest.skipUnless(torch.cuda.is_available(), "skip")
    def test_dinov2_projection_head_cuda(
        self, device: str = "cuda", seed: int = 0
    ) -> None:
        self.test_dinov2_projection_head(device=device, seed=seed)

    def test_simclr_projection_head_multiple_layers(
        self, device: str = "cpu", seed: int = 0
    ) -> None:
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
                        msg = (
                            f"head: SimCLRProjectionHead"
                            + f"d_in, d_h, d_out = "
                            + f"{in_features}x{hidden_features}x{out_features}"
                        )
                        with self.subTest(msg=msg):
                            x = torch.torch.rand((batch_size, in_features)).to(device)
                            with torch.no_grad():
                                y = head(x)
                            self.assertEqual(y.shape[0], batch_size)
                            self.assertEqual(y.shape[1], out_features)

    def test_moco_projection_head_multiple_layers(
        self, device: str = "cpu", seed: int = 0
    ) -> None:
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
                        msg = (
                            f"head: MoCoProjectionHead"
                            + f"d_in, d_h, d_out = "
                            + f"{in_features}x{hidden_features}x{out_features}"
                        )
                        with self.subTest(msg=msg):
                            x = torch.torch.rand((batch_size, in_features)).to(device)
                            with torch.no_grad():
                                y = head(x)
                            self.assertEqual(y.shape[0], batch_size)
                            self.assertEqual(y.shape[1], out_features)
