import unittest

import torch

from lightly.models.modules.heads import BarlowTwinsProjectionHead
from lightly.models.modules.heads import BYOLProjectionHead
from lightly.models.modules.heads import DINOProjectionHead
from lightly.models.modules.heads import MoCoProjectionHead
from lightly.models.modules.heads import NNCLRProjectionHead
from lightly.models.modules.heads import NNCLRPredictionHead
from lightly.models.modules.heads import SimCLRProjectionHead
from lightly.models.modules.heads import SimSiamProjectionHead
from lightly.models.modules.heads import SimSiamPredictionHead
from lightly.models.modules.heads import SwaVProjectionHead
from lightly.models.modules.heads import SwaVPrototypes


class TestProjectionHeads(unittest.TestCase):

    def setUp(self):

        self.n_features = [
            (8, 16, 32),
            (8, 32, 16),
            (16, 8, 32),
            (16, 32, 8),
            (32, 8, 16),
            (32, 16, 8),
        ]
        self.heads = [
            BarlowTwinsProjectionHead,
            BYOLProjectionHead,
            MoCoProjectionHead,
            NNCLRProjectionHead,
            NNCLRPredictionHead,
            SimCLRProjectionHead,
            SimSiamProjectionHead,
            SimSiamPredictionHead,
            SwaVProjectionHead,
            DINOProjectionHead,       
        ]


    def test_single_projection_head(self, device: str = 'cpu', seed=0):
        for head_cls in self.heads:
            for in_features, hidden_features, out_features in self.n_features:
                torch.manual_seed(seed)
                head = head_cls(in_features, hidden_features, out_features)
                head = head.eval()
                head = head.to(device)
                for batch_size in [1, 2]:
                    msg = f'head: {head_cls}' + \
                        f'd_in, d_h, d_out = ' + \
                            f'{in_features}x{hidden_features}x{out_features}'
                    with self.subTest(msg=msg):
                        x = torch.torch.rand((batch_size, in_features)).to(device)
                        with torch.no_grad():
                            y = head(x)
                        self.assertEqual(y.shape[0], batch_size)
                        self.assertEqual(y.shape[1], out_features)

    @unittest.skipUnless(torch.cuda.is_available(), "skip")
    def test_single_projection_head_cuda(self, seed=0):
        self.test_single_projection_head(device='cuda', seed=seed)

    def test_swav_prototypes(self, device: str = 'cpu', seed=0):
        for in_features, _, n_prototypes in self.n_features:
            torch.manual_seed(seed)
            prototypes = SwaVPrototypes(in_features, n_prototypes)
            prototypes = prototypes.eval()
            prototypes = prototypes.to(device)
            for batch_size in [1, 2]:
                msg = 'prototypes d_in, n_prototypes = ' +\
                    f'{in_features} x {n_prototypes}'
                with self.subTest(msg=msg):
                        x = torch.torch.rand((batch_size, in_features)).to(device)
                        with torch.no_grad():
                            y = prototypes(x)
                        self.assertEqual(y.shape[0], batch_size)
                        self.assertEqual(y.shape[1], n_prototypes)

    @unittest.skipUnless(torch.cuda.is_available(), "skip")
    def test_swav_prototypes_cuda(self, seed=0):
        self.test_swav_prototypes(device='cuda', seed=seed)
    
    def test_dino_projection_head(self, device="cpu", seed=0):
        input_dim, hidden_dim, output_dim = self.n_features[0]
        for bottleneck_dim in [8, 16, 32]:
            for batch_norm in [False, True]:
                for norm_last_layer in [False, True]:
                    torch.manual_seed(seed)
                    head = DINOProjectionHead(
                        input_dim=input_dim,
                        hidden_dim=hidden_dim,
                        output_dim=output_dim,
                        bottleneck_dim=bottleneck_dim,
                        batch_norm=batch_norm,
                        norm_last_layer=norm_last_layer,
                    )
                    head = head.eval()
                    head = head.to(device)
                    for batch_size in [1, 2]:
                        msg = (
                            f"bottleneck_dim={bottleneck_dim}, "
                            f"batch_norm={batch_norm}, "
                            f"norm_last_layer={norm_last_layer}"
                        )
                        with self.subTest(msg=msg):
                            x = torch.torch.rand((batch_size, input_dim)).to(device)
                            with torch.no_grad():
                                y = head(x)
                            self.assertEqual(y.shape[0], batch_size)
                            self.assertEqual(y.shape[1], output_dim)

    @unittest.skipUnless(torch.cuda.is_available(), "skip")
    def test_dino_projection_head_cuda(self, seed=0):
        self.test_dino_projection_head(device="cuda", seed=seed)
