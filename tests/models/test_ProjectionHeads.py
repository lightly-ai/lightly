import unittest

import torch
import torch.nn as nn


from lightly.models.modules.heads import BarlowTwinsProjectionHead
from lightly.models.modules.heads import BYOLProjectionHead
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
        ]


    def test_single_projection_head(self, device: str = 'cpu'):
        for head_cls in self.heads:
            for in_features, hidden_features, out_features in self.n_features:
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
    def test_single_projection_head_cuda(self):
        self.test_single_projection_head(device='cuda')

    def test_swav_prototypes(self, device: str = 'cpu'):
        for in_features, _, n_prototypes in self.n_features:
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
    def test_swav_prototypes_cuda(self):
        self.test_swav_prototypes(device='cuda')