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
        ]


    def _test_single_projection_head(self, head_cls, device: str = 'cpu'):
        for in_features, hidden_features, out_features in self.n_features:
            head = head_cls(in_features, hidden_features, out_features)
            head = head.eval()
            head = head.to(device)
            for batch_size in [1, 2]:
                x = torch.torch.rand((batch_size, in_features)).to(device)
                with torch.no_grad():
                    y = head(x)
                self.assertEqual(y.shape[0], batch_size)
                self.assertEqual(y.shape[1], out_features)

    def test_barlow_twins_projection_head_cpu(self):
        self._test_single_projection_head(BarlowTwinsProjectionHead)

    def test_byol_projection_head_cpu(self):
        self._test_single_projection_head(BYOLProjectionHead)

    def test_moco_projection_head_cpu(self):
        self._test_single_projection_head(MoCoProjectionHead)

    def test_nnclr_projection_head_cpu(self):
        self._test_single_projection_head(NNCLRProjectionHead)

    def test_nnclr_prediction_head_cpu(self):
        self._test_single_projection_head(NNCLRPredictionHead)

    def test_simclr_projection_head_cpu(self):
        self._test_single_projection_head(SimCLRProjectionHead)

    def test_simsiam_projection_head_cpu(self):
        self._test_single_projection_head(SimSiamProjectionHead)

    def test_simsiam_prediction_head_cpu(self):
        self._test_single_projection_head(SimSiamPredictionHead)

    def test_barlow_twins_projection_head_gpu(self):
        if torch.cuda.is_available():
            self._test_single_projection_head(BarlowTwinsProjectionHead, 'cuda')

    def test_byol_projection_head_gpu(self):
        if torch.cuda.is_available():
            self._test_single_projection_head(BYOLProjectionHead, 'cuda')

    def test_moco_projection_head_gpu(self):
        if torch.cuda.is_available():
            self._test_single_projection_head(MoCoProjectionHead, 'cuda')

    def test_nnclr_projection_head_gpu(self):
        if torch.cuda.is_available():
            self._test_single_projection_head(NNCLRProjectionHead, 'cuda')

    def test_nnclr_prediction_head_gpu(self):
        if torch.cuda.is_available():
            self._test_single_projection_head(NNCLRPredictionHead, 'cuda')

    def test_simclr_projection_head_gpu(self):
        if torch.cuda.is_available():
            self._test_single_projection_head(SimCLRProjectionHead, 'cuda')

    def test_simsiam_projection_head_gpu(self):
        if torch.cuda.is_available():
            self._test_single_projection_head(SimSiamProjectionHead, 'cuda')

    def test_simsiam_prediction_head_gpu(self):
        if torch.cuda.is_available():
            self._test_single_projection_head(SimSiamPredictionHead, 'cuda')
