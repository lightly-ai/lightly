import unittest

import torch
import torch.nn as nn
import torchvision

from lightly.models import NNCLR
from lightly.models.modules import NNmemoryBankModule


def resnet_generator(name: str):
    if name == 'resnet18':
        return torchvision.models.resnet18()
    elif name == 'resnet50':
        return torchvision.models.resnet50()
    raise NotImplementedError


def get_backbone(model: nn.Module):
    backbone = torch.nn.Sequential(*(list(model.children())[:-1]))
    return backbone

class TestNNCLR(unittest.TestCase):

    def setUp(self):
        self.resnet_variants = dict(
            resnet18 = dict(
                num_ftrs=512,
                proj_hidden_dim=512,
                pred_hidden_dim=128,
                out_dim=512,
                num_mlp_layers=2
            ),
            resnet50 = dict(
                num_ftrs=2048,
                proj_hidden_dim=2048,
                pred_hidden_dim=512,
                out_dim=2048,
                num_mlp_layers=3
            )
        )
        self.batch_size = 2
        self.input_tensor = torch.rand((self.batch_size, 3, 32, 32))

    def test_create_variations_cpu(self):
        for model_name, config in self.resnet_variants.items():
            resnet = resnet_generator(model_name)
            model = NNCLR(get_backbone(resnet), **config)
            self.assertIsNotNone(model)

    def test_create_variations_gpu(self):
        if not torch.cuda.is_available():
            return

        for model_name, config in self.resnet_variants.items():
            resnet = resnet_generator(model_name)
            model = NNCLR(get_backbone(resnet), **config).to('cuda')
            self.assertIsNotNone(model)

    def test_feature_dim_configurable(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for model_name, config in self.resnet_variants.items():
            resnet = resnet_generator(model_name)
            model = NNCLR(get_backbone(resnet), **config).to(device)

            # check that feature vector has correct dimension
            with torch.no_grad():
                out_features = model.backbone(
                    self.input_tensor.to(device)
                )
            self.assertEqual(out_features.shape[1], config['num_ftrs'])

            # check that projection head output has right dimension
            with torch.no_grad():
                out_projection = model.projection_mlp(
                    out_features.squeeze()
                )
            self.assertEqual(out_projection.shape[1], config['out_dim'])

            # check that prediction head output has right dimension
            with torch.no_grad():
                out_prediction = model.prediction_mlp(
                    out_projection.squeeze()
                )
            self.assertEqual(out_prediction.shape[1], config['out_dim'])

    def test_tuple_input(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for model_name, config in self.resnet_variants.items():
            resnet = resnet_generator(model_name)
            model = NNCLR(get_backbone(resnet), **config).to(device)

            x0 = torch.rand((self.batch_size, 3, 64, 64)).to(device)
            x1 = torch.rand((self.batch_size, 3, 64, 64)).to(device)

            out = model(x0)
            self.assertEqual(out[0].shape, (self.batch_size, config['out_dim']))
            self.assertEqual(out[1].shape, (self.batch_size, config['out_dim']))

            out, features = model(x0, return_features=True)
            self.assertEqual(out[0].shape, (self.batch_size, config['out_dim']))
            self.assertEqual(out[1].shape, (self.batch_size, config['out_dim']))
            self.assertEqual(features.shape, (self.batch_size, config['num_ftrs']))

            out0, out1 = model(x0, x1)
            self.assertEqual(out0[0].shape, (self.batch_size, config['out_dim']))
            self.assertEqual(out0[1].shape, (self.batch_size, config['out_dim']))
            self.assertEqual(out1[0].shape, (self.batch_size, config['out_dim']))
            self.assertEqual(out1[1].shape, (self.batch_size, config['out_dim']))

            (out0, f0), (out1, f1) = model(x0, x1, return_features=True)
            self.assertEqual(out0[0].shape, (self.batch_size, config['out_dim']))
            self.assertEqual(out0[1].shape, (self.batch_size, config['out_dim']))
            self.assertEqual(out1[0].shape, (self.batch_size, config['out_dim']))
            self.assertEqual(out1[1].shape, (self.batch_size, config['out_dim']))
            self.assertEqual(f0.shape, (self.batch_size, config['num_ftrs']))
            self.assertEqual(f1.shape, (self.batch_size, config['num_ftrs']))

    def test_memory_bank(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for model_name, config in self.resnet_variants.items():
            resnet = resnet_generator(model_name)
            model = NNCLR(get_backbone(resnet), **config).to(device)

            for nn_size in [2 ** 3, 2 ** 8]:
                nn_replacer = NNmemoryBankModule(size=nn_size)

                with torch.no_grad():
                    for i in range(10):
                        x0 = torch.rand((self.batch_size, 3, 64, 64)).to(device)
                        x1 = torch.rand((self.batch_size, 3, 64, 64)).to(device)
                        (z0, p0), (z1, p1) = model(x0, x1)
                        z0 = nn_replacer(z0.detach(), update=False)
                        z1 = nn_replacer(z1.detach(), update=True)