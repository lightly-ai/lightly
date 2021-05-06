
import unittest

import torch
import torch.nn as nn
import torchvision

import lightly
from lightly.models import ResNetGenerator
from lightly.models import BYOL


def get_backbone(resnet, num_ftrs=64):
    last_conv_channels = list(resnet.children())[-1].in_features
    backbone = nn.Sequential(
        lightly.models.batchnorm.get_norm_layer(3, 0),
        *list(resnet.children())[:-1],
        nn.Conv2d(last_conv_channels, num_ftrs, 1),
        nn.AdaptiveAvgPool2d(1),
    )
    return backbone


class TestModelsBYOL(unittest.TestCase):

    def setUp(self):
        self.resnet_variants = [
            'resnet-18',
            'resnet-50'
        ]
        self.batch_size = 2
        self.input_tensor = torch.rand((self.batch_size, 3, 32, 32))

    def test_create_variations_cpu(self):
        for model_name in self.resnet_variants:
            resnet = ResNetGenerator(model_name)
            model = BYOL(get_backbone(resnet))
            self.assertIsNotNone(model)

    def test_create_variations_gpu(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            for model_name in self.resnet_variants:
                resnet = ResNetGenerator(model_name)
                model = BYOL(get_backbone(resnet)).to(device)
                self.assertIsNotNone(model)
        else:
            pass

    def test_feature_dim_configurable(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for model_name in self.resnet_variants:
            for num_ftrs, out_dim in zip([16, 64], [64, 256]):
                resnet = ResNetGenerator(model_name)
                model = BYOL(get_backbone(resnet,  num_ftrs=num_ftrs),
                            num_ftrs=num_ftrs,
                            out_dim=out_dim).to(device)

                # check that feature vector has correct dimension
                with torch.no_grad():
                    out_features = model.backbone(
                        self.input_tensor.to(device)
                    )
                self.assertEqual(out_features.shape[1], num_ftrs)

                # check that projection head output has right dimension
                with torch.no_grad():
                    out_projection = model.projection_head(
                        out_features.squeeze()
                    )
                self.assertEqual(out_projection.shape[1], out_dim)
                self.assertIsNotNone(model)

    def test_variations_input_dimension(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for model_name in self.resnet_variants:
            for input_width, input_height in zip([32, 64], [64, 64]):
                resnet = ResNetGenerator(model_name)
                model = BYOL(
                    get_backbone(resnet, num_ftrs=32),
                    num_ftrs=32
                ).to(device)

                input_tensor = torch.rand((self.batch_size,
                                            3,
                                            input_height,
                                            input_width))
                with torch.no_grad():
                    out, _ = model(
                        input_tensor.to(device),
                        input_tensor.to(device)
                    )

                self.assertIsNotNone(model)
                self.assertIsNotNone(out)

    def test_tuple_input(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        resnet = ResNetGenerator('resnet-18')
        model = BYOL(
            get_backbone(resnet, num_ftrs=32),
            num_ftrs=32,
            out_dim=128
        ).to(device)

        x0 = torch.rand((self.batch_size, 3, 64, 64)).to(device)
        x1 = torch.rand((self.batch_size, 3, 64, 64)).to(device)

        (z0, p0), (z1, p1) = model(x0, x1)
        self.assertEqual(z0.shape, (self.batch_size, 128))
        self.assertEqual(z1.shape, (self.batch_size, 128))
        self.assertEqual(p0.shape, (self.batch_size, 128))
        self.assertEqual(p1.shape, (self.batch_size, 128))

    def test_raises(self):
        
        resnet = ResNetGenerator('resnet-18')
        model = BYOL(get_backbone(resnet))
        x0 = torch.rand((self.batch_size, 3, 64, 64))

        with self.assertRaises(ValueError):
            model(x0, None)

        with self.assertRaises(ValueError):
            model(None, x0)

        # test different input shape
        x1 = torch.rand((self.batch_size, 5, 32, 32))
        with self.assertRaises(ValueError):
            model(x0, x1)


if __name__ == '__main__':
    unittest.main()
