import unittest
from typing import List

import torch
import torch.nn as nn

import lightly
from lightly.models import DirectCLR, ResNetGenerator


def get_backbone(resnet: nn.Module, num_ftrs: int = 64) -> nn.Module:
    # ignoring type checking as mypy infers this to be Tensor | Module
    last_conv_channels: int = list(resnet.children())[-1].in_features  # type: ignore
    backbone = nn.Sequential(
        lightly.models.batchnorm.get_norm_layer(3, 0),
        *list(resnet.children())[:-1],
        nn.Conv2d(last_conv_channels, num_ftrs, 1),
        nn.AdaptiveAvgPool2d(1),
    )
    return backbone


class TestModelsDirectCLR(unittest.TestCase):
    def setUp(self) -> None:
        self.resnet_variants: List[str] = ["resnet-18", "resnet-50"]
        self.batch_size: int = 2
        self.input_tensor: torch.Tensor = torch.rand((self.batch_size, 3, 32, 32))

    def test_create_variations_cpu(self) -> None:
        for model_name in self.resnet_variants:
            resnet = ResNetGenerator(model_name)
            model = DirectCLR(get_backbone(resnet))
            self.assertIsNotNone(model)

    def test_create_variations_gpu(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            for model_name in self.resnet_variants:
                resnet = ResNetGenerator(model_name)
                model = DirectCLR(get_backbone(resnet)).to(device)
                self.assertIsNotNone(model)
        else:
            pass

    def test_feature_dim_configurable(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for model_name in self.resnet_variants:
            for num_ftrs in [16, 64]:
                resnet = ResNetGenerator(model_name)
                model = DirectCLR(get_backbone(resnet, num_ftrs=num_ftrs)).to(device)

                # check that feature vector has correct dimension
                with torch.no_grad():
                    out_features = model.backbone(self.input_tensor.to(device))
                self.assertEqual(out_features.shape[1], num_ftrs)
                self.assertIsNotNone(model)

    def test_variations_input_dimension(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for model_name in self.resnet_variants:
            for input_width, input_height in zip([32, 64], [64, 64]):
                resnet = ResNetGenerator(model_name)
                model = DirectCLR(get_backbone(resnet, num_ftrs=64)).to(device)

                input_tensor = torch.rand(
                    (self.batch_size, 3, input_height, input_width)
                )
                with torch.no_grad():
                    out = model(input_tensor.to(device))

                self.assertIsNotNone(model)
                self.assertIsNotNone(out)

    def test_tuple_input(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        resnet = ResNetGenerator("resnet-18")
        model = DirectCLR(get_backbone(resnet, num_ftrs=64), dim=32).to(device)

        x0 = torch.rand((self.batch_size, 3, 64, 64)).to(device)
        x1 = torch.rand((self.batch_size, 3, 64, 64)).to(device)

        out = model(x0)
        self.assertEqual(out.shape, (self.batch_size, 32))

        out, features = model(x0, return_features=True)
        self.assertEqual(out.shape, (self.batch_size, 32))
        self.assertEqual(features.shape, (self.batch_size, 64))

        out0, out1 = model(x0, x1)
        self.assertEqual(out0.shape, (self.batch_size, 32))
        self.assertEqual(out1.shape, (self.batch_size, 32))

        (out0, f0), (out1, f1) = model(x0, x1, return_features=True)
        self.assertEqual(out0.shape, (self.batch_size, 32))
        self.assertEqual(out1.shape, (self.batch_size, 32))
        self.assertEqual(f0.shape, (self.batch_size, 64))
        self.assertEqual(f1.shape, (self.batch_size, 64))


if __name__ == "__main__":
    unittest.main()
