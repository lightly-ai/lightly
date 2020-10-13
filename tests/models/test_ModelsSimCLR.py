import unittest

import torch

from lightly.models import ResNetSimCLR


class TestModelsSimCLR(unittest.TestCase):

    def setUp(self):
        self.resnet_variants = [
            'resnet-18',
            'resnet-34',
            'resnet-50',
            'resnet-101',
            'resnet-152'
        ]
        self.batch_size = 4
        self.input_tensor = torch.rand((self.batch_size, 3, 32, 32))

    def test_create_variations_cpu(self):
        for model_name in self.resnet_variants:
            model = ResNetSimCLR(name=model_name)
            self.assertIsNotNone(model)

    def test_create_variations_gpu(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cuda':
            for model_name in self.resnet_variants:
                model = ResNetSimCLR(name=model_name).to(device)
                self.assertIsNotNone(model)
        else:
            pass

    def test_feature_dim_configurable(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for model_name in self.resnet_variants:
            for num_ftrs in [2, 16, 32, 64]:
                for out_dim in [64, 128, 256]:
                    model = ResNetSimCLR(name=model_name,
                                         num_ftrs=num_ftrs,
                                         out_dim=out_dim).to(device)

                    # check that feature vector has correct dimension
                    with torch.no_grad():
                        out_features = model.features(
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
            for input_width in [16, 32, 64, 256]:
                for input_height in [16, 32, 64, 256]:
                    model = ResNetSimCLR(name=model_name).to(device)

                    input_tensor = torch.rand((self.batch_size,
                                               3,
                                               input_height,
                                               input_width))
                    with torch.no_grad():
                        out = model(input_tensor.to(device))

                    self.assertIsNotNone(model)
                    self.assertIsNotNone(out)


if __name__ == '__main__':
    unittest.main()
