import unittest
import torch
import torchvision
from lightly.models import utils

try:
    from lightly.models.modules import MAEEncoder, MAEDecoder, MAEBackbone
    TORCHVISION_VERSION_TOO_LOW = False
except ImportError:
    TORCHVISION_VERSION_TOO_LOW = True

@unittest.skipIf(TORCHVISION_VERSION_TOO_LOW, f"Torchvision version {torchvision.__version__} < 0.12")
class TestMAEEncoder(unittest.TestCase):
    def _vit(self):
        return torchvision.models.vision_transformer.vit_b_32(progress=False)

    def test_from_vit(self):
        MAEEncoder.from_vit_encoder(self._vit().encoder)

    def _test_forward(self, device, batch_size=8, seed=0):
        torch.manual_seed(seed)
        vit = self._vit()
        encoder = MAEEncoder.from_vit_encoder(vit.encoder).to(device)
        tokens = torch.rand(batch_size, vit.seq_length, vit.hidden_dim).to(device)
        _idx_keep, _ = utils.random_token_mask(
            size=(batch_size, vit.seq_length),
            device=device,
        )
        for idx_keep in [None, _idx_keep]:
            with self.subTest(idx_keep=idx_keep):
                out = encoder(tokens, idx_keep)

                # output shape must be correct
                expected_shape = list(tokens.shape)
                if idx_keep is not None:
                    expected_shape[1] = idx_keep.shape[1]
                self.assertListEqual(list(out.shape), expected_shape)

                # output must have reasonable numbers
                self.assertTrue(torch.all(torch.not_equal(out, torch.inf)))

    def test_forward(self):
        self._test_forward(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available.")
    def test_forward_cuda(self):
        self._test_forward(torch.device('cuda'))


@unittest.skipIf(TORCHVISION_VERSION_TOO_LOW, f"Torchvision version {torchvision.__version__} < 0.12")
class TestMAEBackbone(unittest.TestCase):
    def _vit(self):
        return torchvision.models.vision_transformer.vit_b_32(progress=False)

    def test_from_vit(self):
        MAEBackbone.from_vit(self._vit())

    def _test_forward(self, device, batch_size=8, seed=0):
        torch.manual_seed(seed)
        vit = self._vit()
        backbone = MAEBackbone.from_vit(vit).to(device)
        images = torch.rand(batch_size, 3, vit.image_size, vit.image_size).to(device)
        _idx_keep, _ = utils.random_token_mask(
            size=(batch_size, vit.seq_length),
            device=device,
        )
        for idx_keep in [None, _idx_keep]:
            with self.subTest(idx_keep=idx_keep):
                class_tokens = backbone(images, idx_keep)

                # output shape must be correct
                expected_shape = [batch_size, vit.hidden_dim]
                self.assertListEqual(list(class_tokens.shape), expected_shape)

                # output must have reasonable numbers
                self.assertTrue(torch.all(torch.not_equal(class_tokens, torch.inf)))

    def test_forward(self):
        self._test_forward(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available.")
    def test_forward_cuda(self):
        self._test_forward(torch.device('cuda'))

@unittest.skipIf(TORCHVISION_VERSION_TOO_LOW, f"Torchvision version {torchvision.__version__} < 0.12")
class TestMAEDecoder(unittest.TestCase):

    def test_init(self):
        return MAEDecoder(
            seq_length=50,
            num_layers=2,
            num_heads=4,
            embed_input_dim=128,
            hidden_dim=256,
            mlp_dim=256 * 4,
            out_dim=3 * 32 ** 2,
        )

    def _test_forward(self, device, batch_size=8, seed=0):
        torch.manual_seed(seed)
        seq_length = 50
        embed_input_dim = 128
        out_dim = 3 * 32 ** 2
        decoder = MAEDecoder(
            seq_length=seq_length,
            num_layers=2,
            num_heads=4,
            embed_input_dim=embed_input_dim,
            hidden_dim=256,
            mlp_dim=256 * 4,
            out_dim=out_dim,
        ).to(device)
        tokens = torch.rand(batch_size, seq_length, embed_input_dim).to(device)
        predictions = decoder(tokens)

        # output shape must be correct
        expected_shape = [batch_size, seq_length, out_dim]
        self.assertListEqual(list(predictions.shape), expected_shape)

        # output must have reasonable numbers
        self.assertTrue(torch.all(torch.not_equal(predictions, torch.inf)))

    def test_forward(self):
        self._test_forward(torch.device('cpu'))

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available.")
    def test_forward_cuda(self):
        self._test_forward(torch.device('cuda'))
