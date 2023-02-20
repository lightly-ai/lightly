from PIL import Image
import unittest
from lightly.transforms.multi_view_transform import MultiViewTransform
import torchvision.transforms as T
class TestMultiViewTransform(unittest.TestCase):

    def test_multi_view_on_pil_image(self):
            for w in range(1, 10):
                for h in range(1, 10):
                    multi_view_transform = MultiViewTransform(
                        [T.RandomHorizontalFlip(p=0.1),
                        T.RandomVerticalFlip(p=0.5),
                        T.RandomGrayscale(p=0.3)]
                    )
                    sample = Image.new('RGB', (w, h))
                    output = multi_view_transform(sample)
                    self.assertEqual(len(output), 3)