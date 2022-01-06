import random
import unittest

import torchvision
import torchvision.transforms as transforms

from lightly.data import BaseCollateFunction
from lightly.data import ImageCollateFunction
from lightly.data import SimCLRCollateFunction
from lightly.data import MultiCropCollateFunction
from lightly.data import SwaVCollateFunction


class TestDataCollate(unittest.TestCase):

    def create_batch(self, batch_size=16):
        rnd_images = torchvision.datasets.FakeData(size=batch_size)

        fnames = [f'img_{i}.jpg' for i in range(batch_size)]
        labels = [random.randint(0, 5) for i in range(batch_size)]

        batch = []

        for i in range(batch_size):
            batch.append((rnd_images[i][0], labels[i], fnames[i]))

        return batch

    def test_base_collate(self):
        batch = self.create_batch()
        transform = transforms.ToTensor()
        collate = BaseCollateFunction(transform)
        samples, labels, fnames = collate(batch)
        samples0, samples1 = samples

        self.assertIsNotNone(collate)
        self.assertEqual(len(samples0), len(samples1))
        self.assertEqual(len(samples1), len(labels), len(fnames))

    def test_image_collate(self):
        batch = self.create_batch()
        img_collate = ImageCollateFunction()
        samples, labels, fnames = img_collate(batch)
        samples0, samples1 = samples

        self.assertIsNotNone(img_collate)
        self.assertEqual(len(samples0), len(samples1))
        self.assertEqual(len(samples1), len(labels), len(fnames))

    def test_image_collate_tuple_input_size(self):
        batch = self.create_batch()
        img_collate = ImageCollateFunction(
            input_size=(32, 32),
        )
        samples, labels, fnames = img_collate(batch)
        samples0, samples1 = samples

        self.assertIsNotNone(img_collate)
        self.assertEqual(len(samples0), len(samples1))
        self.assertEqual(len(samples1), len(labels), len(fnames))

    def test_simclr_collate_tuple_input_size(self):
        batch = self.create_batch()
        img_collate = SimCLRCollateFunction(
            input_size=(32, 32),
        )
        samples, labels, fnames = img_collate(batch)
        samples0, samples1 = samples

        self.assertIsNotNone(img_collate)
        self.assertEqual(len(samples0), len(samples1))
        self.assertEqual(len(samples1), len(labels), len(fnames))

    def test_multi_crop_collate(self):
        batch = self.create_batch()
        for high in range(2, 4):
            for low in range(6):
                with self.subTest(msg='n_low_res={low}, n_high_res={high}'):
                    multi_crop_collate = MultiCropCollateFunction(
                        crop_sizes=[32, 16],
                        crop_counts=[high, low],
                        crop_min_scales=[0.14, 0.04],
                        crop_max_scales=[1.0, 0.14],
                        transforms=torchvision.transforms.ToTensor(),
                    )
                    samples, labels, fnames = multi_crop_collate(batch)
                    self.assertIsNotNone(multi_crop_collate)
                    self.assertEqual(len(samples), low + high)
                    for i, crop in enumerate(samples):
                        if i < high:
                            self.assertEqual(crop.shape[-1], 32)
                            self.assertEqual(crop.shape[-2], 32)
                        else:
                            self.assertEqual(crop.shape[-1], 16)
                            self.assertEqual(crop.shape[-2], 16)
                        self.assertEqual(len(crop), len(labels), len(fnames))


    def test_swav_collate_init(self):
        swav_collate = SwaVCollateFunction()

    def test_swav_collate_init_fail(self):
        with self.assertRaises(ValueError):
            SwaVCollateFunction(
                crop_sizes=[1],
                crop_counts=[2, 3],
            )
                        
