import random
import unittest

import torchvision
import torchvision.transforms as transforms

from lightly.data import BaseCollateFunction
from lightly.data import ImageCollateFunction
from lightly.data import SimCLRCollateFunction


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
