import torch
from torch import Tensor
from typing import List, Tuple, Union
from warnings import warn
from lightly.data.multi_view_collate import MultiViewCollate
import unittest


class TestMultiViewCollate(unittest.TestCase):
    def setUp(self):
        self.multi_view_collate = MultiViewCollate()

    def test_empty_batch(self):
        batch = []
        views, labels, fnames = self.multi_view_collate(batch)
        self.assertEqual(len(views), 0)
        self.assertEqual(len(labels), 0)
        self.assertEqual(len(fnames), 0)

    def test_single_item_batch(self):
        img = [torch.randn((3, 224, 224)) for _ in range(5)]
        label = 1
        fname = "image1.jpg"
        batch = [(img, label, fname)]
        views, labels, fnames = self.multi_view_collate(batch)
        self.assertEqual(len(views), 5)
        self.assertEqual(views[0].shape, (1, 3, 224, 224))
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(fnames), 1)

    def test_multiple_item_batch(self):
        img1 = [torch.randn((3, 224, 224)) for _ in range(5)]
        label1 = 1
        fname1 = "image1.jpg"
        img2 = [torch.randn((3, 224, 224)) for _ in range(5)]
        label2 = 2
        fname2 = "image2.jpg"
        batch = [(img1, label1, fname1), (img2, label2, fname2)]
        views, labels, fnames = self.multi_view_collate(batch)
        self.assertEqual(len(views), 5)
        self.assertEqual(views[0].shape, (2, 3, 224, 224))
        self.assertEqual(len(labels), 2)
        self.assertEqual(len(fnames), 2)