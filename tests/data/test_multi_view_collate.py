import torch
from torch import Tensor
from typing import List, Tuple, Union
from warnings import warn
from lightly.data.multi_view_collate import MultiViewCollate

def test_empty_batch():
    multi_view_collate = MultiViewCollate()
    batch = []
    views, labels, fnames = multi_view_collate(batch)
    assert len(views) == 0
    assert len(labels) == 0
    assert len(fnames) == 0

def test_single_item_batch():
    multi_view_collate = MultiViewCollate()
    img = [torch.randn((3, 224, 224)) for _ in range(5)]
    label = 1
    fname = "image1.jpg"
    batch = [(img, label, fname)]
    views, labels, fnames = multi_view_collate(batch)
    assert len(views) == 5
    assert views[0].shape == (1, 3, 224, 224)
    assert labels == [1]
    assert fnames == ["image1.jpg]

def test_multiple_item_batch():
    multi_view_collate = MultiViewCollate()
    img1 = [torch.randn((3, 224, 224)) for _ in range(5)]
    label1 = 1
    fname1 = "image1.jpg"
    img2 = [torch.randn((3, 224, 224)) for _ in range(5)]
    label2 = 2
    fname2 = "image2.jpg"
    batch = [(img1, label1, fname1), (img2, label2, fname2)]
    views, labels, fnames = multi_view_collate(batch)
    assert len(views) == 5
    assert views[0].shape == (2, 3, 224, 224)
    assert labels == [1, 2]
    assert fnames == ["image1.jpg", "image2.jpg"]