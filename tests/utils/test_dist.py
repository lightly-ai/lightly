import unittest
from unittest import mock

import torch

from lightly.utils import dist



class TestDist(unittest.TestCase):

    def test_eye_rank_undist(self):
        self.assertTrue(torch.all(dist.eye_rank(3) == torch.eye(3)))

    def test_eye_rank_dist(self):
        n = 3
        zeros = torch.zeros((n, n)).bool()
        eye = torch.eye(n).bool()
        for world_size in [1, 3]:
            for rank in range(0, world_size):
                with mock.patch('lightly.utils.dist.is_dist', lambda: True),\
                    mock.patch('lightly.utils.dist.world_size', lambda: world_size),\
                    mock.patch('lightly.utils.dist.rank', lambda: rank):
                    expected = []
                    for _ in range(0, rank):
                        expected.append(zeros)
                    expected.append(eye)
                    for _ in range(rank, world_size - 1):
                        expected.append(zeros)
                    expected = torch.cat(expected, dim=1)
                    self.assertTrue(torch.all(dist.eye_rank(n) == expected)) 
