import unittest
from unittest import mock

import torch

from lightly.utils import dist
from pytest import CaptureFixture


class TestDist(unittest.TestCase):
    def test_eye_rank_undist(self):
        self.assertTrue(torch.all(dist.eye_rank(3) == torch.eye(3)))

    def test_eye_rank_dist(self):
        n = 3
        zeros = torch.zeros((n, n)).bool()
        eye = torch.eye(n).bool()
        for world_size in [1, 3]:
            for rank in range(0, world_size):
                with mock.patch(
                    "torch.distributed.is_initialized", lambda: True
                ), mock.patch(
                    "lightly.utils.dist.world_size", lambda: world_size
                ), mock.patch(
                    "lightly.utils.dist.rank", lambda: rank
                ):
                    expected = []
                    for _ in range(0, rank):
                        expected.append(zeros)
                    expected.append(eye)
                    for _ in range(rank, world_size - 1):
                        expected.append(zeros)
                    expected = torch.cat(expected, dim=1)
                    self.assertTrue(torch.all(dist.eye_rank(n) == expected))


def test_rank_zero_only__rank_0() -> None:
    @dist.rank_zero_only
    def fn():
        return 0

    assert fn() == 0


def test_rank_zero_only__rank_1() -> None:
    @dist.rank_zero_only
    def fn():
        return 0

    with mock.patch.object(dist, "rank", lambda: 1):
        assert fn() is None


def test_print_rank_zero(capsys: CaptureFixture[str]) -> None:
    dist.print_rank_zero("message")
    assert capsys.readouterr().out == "message\n"
