import unittest

import torch

from lightly.loss import VICRegLLoss


class TestVICRegLLoss(unittest.TestCase):
    def test_forward(self) -> None:
        torch.manual_seed(0)
        criterion = VICRegLLoss()
        global_view_features = [
            (torch.randn((2, 32)), torch.randn((2, 7, 7, 8))) for _ in range(2)
        ]
        global_view_grids = [torch.randn((2, 7, 7, 2)) for _ in range(2)]
        local_view_features = [
            (torch.randn((2, 32)), torch.randn((2, 4, 4, 8))) for _ in range(6)
        ]
        local_view_grids = [torch.randn((2, 4, 4, 2)) for _ in range(6)]
        loss = criterion.forward(
            global_view_features=global_view_features,
            global_view_grids=global_view_grids,
            local_view_features=local_view_features,
            local_view_grids=local_view_grids,
        )
        assert loss > 0

    @unittest.skipUnless(torch.cuda.is_available(), "Cuda not available")
    def test_forward__cuda(self) -> None:
        torch.manual_seed(0)
        criterion = VICRegLLoss()
        global_view_features = [
            (torch.randn((2, 32)).cuda(), torch.randn((2, 7, 7, 8)).cuda())
            for _ in range(2)
        ]
        global_view_grids = [torch.randn((2, 7, 7, 2)).cuda() for _ in range(2)]
        local_view_features = [
            (torch.randn((2, 32)).cuda(), torch.randn((2, 4, 4, 8)).cuda())
            for _ in range(6)
        ]
        local_view_grids = [torch.randn((2, 4, 4, 2)).cuda() for _ in range(6)]
        loss = criterion.forward(
            global_view_features=global_view_features,
            global_view_grids=global_view_grids,
            local_view_features=local_view_features,
            local_view_grids=local_view_grids,
        )
        assert loss > 0

    def test_forward__error_global_view_features_and_grids_not_same_length(
        self,
    ) -> None:
        criterion = VICRegLLoss()
        global_view_features = [
            (torch.randn((2, 32)), torch.randn((2, 7, 7, 8))) for _ in range(2)
        ]
        global_view_grids = [torch.randn((2, 7, 7, 2)) for _ in range(1)]
        error_msg = (
            "global_view_features and global_view_grids must have same length but "
            "found 2 and 1."
        )
        with self.assertRaisesRegex(ValueError, error_msg):
            criterion.forward(
                global_view_features=global_view_features,
                global_view_grids=global_view_grids,
            )

    def test_forward__error_local_view_features_and_grids_not_same_length(self) -> None:
        criterion = VICRegLLoss()
        local_view_features = [
            (torch.randn((2, 32)), torch.randn((2, 4, 4, 8))) for _ in range(2)
        ]
        local_view_grids = [torch.randn((2, 4, 4, 2)) for _ in range(1)]
        error_msg = (
            "local_view_features and local_view_grids must have same length but found "
            "2 and 1."
        )
        with self.assertRaisesRegex(ValueError, error_msg):
            criterion.forward(
                global_view_features=[],
                global_view_grids=[],
                local_view_features=local_view_features,
                local_view_grids=local_view_grids,
            )

    def test_forward__error_local_view_features_and_grids_must_both_be_set(
        self,
    ) -> None:
        criterion = VICRegLLoss()
        local_view_features = [
            (torch.randn((2, 32)), torch.randn((2, 4, 4, 8))) for _ in range(2)
        ]
        local_view_grids = [torch.randn((2, 4, 4, 2)) for _ in range(2)]
        error_msg = (
            "local_view_features and local_view_grids must either both be set or None "
            "but found <class 'list'> and <class 'NoneType'>."
        )
        with self.assertRaisesRegex(ValueError, error_msg):
            criterion.forward(
                global_view_features=[],
                global_view_grids=[],
                local_view_features=local_view_features,
                local_view_grids=None,
            )

        error_msg = (
            "local_view_features and local_view_grids must either both be set or None "
            "but found <class 'NoneType'> and <class 'list'>."
        )
        with self.assertRaisesRegex(ValueError, error_msg):
            criterion.forward(
                global_view_features=[],
                global_view_grids=[],
                local_view_features=None,
                local_view_grids=local_view_grids,
            )
