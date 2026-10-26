import typing
from typing import List

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from lightly.loss import VICRegLLoss


class TestVICRegLLoss:
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No cuda")
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
        with pytest.raises(ValueError, match=error_msg):
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
        with pytest.raises(ValueError, match=error_msg):
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
        with pytest.raises(ValueError, match=error_msg):
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
        with pytest.raises(ValueError, match=error_msg):
            criterion.forward(
                global_view_features=[],
                global_view_grids=[],
                local_view_features=None,
                local_view_grids=local_view_grids,
            )

    def test_global_loss__compare(self) -> None:
        # Compare against original implementation.
        torch.manual_seed(0)
        criterion = VICRegLLoss()
        global_view_features = [
            (torch.randn((2, 32)), torch.randn((2, 7, 7, 8))) for _ in range(2)
        ]
        local_view_features = [
            (torch.randn((2, 32)), torch.randn((2, 4, 4, 8))) for _ in range(6)
        ]
        loss = criterion._global_loss(
            global_view_features=global_view_features,
            local_view_features=local_view_features,
        )

        embedding = [x for x, _ in global_view_features + local_view_features]
        expected_loss = _reference_global_loss(embedding=embedding)
        assert loss == expected_loss

    # Note: We cannot compare our local loss implementation against the original code
    # because the resulting values slightly differ. See VICRegLLoss._local_loss for
    # details.


@typing.no_type_check
def _reference_global_loss(
    embedding: List[Tensor],
    inv_coeff: float = 25.0,
    var_coeff: float = 25.0,
    cov_coeff: float = 1.0,
) -> Tensor:
    # Original global loss from VICRegL:
    # https://github.com/facebookresearch/VICRegL/blob/803ae4c8cd1649a820f03afb4793763e95317620/main_vicregl.py#L421
    def center(x):
        return x - x.mean(dim=0)

    def off_diagonal(x: Tensor) -> Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    num_views = len(embedding)
    inv_loss = 0.0
    iter_ = 0
    for i in range(2):
        for j in np.delete(np.arange(np.sum(num_views)), i):
            inv_loss = inv_loss + F.mse_loss(embedding[i], embedding[j])
            iter_ = iter_ + 1
    inv_loss = inv_coeff * inv_loss / iter_

    var_loss = 0.0
    cov_loss = 0.0
    iter_ = 0
    embedding_dim = embedding[0].shape[1]
    for i in range(num_views):
        x = center(embedding[i])
        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
        cov_x = (x.T @ x) / (x.size(0) - 1)
        cov_loss = cov_loss + off_diagonal(cov_x).pow_(2).sum().div(embedding_dim)
        iter_ = iter_ + 1
    var_loss = var_coeff * var_loss / iter_
    cov_loss = cov_coeff * cov_loss / iter_

    return inv_loss + var_loss + cov_loss
