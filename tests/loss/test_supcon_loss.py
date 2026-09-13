from typing import List

import pytest
import torch
from pytest_mock import MockerFixture
from torch import Tensor
from torch import distributed as dist
from torch import nn

from lightly.loss import NTXentLoss, SupConLoss


class TestSupConLoss:
    temperature = 0.5

    def test__gather_distributed(self, mocker: MockerFixture) -> None:
        mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
        SupConLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test__gather_distributed_dist_not_available(
        self, mocker: MockerFixture
    ) -> None:
        mock_is_available = mocker.patch.object(
            dist, "is_available", return_value=False
        )
        with pytest.raises(ValueError):
            SupConLoss(gather_distributed=True)
        mock_is_available.assert_called_once()

    def test_simple_input(self) -> None:
        out1 = torch.rand((3, 10))
        out2 = torch.rand((3, 10))
        my_label = Tensor([0, 1, 1])
        my_loss = SupConLoss()
        my_loss(out1, out2, my_label)

    def test_unsup_equal_to_simclr(self) -> None:
        supcon = SupConLoss(temperature=self.temperature, rescale=False)
        ntxent = NTXentLoss(temperature=self.temperature)
        out1 = torch.rand((8, 10))
        out2 = torch.rand((8, 10))
        supcon_loss = supcon(out1, out2)
        ntxent_loss = ntxent(out1, out2)
        assert (supcon_loss - ntxent_loss).pow(2).item() == pytest.approx(0.0)

    @pytest.mark.parametrize("labels", [[0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 2, 3]])
    def test_equivalence(self, labels: List[int]) -> None:
        DistributedSupCon = SupConLoss(temperature=self.temperature)
        NonDistributedSupCon = SupConLossNonDistributed(temperature=self.temperature)
        out1 = nn.functional.normalize(torch.rand(4, 10), dim=-1)
        out2 = nn.functional.normalize(torch.rand(4, 10), dim=-1)
        test_labels = Tensor(labels)

        loss1 = DistributedSupCon(out1, out2, test_labels)
        loss2 = NonDistributedSupCon(
            torch.vstack((out1, out2)), test_labels.view(-1, 1)
        )

        assert (loss1 - loss2).pow(2).item() == pytest.approx(0.0)


class SupConLossNonDistributed(nn.Module):
    def __init__(
        self,
        temperature: float = 0.1,
    ):
        """Contrastive Learning Loss Function: SupConLoss and InfoNCE Loss. Non-distributed version by Yutong.

        SupCon from Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
        InfoNCE (NT-Xent) from SimCLR: https://arxiv.org/pdf/2002.05709.pdf.

        Adapted from Yonglong Tian's work at https://github.com/HobbitLong/SupContrast/blob/master/losses.py and
        https://github.com/google-research/syn-rep-learn/blob/main/StableRep/models/losses.py.

        The function first creates a contrastive mask of shape [batch_size * n_views, batch_size * n_views], where
        mask_{i,j}=1 if sample j has the same class as sample i, except for the sample i itself.

        Next, it computes the logits from the features and then computes the soft cross-entropy loss.

        The loss is rescaled by the temperature parameter.

        For self-supervised learning, the labels should be the indices of the samples. In this case it is equivalent to InfoNCE loss.

        Attributes:
        - temperature (float): A temperature parameter to control the similarity. Default is 0.1.

        Args:
        - features (torch.Tensor): hidden vector of shape [batch_size * n_views, ...].
        - labels (torch.Tensor): ground truth of shape [batch_size, 1].
        """
        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # create n-viewed mask
        labels_n_views = labels.contiguous().repeat(
            features.shape[0] // labels.shape[0], 1
        )  # [batch_size * n_views, 1]
        contrastive_mask_n_views = torch.eq(
            labels_n_views, labels_n_views.T
        ).float()  # [batch_size * n_views, batch_size * n_views]
        contrastive_mask_n_views.fill_diagonal_(0)  # mask-out self-contrast cases

        # compute logits
        logits = (
            torch.matmul(features, features.T) / self.temperature
        )  # [batch_size * n_views, batch_size * n_views]
        logits.fill_diagonal_(-1e9)  # suppress logit for self-contrast cases

        # compute log probabilities and soft labels
        soft_label = contrastive_mask_n_views / contrastive_mask_n_views.sum(dim=1)
        log_proba = nn.functional.log_softmax(logits, dim=-1)

        # compute soft cross-entropy loss
        loss_all = torch.sum(soft_label * log_proba, dim=-1)
        loss = -loss_all.mean()

        # rescale for stable training
        loss *= self.temperature

        return loss
