import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

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

    def test_global_loss(self) -> None:
        torch.manual_seed(0)
        criterion = VICRegLLoss()
        reference_criterion = _ReferenceLoss()
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
        expected_loss = reference_criterion.global_loss(embedding=embedding)
        assert loss == sum(expected_loss)

    def test_local_loss(self) -> None:
        torch.manual_seed(0)
        criterion = VICRegLLoss(num_matches=(3, 2))
        reference_criterion = _ReferenceLoss(num_matches=(3, 2))
        global_view_features = [
            (torch.randn((2, 32)), torch.randn((2, 3, 3, 2))) for _ in range(2)
        ]
        global_view_grids = [torch.randn((2, 3, 3, 2)) for _ in range(2)]
        local_view_features = [
            (torch.randn((2, 32)), torch.randn((2, 2, 2, 2))) for _ in range(0)
        ]
        local_view_grids = [torch.randn((2, 2, 2, 2)) for _ in range(0)]
        loss = criterion._local_loss(
            global_view_features=global_view_features,
            global_view_grids=global_view_grids,
            local_view_features=local_view_features,
            local_view_grids=local_view_grids,
        )

        maps_embedding = [
            x.flatten(1, 2) for _, x in global_view_features + local_view_features
        ]
        locations = [x for x in global_view_grids + local_view_grids]
        expected_loss = reference_criterion.local_loss(
            maps_embedding=maps_embedding,
            locations=locations,
        )
        assert loss == sum(expected_loss)

    def test_loss(self) -> None:
        torch.manual_seed(0)
        criterion = VICRegLLoss()
        reference_criterion = _ReferenceLoss()
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

        embedding = [x for x, _ in global_view_features + local_view_features]
        maps_embedding = [
            x.flatten(1, 2) for _, x in global_view_features + local_view_features
        ]
        locations = [x for x in global_view_grids + local_view_grids]
        expected_loss = reference_criterion.loss(
            embedding=embedding,
            maps_embedding=maps_embedding,
            locations=locations,
        )
        assert loss == expected_loss


class _ReferenceLoss:
    def __init__(
        self,
        alpha: float = 0.75,
        inv_coeff: float = 25.0,
        var_coeff: float = 25.0,
        cov_coeff: float = 1.0,
        num_matches=(20, 4),
    ) -> None:
        self.alpha = alpha
        self.inv_coeff = inv_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.num_matches = num_matches
        self.l2_all_matches = False
        self.fast_vc_reg = False

    def loss(self, embedding, maps_embedding, locations):
        loss = 0
        if self.alpha > 0.0:
            inv_loss, var_loss, cov_loss = self.global_loss(embedding=embedding)
            loss = loss + self.alpha * (inv_loss + var_loss + cov_loss)
        if self.alpha < 1.0:
            inv_loss, var_loss, cov_loss = self.local_loss(
                maps_embedding=maps_embedding,
                locations=locations,
            )
            loss = loss + (1 - self.alpha) * (inv_loss + var_loss + cov_loss)
        return loss

    def _vicreg_loss(self, x, y):
        print("-x", x, x.sum())
        print("-y", y, y.sum())
        repr_loss = self.inv_coeff * F.mse_loss(x, y)

        x = self.center(x)
        y = self.center(y)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = self.var_coeff * (
            torch.mean(F.relu(1.0 - std_x)) / 2 + torch.mean(F.relu(1.0 - std_y)) / 2
        )

        x = x.permute((1, 0, 2))
        y = y.permute((1, 0, 2))

        *_, sample_size, num_channels = x.shape
        non_diag_mask = ~torch.eye(num_channels, device=x.device, dtype=torch.bool)
        # Center features
        # centered.shape = NC
        # x = x - x.mean(dim=-2, keepdim=True)
        # y = y - y.mean(dim=-2, keepdim=True)

        cov_x = torch.einsum("...nc,...nd->...cd", x, x) / (sample_size - 1)
        cov_y = torch.einsum("...nc,...nd->...cd", y, y) / (sample_size - 1)
        cov_loss = (cov_x[..., non_diag_mask].pow(2).sum(-1) / num_channels) / 2 + (
            cov_y[..., non_diag_mask].pow(2).sum(-1) / num_channels
        ) / 2
        cov_loss = cov_loss.mean()
        cov_loss = self.cov_coeff * cov_loss
        print("-vic", repr_loss, std_loss, cov_loss)
        return repr_loss, std_loss, cov_loss

    def _local_loss(self, maps_1, maps_2, location_1, location_2):
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0

        # L2 distance based bacthing
        if self.l2_all_matches:
            num_matches_on_l2 = [None, None]
        else:
            num_matches_on_l2 = self.num_matches

        maps_1_filtered, maps_1_nn = self.neirest_neighbores_on_l2(
            maps_1, maps_2, num_matches=num_matches_on_l2[0]
        )
        maps_2_filtered, maps_2_nn = self.neirest_neighbores_on_l2(
            maps_2, maps_1, num_matches=num_matches_on_l2[1]
        )
        # print('-l2 sum', (maps_1_filtered - maps_1_nn).sum())

        if self.fast_vc_reg:
            inv_loss_1 = F.mse_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2 = F.mse_loss(maps_2_filtered, maps_2_nn)
        else:
            inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(
                maps_1_filtered, maps_1_nn
            )
            inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(
                maps_2_filtered, maps_2_nn
            )
            var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
            cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)
        # print('-l2', (inv_loss + var_loss + cov_loss))

        # Location based matching
        location_1 = location_1.flatten(1, 2)
        location_2 = location_2.flatten(1, 2)

        maps_1_filtered, maps_1_nn = self.neirest_neighbores_on_location(
            location_1,
            location_2,
            maps_1,
            maps_2,
            num_matches=self.num_matches[0],
        )
        maps_2_filtered, maps_2_nn = self.neirest_neighbores_on_location(
            location_2,
            location_1,
            maps_2,
            maps_1,
            num_matches=self.num_matches[1],
        )
        # print('-loc sum', (maps_1_filtered - maps_1_nn).sum())

        if self.fast_vc_reg:
            inv_loss_1 = F.mse_loss(maps_1_filtered, maps_1_nn)
            inv_loss_2 = F.mse_loss(maps_2_filtered, maps_2_nn)
        else:
            inv_loss_1, var_loss_1, cov_loss_1 = self._vicreg_loss(
                maps_1_filtered, maps_1_nn
            )
            inv_loss_2, var_loss_2, cov_loss_2 = self._vicreg_loss(
                maps_2_filtered, maps_2_nn
            )
            var_loss = var_loss + (var_loss_1 / 2 + var_loss_2 / 2)
            cov_loss = cov_loss + (cov_loss_1 / 2 + cov_loss_2 / 2)

        inv_loss = inv_loss + (inv_loss_1 / 2 + inv_loss_2 / 2)
        # print('-loc', ((var_loss_1 / 2 + var_loss_2 / 2) + (cov_loss_1 / 2 + cov_loss_2 / 2) + (inv_loss_1 / 2 + inv_loss_2 / 2)))
        return inv_loss, var_loss, cov_loss

    def local_loss(self, maps_embedding, locations):
        num_views = len(maps_embedding)
        inv_loss = 0.0
        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss_this, var_loss_this, cov_loss_this = self._local_loss(
                    maps_embedding[i],
                    maps_embedding[j],
                    locations[i],
                    locations[j],
                )
                inv_loss = inv_loss + inv_loss_this
                var_loss = var_loss + var_loss_this
                cov_loss = cov_loss + cov_loss_this
                # print('-- local', inv_loss_this + var_loss_this + cov_loss_this)
                iter_ += 1

        inv_loss = inv_loss / iter_
        var_loss = var_loss / iter_
        cov_loss = cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def global_loss(self, embedding):
        num_views = len(embedding)
        inv_loss = 0.0
        iter_ = 0
        for i in range(2):
            for j in np.delete(np.arange(np.sum(num_views)), i):
                inv_loss = inv_loss + F.mse_loss(embedding[i], embedding[j])
                iter_ = iter_ + 1
        inv_loss = self.inv_coeff * inv_loss / iter_

        var_loss = 0.0
        cov_loss = 0.0
        iter_ = 0
        embedding_dim = embedding[0].shape[1]
        for i in range(num_views):
            x = self.center(embedding[i])
            std_x = torch.sqrt(x.var(dim=0) + 0.0001)
            var_loss = var_loss + torch.mean(torch.relu(1.0 - std_x))
            cov_x = (x.T @ x) / (x.size(0) - 1)
            cov_loss = cov_loss + self.off_diagonal(cov_x).pow_(2).sum().div(
                embedding_dim
            )
            iter_ = iter_ + 1
        var_loss = self.var_coeff * var_loss / iter_
        cov_loss = self.cov_coeff * cov_loss / iter_

        return inv_loss, var_loss, cov_loss

    def batched_index_select(self, input, dim, index):
        for ii in range(1, len(input.shape)):
            if ii != dim:
                index = index.unsqueeze(ii)
        expanse = list(input.shape)
        expanse[0] = -1
        expanse[dim] = -1
        index = index.expand(expanse)
        return torch.gather(input, dim, index)

    def nearest_neighbors(self, input_maps, candidate_maps, distances, num_matches):
        batch_size = input_maps.size(0)

        if num_matches is None or num_matches == -1:
            num_matches = input_maps.size(1)

        topk_values, topk_indices = distances.topk(k=1, largest=False)
        topk_values = topk_values.squeeze(-1)
        topk_indices = topk_indices.squeeze(-1)

        sorted_values, sorted_values_indices = torch.sort(topk_values, dim=1)
        sorted_indices, sorted_indices_indices = torch.sort(
            sorted_values_indices, dim=1
        )

        mask = torch.stack(
            [
                torch.where(sorted_indices_indices[i] < num_matches, True, False)
                for i in range(batch_size)
            ]
        )
        topk_indices_selected = topk_indices.masked_select(mask)
        topk_indices_selected = topk_indices_selected.reshape(batch_size, num_matches)

        indices = (
            torch.arange(0, topk_values.size(1))
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(topk_values.device)
        )
        indices_selected = indices.masked_select(mask)
        indices_selected = indices_selected.reshape(batch_size, num_matches)

        filtered_input_maps = self.batched_index_select(input_maps, 1, indices_selected)
        filtered_candidate_maps = self.batched_index_select(
            candidate_maps, 1, topk_indices_selected
        )

        return filtered_input_maps, filtered_candidate_maps

    def neirest_neighbores_on_l2(self, input_maps, candidate_maps, num_matches):
        """
        input_maps: (B, H * W, C)
        candidate_maps: (B, H * W, C)
        """
        distances = torch.cdist(input_maps, candidate_maps)
        return self.nearest_neighbors(
            input_maps, candidate_maps, distances, num_matches
        )

    def neirest_neighbores_on_location(
        self,
        input_location,
        candidate_location,
        input_maps,
        candidate_maps,
        num_matches,
    ):
        """
        input_location: (B, H * W, 2)
        candidate_location: (B, H * W, 2)
        input_maps: (B, H * W, C)
        candidate_maps: (B, H * W, C)
        """
        distances = torch.cdist(input_location, candidate_location)
        return self.nearest_neighbors(
            input_maps, candidate_maps, distances, num_matches
        )

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    def center(self, x):
        return x - x.mean(dim=0)

    def off_diagonal(self, x: Tensor) -> Tensor:
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def test_nn():
    criterion = _ReferenceLoss()

    input_maps = torch.Tensor(
        [
            [[1], [2], [3]],
        ]
    )
    candidate_maps = torch.Tensor(
        [
            [[11], [22], [33]],
        ]
    )
    distances = torch.Tensor([[[1, 3, 5], [0, 8, 4], [2, 6, 7]]])
    nn_input, nn_candidate = criterion.nearest_neighbors(
        input_maps=input_maps,
        candidate_maps=candidate_maps,
        distances=distances,
        num_matches=2,
    )
    assert nn_input.tolist() == [[[1], [2]]]
    assert nn_candidate.tolist() == [[[11], [11]]]

    input_maps = torch.Tensor(
        [
            [[1], [2]],
        ]
    )
    candidate_maps = torch.Tensor(
        [
            [[11], [22]],
        ]
    )
    distances = torch.Tensor([[[0, 1], [3, 2]]])
    nn_input, nn_candidate = criterion.nearest_neighbors(
        input_maps=input_maps,
        candidate_maps=candidate_maps,
        distances=distances,
        num_matches=2,
    )
    assert nn_input.tolist() == [[[1], [2]]]
    assert nn_candidate.tolist() == [[[11], [22]]]

    distances = torch.Tensor([[[2, 3], [1, 0]]])
    nn_input, nn_candidate = criterion.nearest_neighbors(
        input_maps=input_maps,
        candidate_maps=candidate_maps,
        distances=distances,
        num_matches=2,
    )
    assert nn_input.tolist() == [[[2], [1]]]
    assert nn_candidate.tolist() == [[[22], [11]]]

    distances = torch.Tensor([[[0, 2], [1, 3]]])
    nn_input, nn_candidate = criterion.nearest_neighbors(
        input_maps=input_maps,
        candidate_maps=candidate_maps,
        distances=distances,
        num_matches=2,
    )
    assert nn_input.tolist() == [[[1], [2]]]
    assert nn_candidate.tolist() == [[[11], [11]]]
    criterion.nearest_neighbors(
        input_maps=input_maps,
        candidate_maps=candidate_maps,
        distances=distances,
        num_matches=6,
    )


def vicreg_invariance_loss(x: Tensor, y: Tensor) -> Tensor:
    return F.mse_loss(x, y)


def vicreg_variance_loss(x: Tensor, eps: float = 0.0001) -> Tensor:
    x = x - x.mean(dim=0)
    std = torch.sqrt(x.var(dim=0) + eps)
    loss = torch.mean(F.relu(1 - std))
    return loss


def vicreg_covariance_loss(x: Tensor) -> Tensor:
    x = x - x.mean(dim=0)
    batch_size = x.size(0)
    dim = x.size(-1)
    nondiag_mask = ~torch.eye(dim, device=x.device, dtype=torch.bool)
    cov = torch.einsum("b...c,b...d->...cd", x, x) / (batch_size - 1)
    loss = cov[..., nondiag_mask].pow(2).sum(-1) / dim
    return loss.mean()


def vicreg_loss(x, y, inv_coeff=25.0, var_coeff=25.0, cov_coeff=1.0) -> Tensor:
    inv_loss = vicreg_invariance_loss(x, y)
    var_loss = 0.5 * (vicreg_variance_loss(x) + vicreg_variance_loss(y))
    cov_loss = 0.5 * (vicreg_covariance_loss(x) + vicreg_covariance_loss(y))
    return inv_coeff * inv_loss + var_coeff * var_loss + cov_coeff * cov_loss


def test_vicreg_covariance_loss():
    vicreg_covariance_loss(torch.rand(3, 10))
    vicreg_covariance_loss(torch.rand(3, 4, 10))


def test_vicreg_loss():
    torch.manual_seed(0)
    x = torch.rand(3, 10)
    y = torch.rand(3, 10)
    loss = vicreg_loss(x, y)
    from lightly.loss import VICRegLoss

    criterion = VICRegLoss(nu_param=0.5)
    assert criterion(x, y) == loss


def test_vicreg_large_loss():
    x = torch.rand(3, 4, 10)
    y = torch.rand(3, 4, 10)
    criterion = _ReferenceLoss()
    assert sum(criterion._vicreg_loss(x, y)) == vicreg_loss(x, y)


def test_vicreg_large_loss_():
    x = torch.rand(3, 4, 10)
    y = torch.rand(3, 4, 10)
    criterion = _ReferenceLoss()
    from lightly.loss import VICRegLoss

    c = VICRegLoss(nu_param=0.5)
    assert sum(criterion._vicreg_loss(x, y)) == c(
        x[:, [1, 0, 3, 2]], y[:, [1, 0, 3, 2]]
    )
