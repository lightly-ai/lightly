from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, List, Tuple, TypeVar
from unittest import mock

import numpy as np
import numpy.linalg as npl
import pytest
import scipy.special as sps
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    try:
        from numpy.typing import NDArray
    except ImportError:
        T = TypeVar("T", bound=np.generic)

        class NDArray(Generic[T]):
            pass


from pytest_mock import MockerFixture
from torch import Tensor
from torch import distributed as dist

from lightly.loss import DetConBLoss


@pytest.mark.parametrize(
    "temperature,batch_size,num_classes,sampled_masks,emb_dim",
    [
        [0.1, 4, 16, 24, 32],
        [0.5, 8, 32, 16, 64],
        [1.0, 16, 64, 32, 128],
    ],
)
def test_DetConBLoss_against_original(
    temperature: float,
    batch_size: int,
    num_classes: int,
    sampled_masks: int,
    emb_dim: int,
) -> None:
    pred1 = torch.randn(batch_size, sampled_masks, emb_dim)
    pred2 = torch.randn(batch_size, sampled_masks, emb_dim)
    target1 = torch.randn(batch_size, sampled_masks, emb_dim)
    target2 = torch.randn(batch_size, sampled_masks, emb_dim)

    mask1 = torch.randint(0, num_classes, (batch_size, sampled_masks))
    mask2 = torch.randint(0, num_classes, (batch_size, sampled_masks))

    orig_loss = byol_nce_detcon(
        pred1.numpy(),
        pred2.numpy(),
        target1.numpy(),
        target2.numpy(),
        mask1.numpy(),
        mask2.numpy(),
        mask1.numpy(),
        mask2.numpy(),
        temperature=temperature,
    )

    loss_fn = DetConBLoss(temperature=temperature, gather_distributed=False)
    loss = loss_fn(pred1, pred2, target1, target2, mask1, mask2)

    assert torch.allclose(loss, torch.tensor(orig_loss, dtype=torch.float32), atol=1e-4)


@pytest.mark.parametrize(
    "temperature,batch_size,num_classes,sampled_masks,emb_dim,world_size",
    [
        [0.1, 4, 16, 24, 32, 1],
        [0.5, 8, 32, 16, 64, 2],
        [1.0, 16, 64, 32, 128, 4],
    ],
)
def test_DetConBLoss_distributed_against_original(
    mocker: MockerFixture,
    temperature: float,
    batch_size: int,
    num_classes: int,
    sampled_masks: int,
    emb_dim: int,
    world_size: int,
) -> None:
    tensors = [
        {
            "pred1": torch.randn(batch_size, sampled_masks, emb_dim),
            "pred2": torch.randn(batch_size, sampled_masks, emb_dim),
            "target1": torch.randn(batch_size, sampled_masks, emb_dim),
            "target2": torch.randn(batch_size, sampled_masks, emb_dim),
            "mask1": torch.randint(0, num_classes, (batch_size, sampled_masks)),
            "mask2": torch.randint(0, num_classes, (batch_size, sampled_masks)),
        }
        for _ in range(world_size)
    ]

    # calculate non-distributed by packing all batches
    loss_nondist = byol_nce_detcon(
        torch.cat([t["pred1"] for t in tensors], dim=0).numpy(),
        torch.cat([t["pred2"] for t in tensors], dim=0).numpy(),
        torch.cat([t["target1"] for t in tensors], dim=0).numpy(),
        torch.cat([t["target2"] for t in tensors], dim=0).numpy(),
        torch.cat([t["mask1"] for t in tensors], dim=0).numpy(),
        torch.cat([t["mask2"] for t in tensors], dim=0).numpy(),
        torch.cat([t["mask1"] for t in tensors], dim=0).numpy(),
        torch.cat([t["mask2"] for t in tensors], dim=0).numpy(),
        temperature=temperature,
    )

    mock_is_available = mocker.patch.object(dist, "is_available", return_value=True)
    mock_get_world_size = mocker.patch.object(
        dist, "get_world_size", return_value=world_size
    )

    loss_fn = DetConBLoss(temperature=temperature, gather_distributed=True)

    total_loss: Tensor = torch.tensor(0.0)
    for rank in range(world_size):
        mock_get_rank = mocker.patch.object(dist, "get_rank", return_value=rank)
        mock_gather = mocker.patch.object(
            dist,
            "gather",
            side_effect=[
                [t["target1"] for t in tensors],
                [t["target2"] for t in tensors],
            ],
        )
        loss_val = loss_fn(
            tensors[rank]["pred1"],
            tensors[rank]["pred2"],
            tensors[rank]["target1"],
            tensors[rank]["target2"],
            tensors[rank]["mask1"],
            tensors[rank]["mask2"],
        )
        total_loss += loss_val
    total_loss /= world_size

    assert torch.allclose(
        total_loss, torch.tensor(loss_nondist, dtype=torch.float32), atol=1e-4
    )


### Original JAX/haiku Implementation with Minimal Changes ###
# Source: https://github.com/google-deepmind/detcon/blob/main/utils/losses.py
#
# Changes:
# 1. change any jnp function to np
# 2. remove distributed implementation (doesn't work anyway with numpy)
# 3. use scipy.special.log_softmax instead of jax.nn.log_softmax
# 4. commented unused
# 5. change hk.one_hot to np.eye
# 6. changer helper function (norm) to numpy.linalg.norm
def byol_nce_detcon(
    pred1: NDArray[np.float64],
    pred2: NDArray[np.float64],
    target1: NDArray[np.float64],
    target2: NDArray[np.float64],
    pind1: NDArray[np.int64],
    pind2: NDArray[np.int64],
    tind1: NDArray[np.int64],
    tind2: NDArray[np.int64],
    temperature: float = 0.1,
    use_replicator_loss: bool = True,
    local_negatives: bool = True,
) -> float:
    """Compute the NCE scores from pairs of predictions and targets.

    This implements the batched form of the loss described in
    Section 3.1, Equation 3 in https://arxiv.org/pdf/2103.10957.pdf.

    Args:
      pred1 (jnp.array): the prediction from first view.
      pred2 (jnp.array): the prediction from second view.
      target1 (jnp.array): the projection from first view.
      target2 (jnp.array): the projection from second view.
      pind1 (jnp.array): mask indices for first view's prediction.
      pind2 (jnp.array): mask indices for second view's prediction.
      tind1 (jnp.array): mask indices for first view's projection.
      tind2 (jnp.array): mask indices for second view's projection.
      temperature (float): the temperature to use for the NCE loss.
      use_replicator_loss (bool): use cross-replica samples.
      local_negatives (bool): whether to include local negatives

    Returns:
      A single scalar loss for the XT-NCE objective.

    """
    batch_size = pred1.shape[0]
    num_rois = pred1.shape[1]
    feature_dim = pred1.shape[-1]
    infinity_proxy = 1e9  # Used for masks to proxy a very large number.

    def make_same_obj(
        ind_0: NDArray[np.int64], ind_1: NDArray[np.int64]
    ) -> NDArray[np.float32]:
        same_obj: NDArray[np.bool_] = np.equal(
            ind_0.reshape([batch_size, num_rois, 1]),
            ind_1.reshape([batch_size, 1, num_rois]),
        )
        same_obj2: NDArray[np.float32] = np.expand_dims(
            same_obj.astype(np.float32), axis=2
        )
        return same_obj2

    same_obj_aa = make_same_obj(pind1, tind1)
    same_obj_ab = make_same_obj(pind1, tind2)
    same_obj_ba = make_same_obj(pind2, tind1)
    same_obj_bb = make_same_obj(pind2, tind2)

    # L2 normalize the tensors to use for the cosine-similarity
    pred1 = pred1 / npl.norm(pred1, ord=2, axis=-1, keepdims=True)
    pred2 = pred2 / npl.norm(pred2, ord=2, axis=-1, keepdims=True)
    target1 = target1 / npl.norm(target1, ord=2, axis=-1, keepdims=True)
    target2 = target2 / npl.norm(target2, ord=2, axis=-1, keepdims=True)

    target1_large = target1
    target2_large = target2
    labels_local = np.eye(batch_size)
    # labels_ext = hk.one_hot(np.arange(batch_size), batch_size * 2)

    labels_local = np.expand_dims(np.expand_dims(labels_local, axis=2), axis=1)
    # labels_ext = np.expand_dims(np.expand_dims(labels_ext, axis=2), axis=1)

    # Do our matmuls and mask out appropriately.
    logits_aa = np.einsum("abk,uvk->abuv", pred1, target1_large) / temperature
    logits_bb = np.einsum("abk,uvk->abuv", pred2, target2_large) / temperature
    logits_ab = np.einsum("abk,uvk->abuv", pred1, target2_large) / temperature
    logits_ba = np.einsum("abk,uvk->abuv", pred2, target1_large) / temperature

    labels_aa = labels_local * same_obj_aa
    labels_ab = labels_local * same_obj_ab
    labels_ba = labels_local * same_obj_ba
    labels_bb = labels_local * same_obj_bb

    logits_aa = logits_aa - infinity_proxy * labels_local * same_obj_aa
    logits_bb = logits_bb - infinity_proxy * labels_local * same_obj_bb
    labels_aa = 0.0 * labels_aa
    labels_bb = 0.0 * labels_bb
    if not local_negatives:
        logits_aa = logits_aa - infinity_proxy * labels_local * (1 - same_obj_aa)
        logits_ab = logits_ab - infinity_proxy * labels_local * (1 - same_obj_ab)
        logits_ba = logits_ba - infinity_proxy * labels_local * (1 - same_obj_ba)
        logits_bb = logits_bb - infinity_proxy * labels_local * (1 - same_obj_bb)

    labels_abaa = np.concatenate([labels_ab, labels_aa], axis=2)
    labels_babb = np.concatenate([labels_ba, labels_bb], axis=2)

    labels_0 = np.reshape(labels_abaa, [batch_size, num_rois, -1])
    labels_1 = np.reshape(labels_babb, [batch_size, num_rois, -1])

    num_positives_0 = np.sum(labels_0, axis=-1, keepdims=True)
    num_positives_1 = np.sum(labels_1, axis=-1, keepdims=True)

    labels_0 = labels_0 / np.maximum(num_positives_0, 1)
    labels_1 = labels_1 / np.maximum(num_positives_1, 1)

    obj_area_0 = np.sum(make_same_obj(pind1, pind1), axis=(2, 3))
    obj_area_1 = np.sum(make_same_obj(pind2, pind2), axis=(2, 3))

    weights_0 = np.greater(num_positives_0[..., 0], 1e-3).astype("float32")
    weights_0 = weights_0 / obj_area_0
    weights_1 = np.greater(num_positives_1[..., 0], 1e-3).astype("float32")
    weights_1 = weights_1 / obj_area_1

    logits_abaa = np.concatenate([logits_ab, logits_aa], axis=2)
    logits_babb = np.concatenate([logits_ba, logits_bb], axis=2)

    logits_abaa = np.reshape(logits_abaa, [batch_size, num_rois, -1])
    logits_babb = np.reshape(logits_babb, [batch_size, num_rois, -1])

    # return labels_0, logits_abaa, weights_0, labels_1, logits_babb, weights_1

    loss_a = manual_cross_entropy(labels_0, logits_abaa, weights_0)
    loss_b = manual_cross_entropy(labels_1, logits_babb, weights_1)
    loss = loss_a + loss_b

    return loss


def manual_cross_entropy(
    labels: NDArray[np.float32],
    logits: NDArray[np.float32],
    weight: NDArray[np.float32],
) -> float:
    ce = -weight * np.sum(labels * sps.log_softmax(logits, axis=-1), axis=-1)
    mean: float = np.mean(ce)
    return mean
