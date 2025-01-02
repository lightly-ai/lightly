import torch
import torch.nn.functional as F
from torch import Tensor
from torch import distributed as dist
from torch import distributed as torch_dist
from torch.nn import Module


class DetConSLoss:
    """Implementation of the DetConS loss. [2]_

    The inputs are two-fold:

    - Two latent representations of the same batch under different views, as generated\
        by SimCLR [3]_ and additional pooling over the regions of the segmentation.
    - Two integer masks that indicate the regions of the segmentation that were used\
        for pooling.

    For calculating the contrastive loss, regions under the same mask in the same image
    (under a different view) are considered as positives and everything else as 
    negatives. With :math:`v_m` and :math:`v_{m'}'` being the pooled feature maps under 
    mask :math:`m` and :math:`m'` respectively, and additionally scaled to a norm of 
    :math:`\\frac{1}{\\sqrt{\\tau}}`, the formula for the contrastive loss is

    .. math::
        \\mathcal{L} = \sum_{m}\sum_{m'} \mathbb{1}_{m, m'} \\left[ - \\log\
            \\frac{\\exp(v_m \\cdot v_{m'}')}{\\exp(v_m \\cdot v_{m'}') +\
            \\sum_{n}\\exp (v_m \\cdot v_{m'}')} \\right]

    where :math:`\\mathbb{1}_{m, m'}` is 1 if the masks are the same and 0 otherwise.

    References:
        .. [2] DetCon https://arxiv.org/abs/2103.10957
        .. [3] SimCLR https://arxiv.org/abs/2002.05709

    Attributes:
        temperature:
            The temperature :math:`\\tau` in the contrastive loss.
        gather_distributed:
            If True, the similarity matrix is gathered across all GPUs before the loss
            is calculated. Else, the loss is calculated on each GPU separately.
    """

    def __init__(
        self, temperature: float = 0.1, gather_distributed: bool = True
    ) -> None:
        self.detconbloss = DetConBLoss(
            temperature=temperature, gather_distributed=gather_distributed
        )

    def forward(
        self, view0: Tensor, view1: Tensor, mask_view0: Tensor, mask_view1: Tensor
    ) -> Tensor:
        """Calculate the contrastive loss under the same mask in the same image.

        The tensor shapes and value ranges are given by variables :math:`B, M, D, N`,
        where :math:`B` is the batch size, :math:`M` is the sampled number of image
        masks / regions, :math:`D` is the embedding size and :math:`N` is the total
        number of masks.

        Args:
            view0: Mask-pooled output for the first view, a float tensor of shape
                :math:`(B, M, D)`.
            pred_view1: Mask-pooled output for the second view, a float tensor of shape
                :math:`(B, M, D)`.
            mask_view0: Indices corresponding to the sampled masks for the first view,
                an integer tensor of shape :math:`(B, M)` with (possibly repeated)
                indices in the range :math:`[0, N)`.
            mask_view1: Indices corresponding to the sampled masks for the second view,
                an integer tensor of shape (B, M) with (possibly repeated) indices in
                the range :math:`[0, N)`.

        Returns:
            A scalar float tensor containing the contrastive loss.
        """
        loss: Tensor = self.detconbloss(
            view0, view1, view0, view1, mask_view0, mask_view1
        )
        return loss


class DetConBLoss(Module):
    """Implementation of the DetConB loss. [0]_

    The inputs are three-fold:

    - Two latent representations of the same batch under different views, as generated\
        by BYOL's [1]_ prediction branch and additional pooling over the regions of\
        the segmentation.
    - Two latent representations of the same batch under different views, as generated\
        by BYOL's target branch and additional pooling over the regions of the\
        segmentation.
    - Two integer masks that indicate the regions of the segmentation that were used\
        for pooling.

    For calculating the contrastive loss, regions under the same mask in the same image
    (under a different view) are considered as positives and everything else as 
    negatives. With :math:`v_m` and :math:`v_{m'}'` being the pooled feature maps under 
    mask :math:`m` and :math:`m'` respectively, and additionally scaled to a norm of 
    :math:`\\frac{1}{\\sqrt{\\tau}}`, the formula for the contrastive loss is

    .. math::
        \\mathcal{L} = \sum_{m}\sum_{m'} \mathbb{1}_{m, m'} \\left[ - \\log \\frac{\\exp(v_m \\cdot v_{m'}')}{\\exp(v_m \\cdot v_{m'}') + \\sum_{n}\\exp (v_m \\cdot v_{m'}')} \\right]

    where :math:`\\mathbb{1}_{m, m'}` is 1 if the masks are the same and 0 otherwise.
    Since :math:`v_m` and :math:`v_{m'}'` stem from different branches, the loss is
    symmetrized by also calculating the loss with the roles of the views reversed. [1]_

    References:
        .. [0] DetCon https://arxiv.org/abs/2103.10957
        .. [1] BYOL https://arxiv.org/abs/2006.07733

    Attributes:
        temperature:
            The temperature :math:`\\tau` in the contrastive loss.
        gather_distributed:
            If True, the similarity matrix is gathered across all GPUs before the loss
            is calculated. Else, the loss is calculated on each GPU separately.
    """

    def __init__(
        self, temperature: float = 0.1, gather_distributed: bool = True
    ) -> None:
        super().__init__()
        self.eps = 1e-8
        self.temperature = temperature
        self.gather_distributed = gather_distributed
        self.eps = 1e-11

        if abs(self.temperature) < self.eps:
            raise ValueError(
                "Illegal temperature: abs({}) < 1e-8".format(self.temperature)
            )
        if self.gather_distributed and not torch_dist.is_available():
            raise ValueError(
                "gather_distributed is True but torch.distributed is not available. "
                "Please set gather_distributed=False or install a torch version with "
                "distributed support."
            )

    def forward(
        self,
        pred_view0: Tensor,
        pred_view1: Tensor,
        target_view0: Tensor,
        target_view1: Tensor,
        mask_view0: Tensor,
        mask_view1: Tensor,
    ) -> Tensor:
        """Calculate the contrastive loss under the same mask in the same image.

        The tensor shapes and value ranges are given by variables :math:`B, M, D, N`,
        where :math:`B` is the batch size, :math:`M` is the sampled number of image
        masks / regions, :math:`D` is the embedding size and :math:`N` is the total
        number of masks.

        Args:
            pred_view0: Mask-pooled output of the prediction branch for the first view,
                a float tensor of shape :math:`(B, M, D)`.
            pred_view1: Mask-pooled output of the prediction branch for the second view,
                a float tensor of shape :math:`(B, M, D)`.
            target_view0: Mask-pooled output of the target branch for the first view,
                a float tensor of shape :math:`(B, M, D)`.
            target_view1: Mask-pooled output of the target branch for the second view,
                a float tensor of shape :math:`(B, M, D)`.
            mask_view0: Indices corresponding to the sampled masks for the first view,
                an integer tensor of shape :math:`(B, M)` with (possibly repeated)
                indices in the range :math:`[0, N)`.
            mask_view1: Indices corresponding to the sampled masks for the second view,
                an integer tensor of shape (B, M) with (possibly repeated) indices in
                the range :math:`[0, N)`.

        Returns:
            A scalar float tensor containing the contrastive loss.
        """
        b, m, d = pred_view0.size()
        infinity_proxy = 1e9

        # gather distributed
        if not self.gather_distributed or dist.get_world_size() < 2:
            target_view0_large = target_view0
            target_view1_large = target_view1
            labels_local = torch.eye(b, device=pred_view0.device)
            labels_ext = torch.cat(
                [
                    torch.eye(b, device=pred_view0.device),
                    torch.zeros_like(labels_local),
                ],
                dim=1,
            )
        else:
            target_view0_large = torch.cat(dist.gather(target_view0), dim=0)
            target_view1_large = torch.cat(dist.gather(target_view1), dim=0)
            replica_id = dist.get_rank()
            labels_idx = torch.arange(b, device=pred_view0.device) + replica_id * b
            enlarged_b = b * dist.get_world_size()
            labels_local = F.one_hot(labels_idx, num_classes=enlarged_b)
            labels_ext = F.one_hot(labels_idx, num_classes=2 * enlarged_b)

        # normalize
        pred_view0 = F.normalize(pred_view0, p=2, dim=2)
        pred_view1 = F.normalize(pred_view1, p=2, dim=2)
        target_view0_large = F.normalize(target_view0_large, p=2, dim=2)
        target_view1_large = F.normalize(target_view1_large, p=2, dim=2)

        labels_local = labels_local[:, None, :, None]
        labels_ext = labels_ext[:, None, :, None]

        # calculate similarity matrices
        logits_aa = (
            torch.einsum("abk,uvk->abuv", pred_view0, target_view0_large)
            / self.temperature
        )
        logits_bb = (
            torch.einsum("abk,uvk->abuv", pred_view1, target_view1_large)
            / self.temperature
        )
        logits_ab = (
            torch.einsum("abk,uvk->abuv", pred_view0, target_view1_large)
            / self.temperature
        )
        logits_ba = (
            torch.einsum("abk,uvk->abuv", pred_view1, target_view0_large)
            / self.temperature
        )

        # determine where the masks are the same
        same_mask_aa = _same_mask(mask_view0, mask_view0)
        same_mask_bb = _same_mask(mask_view1, mask_view1)
        same_mask_ab = _same_mask(mask_view0, mask_view1)
        same_mask_ba = _same_mask(mask_view1, mask_view0)

        # remove similarities between the same masks
        labels_aa = labels_local * same_mask_aa
        labels_bb = labels_local * same_mask_bb
        labels_ab = labels_local * same_mask_ab
        labels_ba = labels_local * same_mask_ba

        logits_aa = logits_aa - infinity_proxy * labels_aa
        logits_bb = logits_bb - infinity_proxy * labels_bb
        labels_aa = 0.0 * labels_aa
        labels_bb = 0.0 * labels_bb

        labels_abaa = torch.cat([labels_ab, labels_aa], dim=2)
        labels_babb = torch.cat([labels_ba, labels_bb], dim=2)

        labels_0 = labels_abaa.view(b, m, -1)
        labels_1 = labels_babb.view(b, m, -1)

        num_positives_0 = torch.sum(labels_0, dim=-1, keepdim=True)
        num_positives_1 = torch.sum(labels_1, dim=-1, keepdim=True)

        labels_0 = labels_0 / torch.maximum(num_positives_0, torch.tensor(1))
        labels_1 = labels_1 / torch.maximum(num_positives_1, torch.tensor(1))

        obj_area_0 = torch.sum(_same_mask(mask_view0, mask_view0), dim=(2, 3))
        obj_area_1 = torch.sum(_same_mask(mask_view1, mask_view1), dim=(2, 3))

        weights_0 = torch.gt(num_positives_0[..., 0], 1e-3).float()
        weights_0 = weights_0 / obj_area_0
        weights_1 = torch.gt(num_positives_1[..., 0], 1e-3).float()
        weights_1 = weights_1 / obj_area_1

        logits_abaa = torch.cat([logits_ab, logits_aa], dim=2)
        logits_babb = torch.cat([logits_ba, logits_bb], dim=2)

        logits_abaa = logits_abaa.view(b, m, -1)
        logits_babb = logits_babb.view(b, m, -1)

        loss_a = torch_manual_cross_entropy(labels_0, logits_abaa, weights_0)
        loss_b = torch_manual_cross_entropy(labels_1, logits_babb, weights_1)
        loss = loss_a + loss_b
        return loss


def _same_mask(mask0: Tensor, mask1: Tensor) -> Tensor:
    return (mask0[:, :, None] == mask1[:, None, :]).float()[:, :, None, :]


def torch_manual_cross_entropy(
    labels: Tensor, logits: Tensor, weight: Tensor
) -> Tensor:
    ce = -weight * torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)
    return torch.mean(ce)
