import torch
import torch.nn.functional as F
from torch import Tensor
from torch import distributed as torch_dist
from torch.nn import Module

import lightly.utils.dist as lightly_dist


class DetConSLoss(Module):
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
        \\mathcal{L} = \\sum_{m}\\sum_{m'} \\mathbb{1}_{m, m'} \\left[ - \\log\
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
        super().__init__()
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
        \\mathcal{L} = \\sum_{m}\\sum_{m'} \\mathbb{1}_{m, m'} \\left[ - \\log \
        \\frac{\\exp(v_m \\cdot v_{m'}')}{\\exp(v_m \\cdot v_{m'}') + \\sum_{n}\\exp \
        (v_m \\cdot v_{m'}')} \\right]

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
        if abs(self.temperature) < self.eps:
            raise ValueError(f"Illegal temperature: abs({self.temperature}) < 1e-8")
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
        if self.gather_distributed and lightly_dist.world_size() > 1:
            target_view0_large = torch.cat(lightly_dist.gather(target_view0), dim=0)
            target_view1_large = torch.cat(lightly_dist.gather(target_view1), dim=0)
            replica_id = lightly_dist.rank()
            labels_idx = torch.arange(b, device=pred_view0.device) + replica_id * b
            enlarged_b = b * lightly_dist.world_size()
            labels_local = F.one_hot(labels_idx, num_classes=enlarged_b)
        else:
            target_view0_large = target_view0
            target_view1_large = target_view1
            labels_local = torch.eye(b, device=pred_view0.device)
            enlarged_b = b

        # normalize
        pred_view0 = F.normalize(pred_view0, p=2, dim=2)
        pred_view1 = F.normalize(pred_view1, p=2, dim=2)
        target_view0_large = F.normalize(target_view0_large, p=2, dim=2)
        target_view1_large = F.normalize(target_view1_large, p=2, dim=2)

        ### Expand Labels ###
        # labels_local at this point only points towards the diagonal of the batch, i.e.
        # indicates to compare between the same samples across views.
        labels_local = labels_local[:, None, :, None]  # (b, 1, b * world_size, 1)

        ### Calculate Similarity Matrices ###
        # tensors of shape (b, m, b * world_size, m), indicating similarities between regions across
        # views and samples in the batch
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

        ### Find Corresponding Regions Across Views ###
        same_mask_aa = _same_mask(mask_view0, mask_view0)
        same_mask_bb = _same_mask(mask_view1, mask_view1)
        same_mask_ab = _same_mask(mask_view0, mask_view1)
        same_mask_ba = _same_mask(mask_view1, mask_view0)

        ### Remove Similarities Between Corresponding Views But Different Regions ###
        # labels_local initially compared all features across views, but we only want to
        # compare the same regions across views.
        # (b, 1, b * world_size, 1) * (b, m, 1, m) -> (b, m, b * world_size, m)
        labels_aa = labels_local * same_mask_aa
        labels_bb = labels_local * same_mask_bb
        labels_ab = labels_local * same_mask_ab
        labels_ba = labels_local * same_mask_ba

        ### Remove Logits And Lables Between The Same View ###
        logits_aa = logits_aa - infinity_proxy * labels_aa
        logits_bb = logits_bb - infinity_proxy * labels_bb
        labels_aa = 0.0 * labels_aa
        labels_bb = 0.0 * labels_bb

        ### Arrange Labels ###
        # (b, m, b * world_size * 2, m)
        labels_abaa = torch.cat([labels_ab, labels_aa], dim=2)
        labels_babb = torch.cat([labels_ba, labels_bb], dim=2)
        # (b, m, b * world_size * 2 * m)
        labels_0 = labels_abaa.view(b, m, -1)
        labels_1 = labels_babb.view(b, m, -1)

        ### Count Number of Positives For Every Region (per sample) ###
        num_positives_0 = torch.sum(labels_0, dim=-1, keepdim=True)
        num_positives_1 = torch.sum(labels_1, dim=-1, keepdim=True)

        ### Scale The Labels By The Number of Positives To Weight Loss Value ###
        labels_0 = labels_0 / torch.maximum(num_positives_0, torch.tensor(1))
        labels_1 = labels_1 / torch.maximum(num_positives_1, torch.tensor(1))

        ### Count How Many Overlapping Regions We Have Across Views ###
        obj_area_0 = torch.sum(same_mask_aa, dim=(2, 3))
        obj_area_1 = torch.sum(same_mask_bb, dim=(2, 3))
        # make sure we don't divide by zero
        obj_area_0 = torch.maximum(obj_area_0, torch.tensor(self.eps))
        obj_area_1 = torch.maximum(obj_area_1, torch.tensor(self.eps))

        ### Calculate Weights For The Loss ###
        # last dim of num_positives is anyway 1, from the torch.sum above
        weights_0 = torch.gt(num_positives_0.squeeze(-1), 1e-3).float()
        weights_0 = weights_0 / obj_area_0
        weights_1 = torch.gt(num_positives_1.squeeze(-1), 1e-3).float()
        weights_1 = weights_1 / obj_area_1

        ### Arrange Logits ###
        logits_abaa = torch.cat([logits_ab, logits_aa], dim=2)
        logits_babb = torch.cat([logits_ba, logits_bb], dim=2)
        logits_abaa = logits_abaa.view(b, m, -1)
        logits_babb = logits_babb.view(b, m, -1)

        # return labels_0, logits_abaa, weights_0, labels_1, logits_babb, weights_1

        ### Derive Cross Entropy Loss ###
        # targets/labels are are a weighted float tensor of same shape as logits,
        # which is why we can't use F.cross_entropy (expects integer targets)
        loss_a = _torch_manual_cross_entropy(labels_0, logits_abaa, weights_0)
        loss_b = _torch_manual_cross_entropy(labels_1, logits_babb, weights_1)
        loss = loss_a + loss_b
        return loss


def _same_mask(mask0: Tensor, mask1: Tensor) -> Tensor:
    """Find equal masks/regions across views of the same image.

    Args:
        mask0: Indices corresponding to the sampled masks for the first view,
            an integer tensor of shape :math:`(B, M)` with (possibly repeated)
            indices in the range :math:`[0, N)`.
        mask1: Indices corresponding to the sampled masks for the second view,
            an integer tensor of shape (B, M) with (possibly repeated) indices in
            the range :math:`[0, N)`.

    Returns:
        Tensor: A float tensor of shape :math:`(B, M, 1, M)` where the first :math:`M`
            dimensions is for the regions/masks of the first view and the last :math:`M`
            dimensions is for the regions/masks of the second view. For every sample
            :math:`k` in the batch (separately), the tensor is effectively a 2D index
            matrix where the entry :math:`(k, i, :, j)` is 1 if the masks :math:`mask0(k, i)`
            and :math:`mask1(k, j)'` are the same and 0 otherwise.
    """
    return (mask0[:, :, None] == mask1[:, None, :]).float()[:, :, None, :]


def _torch_manual_cross_entropy(
    labels: Tensor, logits: Tensor, weight: Tensor
) -> Tensor:
    ce = -weight * torch.sum(labels * F.log_softmax(logits, dim=-1), dim=-1)
    return torch.mean(ce)
