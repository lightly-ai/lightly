# Code adapted from https://github.com/keyu-tian/SparK/blob/main/pretrain/models/resnet.py
from typing import Union

import torch
import torch.nn as nn
from torch import List, Optional, Tensor

from lightly.models.modules.sparse.spark import (
    SparseAvgPooling,
    SparseBatchNorm2d,
    SparseConv2d,
    SparseMaxPooling,
    SparseSyncBatchNorm2d,
    SparseConvNeXtBlock,
    SparseConvNeXtLayerNorm
)

from torchvision.models.convnext import CNBlock, LayerNorm2d

from torchvision.models import ResNet, ConvNeXt


class SparseMask:
    """
    Basic class to store the mask for the sparse model.
    """

    def __init__(self):
        self.mask: Union[Tensor, None] = None


class SparseEncoder(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        input_size: int,
        downsample_ratio: Optional[int] = None,
    ):
        """Sparse encoder as used by SparK [0]

        Default params are the ones explained in the original code base. The backbone is assumed
        to follow the same API as the ResNet or ConvNext models from torchvision.
        [0] Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling https://arxiv.org/abs/2301.03580

        Attributes:
            backbone:
                Backbone model to extract features from images. Should have both
                the methods get_downsample_ratio() and get_feature_map_channels()
                implemented.
            input_size:
                Size of the input image.
            downsample_ratio:

        """
        super().__init__()
        self.input_size = input_size
        if downsample_ratio is None:
            self.downsample_ratio = self.get_downsample_ratio(backbone)
        else:
            self.downsample_ratio = downsample_ratio

        self.enc_feat_map_chs = (
            self.get_feature_map_channels(backbone),
        )
        self.sparse_mask = SparseMask()
        self.sparse_backbone = self.dense_model_to_sparse(
            m=backbone, sparse_mask=self.sparse_mask
        )

    def forward(
        self, x: torch.Tensor, mask: torch.BoolTensor, hierarchical: bool = False
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass of the sparse encoder.

        Args:
            x: The input tensor.
            mask: The mask to apply to the input tensor. This should be a binary mask the
                size of the smallest resolution feature map [0] of the backbone model.

        Returns:
            The output tensor.

        [0] SparK codebase https://github.com/keyu-tian/SparK/tree/main/pretrain#some-details-how-we-mask-images-and-how-to-set-the-patch-size
        """
        assert mask.shape == (
            1,
            1,
            self.input_size // self.downsample_ratio,
            self.input_size // self.downsample_ratio,
        ), "Mask shape must be (1, 1, H // downsample_ratio, W // downsample_ratio)"
        self.sparse_mask.mask = mask

        ls = []

        if isinstance(self.sparse_backbone, ConvNeXt):
            if hierarchical:
                for i in range(0,8,2):
                    x = self.sparse_backbone.features[i](x)
                    x = self.sparse_backbone.features[i+1](x)
                    ls.append(x)
            else:
                x = self.sparse_backbone.avgpool(x)
                x = self.sparse_backbone.classifier(x)
                return x

        elif isinstance(self.sparse_backbone, ResNet):
            x = self.sparse_backbone.conv1(x)
            x = self.sparse_backbone.bn1(x)
            x = self.sparse_backbone.act1(x)
            x = self.sparse_backbone.maxpool(x)

            if hierarchical:
                ls = []
                x = self.sparse_backbone.layer1(x)
                ls.append(x)
                x = self.sparse_backbone.layer2(x)
                ls.append(x)
                x = self.sparse_backbone.layer3(x)
                ls.append(x)
                x = self.sparse_backbone.layer4(x)
                ls.append(x)
                return ls
            else:
                x = self.sparse_backbone.global_pool(x)
                x = self.sparse_backbone.fc(x)
                return x
            
        else:
            raise NotImplementedError("Backbone not supported")

    def get_downsample_ratio(self, backbone: nn.Module) -> int:
        """
        Try to get the downsample ratio of the backbone model.

        Returns:
            The downsample ratio of the backbone model.
        """
        try:
            return backbone.get_downsample_ratio()
        except AttributeError:
            x = torch.randn(1, 3, self.input_size, self.input_size)
            out = nn.Sequential(*list(backbone.children())[:-2])(x)

            return self.input_size // out.shape[-1]


    def dense_model_to_sparse(
        self, m: nn.Module, sparse_mask: SparseMask, sync_batch_norm: bool = False
    ) -> nn.Module:
        """
        Convert a dense model to a sparse model.

        Args:
            m: The dense model to convert.
            sparse_mask: The sparse mask to use for the conversion.
            sync_batch_norm: Whether to convert BatchNorm2d to SyncBatchNorm.

        Returns:
            The sparse model.
        """
        with torch.no_grad():
            oup = m
            if isinstance(m, nn.Conv2d):
                m: nn.Conv2d
                bias = m.bias is not None
                oup = SparseConv2d(
                    m.in_channels,
                    m.out_channels,
                    kernel_size=m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    groups=m.groups,
                    bias=bias,
                    padding_mode=m.padding_mode,
                )
                oup.sparse_mask = sparse_mask
                oup.weight.copy_(m.weight)
                if bias:
                    oup.bias.copy_(m.bias)
            elif isinstance(m, nn.MaxPool2d):
                m: nn.MaxPool2d
                oup = SparseMaxPooling(
                    m.kernel_size,
                    stride=m.stride,
                    padding=m.padding,
                    dilation=m.dilation,
                    return_indices=m.return_indices,
                    ceil_mode=m.ceil_mode,
                )
                oup.sparse_mask = sparse_mask
            elif isinstance(m, nn.AvgPool2d):
                m: nn.AvgPool2d
                oup = SparseAvgPooling(
                    m.kernel_size,
                    m.stride,
                    m.padding,
                    ceil_mode=m.ceil_mode,
                    count_include_pad=m.count_include_pad,
                    divisor_override=m.divisor_override,
                )
                oup.sparse_mask = sparse_mask
            elif isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m: nn.BatchNorm2d
                oup = (
                    SparseSyncBatchNorm2d
                    if isinstance(m, nn.SyncBatchNorm)
                    else SparseBatchNorm2d
                )(
                    m.weight.shape[0],
                    eps=m.eps,
                    momentum=m.momentum,
                    affine=m.affine,
                    track_running_stats=m.track_running_stats,
                )
                oup.sparse_mask = sparse_mask
                oup.weight.copy_(m.weight)
                oup.bias.copy_(m.bias)
                oup.running_mean.copy_(m.running_mean)
                oup.running_var.copy_(m.running_var)
                oup.num_batches_tracked.copy_(m.num_batches_tracked)
                if hasattr(m, "qconfig"):
                    oup.qconfig = m.qconfig
            elif isinstance(m, CNBlock):
                m: CNBlock
                oup = SparseConvNeXtBlock(
                    m.weight.shape[0],
                    m.layer_scale.
                )
            elif isinstance(m, (nn.Conv1d,)):
                raise NotImplementedError

            for name, child in m.named_children():
                oup.add_module(
                    name, self.dense_model_to_sparse(child, sparse_mask=sparse_mask)
                )
            del m
            return oup
