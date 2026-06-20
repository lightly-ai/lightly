# Code adapted from https://github.com/keyu-tian/SparK/blob/main/

import math
from typing import List, Literal, Tuple, Union
from pprint import pformat
import sys

import torch
import torch.nn as nn
from lightly.models.sparse.encoder import SparseResnet
from lightly.models.modules.sparse.spark import SparseBatchNorm2d, SparseSyncBatchNorm2d, SparseConvNeXtLayerNorm

def is_pow2n(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


class UNetBlock(nn.Module):
    def __init__(self, cin, cout, bn2d):
        """
        a UNet block with 2x up sampling
        """
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(
            cin, cin, kernel_size=4, stride=2, padding=1, bias=True
        )
        self.conv = nn.Sequential(
            nn.Conv2d(cin, cin, kernel_size=3, stride=1, padding=1, bias=False),
            bn2d(cin),
            nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1, bias=False),
            bn2d(cout),
        )

    def forward(self, x):
        x = self.up_sample(x)
        return self.conv(x)


class LightDecoder(nn.Module):
    def __init__(self, up_sample_ratio, width=768, sync_batch_norm=True):
        super().__init__()
        self.width = width
        assert is_pow2n(up_sample_ratio)
        n = round(math.log2(up_sample_ratio))
        channels = [self.width // 2**i for i in range(n + 1)]
        bn2d = nn.SyncBatchNorm if sync_batch_norm else nn.BatchNorm2d
        self.dec = nn.ModuleList(
            [
                UNetBlock(cin, cout, bn2d)
                for (cin, cout) in zip(channels[:-1], channels[1:])
            ]
        )
        self.proj = nn.Conv2d(channels[-1], 3, kernel_size=1, stride=1, bias=True)

        self.initialize()

    def forward(self, to_dec: List[torch.Tensor]):
        x = 0
        for i, d in enumerate(self.dec):
            if i < len(to_dec) and to_dec[i] is not None:
                x = x + to_dec[i]
            x = self.dec[i](x)
        return self.proj(x)

    def extra_repr(self) -> str:
        return f"width={self.width}"

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(
                m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.SyncBatchNorm)
            ):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

class SparK(nn.Module):
    """
    The SparK model as used by SparK [0]

    Default params are the ones explained in the original code base. The backbone is assumed
    to follow the same API as the ResNet models from torchvision or a ConvNext model.
    [0] Designing BERT for Convolutional Networks: Sparse and Hierarchical Masked Modeling https://arxiv.org/abs/2301.03580

    Attributes:
        sparse_encoder:
            Sparse encoder to extract features from images. Should have both
            the methods get_downsample_ratio() and get_feature_map_channels()
            implemented.
        dense_decoder:
            Dense decoder to reconstruct the image from the sparse features.
        mask_ratio:
            Ratio of the image to mask. Default: 0.6
        densify_norm:
            Type of normalization to use for densification. Default: 'bn'
        sbn:
            Whether to use SyncBatchNorm. Default: False
    """
    def __init__(
            self, sparse_encoder: SparseResnet, dense_decoder: LightDecoder,
            mask_ratio: float=0.6, densify_norm: Literal['batch_norm', 'layer_norm', 'identity']='bn', sbn: bool=False,
    ):
        super().__init__()
        input_size, downsample_ratio = sparse_encoder.input_size, sparse_encoder.downsample_ratio
        self.downsample_ratio = downsample_ratio
        self.fmap_h, self.fmap_w = input_size // downsample_ratio, input_size // downsample_ratio
        self.mask_ratio = mask_ratio
        self.len_keep = round(self.fmap_h * self.fmap_w * (1 - mask_ratio))
        
        self.sparse_encoder = sparse_encoder
        self.dense_decoder = dense_decoder
        
        self.sbn = sbn
        self.hierarchy = len(sparse_encoder.enc_feat_map_chs)
        self.densify_norm_str = densify_norm.lower()
        self.densify_norms = nn.ModuleList()
        self.densify_projs = nn.ModuleList()
        self.mask_tokens = nn.ParameterList()
        
        e_widths, d_width = self.sparse_encoder.enc_feat_map_chs, self.dense_decoder.width
        e_widths: List[int]
        for i in range(self.hierarchy):
            e_width = e_widths.pop()
            p = nn.Parameter(torch.zeros(1, e_width, 1, 1))
            torch.nn.init.trunc_normal_(p, mean=0, std=.02, a=-.02, b=.02)
            self.mask_tokens.append(p)
            
            
            if self.densify_norm_str == 'batch_norm':
                densify_norm = (SparseSyncBatchNorm2d if self.sbn else SparseBatchNorm2d)(e_width)
            elif self.densify_norm_str == 'layer_norm':
                densify_norm = SparseConvNeXtLayerNorm(e_width, data_format='channels_first', sparse=True)
            else:
                densify_norm = nn.Identity()
            self.densify_norms.append(densify_norm)
            
            if i == 0 and e_width == d_width:
                densify_proj = nn.Identity()   
                print(f'[SparK.__init__, densify {i+1}/{self.hierarchy}]: use nn.Identity() as densify_proj')
            else:
                kernel_size = 1 if i <= 0 else 3
                densify_proj = nn.Conv2d(e_width, d_width, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, bias=True)
                print(f'[SparK.__init__, densify {i+1}/{self.hierarchy}]: densify_proj(ksz={kernel_size}, #para={sum(x.numel() for x in densify_proj.parameters()) / 1e6:.2f}M)')
            self.densify_projs.append(densify_proj)
            d_width //= 2
        
        print(f'[SparK.__init__] dims of mask_tokens={tuple(p.numel() for p in self.mask_tokens)}')
        
    
    def mask(self, B: int, device, generator=None):
        """
        Generate a mask for the input tensor

        Attributes:
            B:
                Batch size
            device:
                Device to put the mask on
            generator:
                Random number generator
        """
        h, w = self.fmap_h, self.fmap_w
        idx = torch.rand(B, h * w, generator=generator).argsort(dim=1)
        idx = idx[:, :self.len_keep].to(device)  # (B, len_keep)
        return torch.zeros(B, h * w, dtype=torch.bool, device=device).scatter_(dim=1, index=idx, value=True).view(B, 1, h, w)
    
    def forward(self, inp_bchw: torch.Tensor, active_b1ff: torch.BoolTensor=None, return_loss: bool=True) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the SparK model

        Attributes:
            inp_bchw:
                Input tensor
            active_b1ff:
                Active mask
            return_loss:
                Whether to return the loss
        """
        if active_b1ff is None:     
            active_b1ff: torch.BoolTensor = self.mask(inp_bchw.shape[0], inp_bchw.device)  
        active_b1hw = active_b1ff.repeat_interleave(self.downsample_ratio, 2).repeat_interleave(self.downsample_ratio, 3)  
        masked_bchw = inp_bchw * active_b1hw
        
        fea_bcffs: List[torch.Tensor] = self.sparse_encoder.forward(masked_bchw, active_b1ff, hierarchical=True)
        fea_bcffs.reverse()  
        
        cur_active = active_b1ff     
        to_dec = []
        for i, bcff in enumerate(fea_bcffs):  
            if bcff is not None:
                bcff = self.densify_norms[i](bcff)
                mask_tokens = self.mask_tokens[i].expand_as(bcff)
                bcff = torch.where(cur_active.expand_as(bcff), bcff, mask_tokens)   
                bcff: torch.Tensor = self.densify_projs[i](bcff)
            to_dec.append(bcff)
            cur_active = cur_active.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3) 
        
        rec_bchw = self.dense_decoder(to_dec)
        inp, rec = self.patchify(inp_bchw), self.patchify(rec_bchw)  
        mean = inp.mean(dim=-1, keepdim=True)
        var = (inp.var(dim=-1, keepdim=True) + 1e-6) ** .5
        inp = (inp - mean) / var
        l2_loss = ((rec - inp) ** 2).mean(dim=2, keepdim=False)    
        
        non_active = active_b1ff.logical_not().int().view(active_b1ff.shape[0], -1)  
        recon_loss = l2_loss.mul_(non_active).sum() / (non_active.sum() + 1e-8) 
        
        if return_loss:
            return recon_loss
        
        masked_bchw = inp_bchw * active_b1hw
        rec_bchw = self.unpatchify(rec * var + mean)
        rec_or_inp = torch.where(active_b1hw, inp_bchw, rec_bchw)
        return inp_bchw, masked_bchw, rec_or_inp

    
    def patchify(self, bchw):
        p = self.downsample_ratio
        h, w = self.fmap_h, self.fmap_w
        B, C = bchw.shape[:2]
        bchw = bchw.reshape(shape=(B, C, h, p, w, p))
        bchw = torch.einsum('bchpwq->bhwpqc', bchw)
        bln = bchw.reshape(shape=(B, h * w, C * p ** 2))  
        return bln
    
    def unpatchify(self, bln):
        p = self.downsample_ratio
        h, w = self.fmap_h, self.fmap_w
        B, C = bln.shape[0], bln.shape[-1] // p ** 2
        bln = bln.reshape(shape=(B, h, w, p, p, C))
        bln = torch.einsum('bhwpqc->bchpwq', bln)
        bchw = bln.reshape(shape=(B, C, h * p, w * p))
        return bchw
    
    def __repr__(self):
        return (
            f'\n'
            f'[SparK.config]: {pformat(self.get_config(), indent=2, width=250)}\n'
            f'[SparK.structure]: {super(SparK, self).__repr__().replace(SparK.__name__, "")}'
        )
    
    def get_config(self):
        return {
            'mask_ratio': self.mask_ratio,
            'densify_norm_str': self.densify_norm_str,
            'sbn': self.sbn, 'hierarchy': self.hierarchy,
            'sparse_encoder.input_size': self.sparse_encoder.input_size,
            'dense_decoder.width': self.dense_decoder.width,
        }
    
    def state_dict(self, destination=None, prefix='', keep_vars=False, with_config=False):
        state = super(SparK, self).state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if with_config:
            state['config'] = self.get_config()
        return state
    
    def load_state_dict(self, state_dict, strict=True):
        config: dict = state_dict.pop('config', None)
        incompatible_keys = super(SparK, self).load_state_dict(state_dict, strict=strict)
        if config is not None:
            for k, v in self.get_config().items():
                ckpt_v = config.get(k, None)
                if ckpt_v != v:
                    err = f'[SparseMIM.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={ckpt_v})'
                    if strict:
                        raise AttributeError(err)
                    else:
                        print(err, file=sys.stderr)
        return incompatible_keys