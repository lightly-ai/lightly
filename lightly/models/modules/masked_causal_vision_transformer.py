from typing import Callable, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from timm.layers import LayerType, Mlp, PatchEmbed
from timm.models import _manipulate
from timm.models.vision_transformer import Attention, Block, VisionTransformer
from torch import Tensor, jit
from torch.nn import GELU, LayerNorm, Module


# Type ignore because superclass has Any types.
class MaskedCausalAttention(Attention):  # type: ignore[misc]
    """Identical to timm.models.vision_transformer.Attention, but supports causal
    attention with masking.

    The implementation is based on AIM [0].

    - [0]: AIM, 2024, https://arxiv.org/abs/2401.08541
    """

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the attention layer.

        Args:
            x:
                Input tensor of shape (batch_size, sequence_length, channels).
            mask:
                Mask of shape (batch_size, sequence_length) indicating which tokens
                should be masked. Tokens where the mask is True will only be used for
                causal attention, while unmasked tokens are used for bidirectional
                attention. If the mask is None, all tokens are used for bidirectional
                attention.
        """
        B, N, C = x.shape
        attn_mask = self._get_attention_mask(x, mask=mask)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            # Type ignore because only new torch versions support this.
            x = F.scaled_dot_product_attention(  # type: ignore[attr-defined]
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            assert False, "Only fused attention is supported for now."
            # TODO: Implement non-fused attention.
            q = q * self.scale  # type: ignore[unreachable]
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _get_attention_mask(
        self, x: Tensor, mask: Optional[Tensor]
    ) -> Optional[Tensor]:
        """Generates an attention mask for causal attention.

        Args:
            x:
                Input tensor of shape (batch_size, sequence_length, channels).
            mask:
                Mask tensor of shape (batch_size, sequence_length) indicating which tokens
                should be masked.

        Returns:
            Attention mask of shape (batch_size, 1, sequence_length, sequence_length).
        """
        B, N = x.shape[:2]

        # Only apply causal attention if mask is not None. This is a bit hacky, but it
        # allows us to use bidirectional instead of causal attention during evaluation
        # and fine-tuning.
        attn_mask = None
        if mask is not None:
            attn_mask = x.new_ones(size=(B, N, N), dtype=torch.bool).tril(diagonal=0)
            # mask has shape (B, N)
            mask = (~mask).unsqueeze(1).expand(B, N, N).bool()
            attn_mask = torch.logical_or(attn_mask, mask)
            attn_mask = attn_mask.unsqueeze(1)  # (B, 1, N, N)
        return attn_mask


# Type ignore because superclass has Any types.
class MaskedCausalBlock(Block):  # type: ignore[misc]
    """Identical to timm.models.vision_transformer.Block, but uses PrefixCausalAttention
    instead of Attention.

    The implementation is based on AIM [0].

    - [0]: AIM, 2024, https://arxiv.org/abs/2401.08541
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[Module] = GELU,
        norm_layer: Type[Module] = LayerNorm,
        mlp_layer: Type[Module] = Mlp,
    ) -> None:
        """Initializes the MaskedCausalBlock with the specified parameters.

        Args:
            dim:
                Dimension of the input tokens.
            num_heads:
                Number of attention heads.
            mlp_ratio:
                Ratio of MLP hidden dim to embedding dim.
            qkv_bias:
                If True, add bias to the query, key, and value tensors.
            qk_norm:
                If True, apply layer normalization to queries and keys.
            proj_drop:
                Percentage of elements set to zero after the projection layer.
            attn_drop:
                Percentage of elements set to zero after the attention head.
            init_values:
                Initial values for the layer.
            drop_path:
                Drop path rate for the block.
            act_layer:
                Activation layer to use.
            norm_layer:
                Normalization layer to use.
            mlp_layer:
                MLP layer to use.
        """
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
        )
        self.attn = MaskedCausalAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the attention block.

        Args:
            x:
                Input tensor of shape (batch_size, sequence_length, channels).
            mask:
                Mask of shape (batch_size, sequence_length) indicating which tokens
                should be masked. Tokens where the mask is True will only be used for
                causal attention, while unmasked tokens are used for bidirectional
                attention. If the mask is None, all tokens are used for bidirectional
                attention.

        Returns:
            Output tensor after applying the attention block.
        """
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), mask=mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


# Type ignore because superclass has Any types.
class MaskedCausalVisionTransformer(VisionTransformer):  # type: ignore[misc]
    """Vision transformer with masked causal attention based on AIM [0].

    - [0]: AIM, 2024, https://arxiv.org/abs/2401.08541
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: str = "",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: str = "",
        embed_layer: Callable = PatchEmbed,  # type: ignore[type-arg]
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: Type[Module] = MaskedCausalBlock,
        mlp_layer: Type[Module] = Mlp,
    ) -> None:
        """Initializes the MaskedCausalVisionTransformer with the specified parameters.

        Args:
            img_size:
                Input image size.
            patch_size:
                Width and height of the image patches.
            in_chans:
                Number of image input channels.
            num_classes:
                Number of classes for the classification head.
            global_pool:
                Global pooling type.
            embed_dim:
                Embedding dimension.
            depth:
                Depth of the transformer.
            num_heads:
                Number of attention heads.
            mlp_ratio:
                Ratio of MLP hidden dim to embedding dim.
            qkv_bias:
                If True, add bias to the query, key, and value tensors.
            qk_norm:
                If True, apply layer normalization to queries and keys.
            init_values:
                Initial values for the layer.
            class_token:
                If True, add class token to the embeddings.
            no_embed_class:
                If True, do not embed class token.
            reg_tokens:
                Number of regularization tokens.
            pre_norm :
                If True, apply layer normalization before the transformer.
            fc_norm:
                If True, apply layer normalization to the final fully connected layer.
            dynamic_img_size:
                If True, dynamically adjust the image size.
            dynamic_img_pad:
                If True, dynamically pad the image.
            drop_rate:
                Percentage of elements set to zero after the dropout layer.
            pos_drop_rate:
                Percentage of elements set to zero after the positional dropout layer.
            patch_drop_rate:
                Percentage of elements set to zero after the patch dropout layer.
            proj_drop_rate:
                Percentage of elements set to zero after the projection dropout layer.
            attn_drop_rate:
                Percentage of elements set to zero after the attention head dropout.
            drop_path_rate:
                Drop path rate for the block.
            weight_init:
                Weight initialization method.
            embed_layer:
                Callable that creates the embedding layer.
            norm_layer:
                Normalization layer to use.
            act_layer:
                Activation layer to use.
            block_fn:
                Block function to use.
            mlp_layer:
                MLP layer to use.
        """
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            init_values=init_values,
            class_token=class_token,
            no_embed_class=no_embed_class,
            reg_tokens=reg_tokens,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            dynamic_img_size=dynamic_img_size,
            dynamic_img_pad=dynamic_img_pad,
            drop_rate=drop_rate,
            pos_drop_rate=pos_drop_rate,
            patch_drop_rate=patch_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=block_fn,
            mlp_layer=mlp_layer,
        )

    def forward_features(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Forward pass of the model without the classification head.

        Args:
            x:
                Input tensor of shape (batch_size, sequence_length, channels).
            mask:
                Mask of shape (batch_size, sequence_length) indicating which tokens
                should be masked. Tokens where the mask is True will only be used for
                causal attention, while unmasked tokens are used for bidirectional
                attention. If the mask is None, all tokens are used for bidirectional
                attention.

        Returns:
            Output tensor after applying the transformer blocks.
        """
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        if self.grad_checkpointing and not jit.is_scripting():
            # TODO: This probably doesn't work correctly as it doesn't consider the
            # mask.
            x = _manipulate.checkpoint_seq(self.blocks, x)
        else:
            for block in self.blocks:
                x = block(x, mask=mask)
        x = self.norm(x)
        return x
