from typing import Type

from timm.layers import Mlp
from torch import Tensor
from torch.nn import GELU, LayerNorm, Module, Sequential


class AIMPredictionHeadBlock(Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        act_layer: Type[Module] = GELU,
        norm_layer: Type[Module] = LayerNorm,
        mlp_layer: Type[Module] = Mlp,
    ) -> None:
        self.norm = norm_layer(input_dim)
        self.mlp = mlp_layer(
            in_features=input_dim,
            hidden_features=int(input_dim * mlp_ratio),
            out_features=output_dim,
            act_layer=act_layer,
            drop=proj_drop,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mlp(self.norm(x))
        return x


class AIMPredictionHead(Module):
    """Prediction head for AIM model.

    Note: This is a best effort implementation based on the descriptions in the paper
    as the official implementation does not include the projection head.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 2048,
        num_blocks: int = 12,
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        act_layer: Type[Module] = GELU,
        norm_layer: Type[Module] = LayerNorm,
        mlp_layer: Type[Module] = Mlp,
        block_fn: Type[Module] = AIMPredictionHeadBlock,
    ) -> None:
        self.blocks = Sequential(
            # First MLP to project the input to the hidden dimension. The paper does not
            # specify how exactly this is done. We assume it happens in the last layer
            # of the first MLP block.
            norm_layer(input_dim),
            mlp_layer(
                in_features=input_dim,
                hidden_features=int(hidden_dim * mlp_ratio),
                out_features=hidden_dim,
                act_layer=act_layer,
                drop=proj_drop,
            ),
            # Main blocks.
            *[
                block_fn(
                    dim=hidden_dim,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for _ in range(num_blocks - 2)
            ],
            # Final block to project the hidden dimension to the output dimension. The
            # paper does not specify how exactly this is done. We assume that the
            # conversion happens in the last linear layer.
            norm_layer(hidden_dim),
            mlp_layer(
                in_features=hidden_dim,
                hidden_features=int(hidden_dim * mlp_ratio),
                out_features=output_dim,
                act_layer=act_layer,
                drop=proj_drop,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)
