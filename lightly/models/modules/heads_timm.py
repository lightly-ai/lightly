from typing import Type

from timm.layers import Mlp
from torch import Tensor
from torch.nn import GELU, LayerNorm, Linear, Module, Sequential


class AIMPredictionHeadBlock(Module):
    """Prediction head block for AIM [0].

    - [0]: AIM, 2024, https://arxiv.org/abs/2401.08541
    """

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
        super().__init__()
        self.norm = norm_layer(input_dim)  # type: ignore[call-arg]
        self.mlp = mlp_layer(  # type: ignore[call-arg]
            in_features=input_dim,
            hidden_features=int(input_dim * mlp_ratio),
            out_features=output_dim,
            act_layer=act_layer,
            drop=proj_drop,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.mlp(self.norm(x))
        return x


class AIMPredictionHead(Module):
    """Prediction head for AIM [0].

    - [0]: AIM, 2024, https://arxiv.org/abs/2401.08541
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
        super().__init__()
        self.blocks = Sequential(
            # Linear layer to project the input dimension to the hidden dimension.
            Linear(input_dim, hidden_dim, bias=False),
            # Main blocks.
            *[
                block_fn(  # type: ignore[call-arg]
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    mlp_ratio=mlp_ratio,
                    proj_drop=proj_drop,
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                    mlp_layer=mlp_layer,
                )
                for _ in range(num_blocks)
            ],
            # Linear layer to project the hidden dimension to the output dimension.
            norm_layer(hidden_dim),  # type: ignore[call-arg]
            Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.blocks(x)
        return x
