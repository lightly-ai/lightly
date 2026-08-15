"""Predictors that map a history of embeddings and actions to the future.

The predictor is the part that every latent world model has. It reads the
embeddings of the past frames and the actions that were taken, and it returns
embeddings of future frames. It never sees pixels, so an encoder trained
alongside it and a frozen pretrained encoder are interchangeable.

- [0]: LeWorldModel, 2026, https://arxiv.org/abs/2603.19312
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

# torch added scaled_dot_product_attention in 2.0. The package still declares
# torch>=1.10, so the fused call is optional and a manual path stands in.
_FUSED_ATTENTION_AVAILABLE = hasattr(nn.functional, "scaled_dot_product_attention")


def _manual_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None,
    dropout_p: float,
    training: bool,
) -> Tensor:
    """Attention for torch versions without a fused kernel.

    The result matches :func:`torch.nn.functional.scaled_dot_product_attention`
    for a boolean mask, where True means that the position is readable.
    """
    scores = (query @ key.transpose(-2, -1)) * (query.size(-1) ** -0.5)
    if attn_mask is not None:
        scores = scores.masked_fill(~attn_mask, float("-inf"))
    weights = scores.softmax(dim=-1)
    if dropout_p > 0.0:
        weights = nn.functional.dropout(weights, p=dropout_p, training=training)
    out: Tensor = weights @ value
    return out


def _modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply an adaptive layer norm (AdaLN) modulation."""
    return x * (1 + scale) + shift


class _PredictorBlock(nn.Module):
    """Transformer block with AdaLN-Zero action conditioning.

    The block follows the AdaLN-Zero scheme of DiT: the conditioning embedding
    produces per-timestep shift, scale and gate parameters, and the gate
    projections start at zero so the block is an identity at initialization.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.dropout = dropout

        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.proj_drop = nn.Dropout(dropout)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        adaln_proj = nn.Linear(hidden_dim, 6 * hidden_dim, bias=True)
        # AdaLN-Zero: the gates start at zero so the block is the identity at
        # initialization.
        nn.init.zeros_(adaln_proj.weight)
        nn.init.zeros_(adaln_proj.bias)
        self.adaln = nn.Sequential(nn.SiLU(), adaln_proj)

    def _attention(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, hidden_dim // self.num_heads
        )
        # (3, B, num_heads, seq_len, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        dropout_p = self.dropout if self.training else 0.0
        out: Tensor
        if _FUSED_ATTENTION_AVAILABLE:
            # Reached through getattr so that type checking against the oldest
            # supported torch, which has no fused kernel, does not fail here.
            fused_attention = getattr(nn.functional, "scaled_dot_product_attention")
            out = fused_attention(
                query, key, value, attn_mask=attn_mask, dropout_p=dropout_p
            )
        else:
            out = _manual_attention(
                query, key, value, attn_mask, dropout_p, self.training
            )
        out = out.transpose(1, 2).reshape(batch_size, seq_len, hidden_dim)
        projected: Tensor = self.proj_drop(self.proj(out))
        return projected

    def forward(self, x: Tensor, attn_mask: Tensor, cond: Tensor) -> Tensor:
        params = self.adaln(cond).chunk(6, dim=-1)
        shift_attn, scale_attn, gate_attn = params[0], params[1], params[2]
        shift_mlp, scale_mlp, gate_mlp = params[3], params[4], params[5]
        x = x + gate_attn * self._attention(
            _modulate(self.norm1(x), shift_attn, scale_attn), attn_mask
        )
        x = x + gate_mlp * self.mlp(_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class LatentDynamicsPredictor(nn.Module):
    """Predicts the next frame embedding from past embeddings and actions.

    The predictor is a causal transformer over a sequence of frame embeddings.
    Frame ``t`` attends to the frames ``0`` to ``t``, and position information
    is a learned embedding over frames. The action embedding of frame ``t``
    produces the shift, scale and gate parameters of every transformer block
    for that frame, which is the AdaLN-Zero conditioning that LeWM uses.

    Reference:
        LeWorldModel, 2026, https://arxiv.org/abs/2603.19312

    Attributes:
        num_frames: Maximum number of frames in the context window.
        input_dim: Dimension of an input frame embedding.
        output_dim: Dimension of a predicted frame embedding.

    Examples:
        >>> predictor = LatentDynamicsPredictor(
        ...     num_frames=4, input_dim=192, hidden_dim=384, depth=6, num_heads=6
        ... )
        >>> emb = torch.randn(8, 4, 192)  # (batch, time, dim)
        >>> action_emb = torch.randn(8, 4, 192)
        >>> pred = predictor(emb, action_emb=action_emb)  # (8, 4, 192)
        >>> # pred[:, t] is the prediction of emb[:, t + 1]
    """

    def __init__(
        self,
        *,
        num_frames: int,
        input_dim: int,
        hidden_dim: int,
        depth: int,
        num_heads: int,
        output_dim: int | None = None,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the latent dynamics predictor.

        All arguments are keyword-only.

        Args:
            num_frames: Maximum number of frames in the context window.
            input_dim:
                Dimension of an input frame embedding. Action embeddings are
                expected to have the same width.
            hidden_dim: Width of the transformer.
            depth: Number of transformer blocks.
            num_heads: Number of attention heads.
            output_dim: Dimension of the output. Defaults to ``input_dim``.
            mlp_ratio: Width of the block MLP relative to ``hidden_dim``.
            dropout: Dropout probability inside the transformer.
        """
        super().__init__()
        if num_frames <= 0:
            raise ValueError("num_frames must be a positive integer.")
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads "
                f"({num_heads})."
            )

        self.num_frames = num_frames
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, self.output_dim)
        self.action_proj = nn.Linear(input_dim, hidden_dim)

        self.frame_pos_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_dim))
        nn.init.trunc_normal_(self.frame_pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [
                _PredictorBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def _causal_mask(self, num_frames: int, device: torch.device) -> Tensor:
        """Build a boolean mask that allows frame t to attend to frames <= t."""
        frame_idx = torch.arange(num_frames, device=device)
        return frame_idx[:, None] >= frame_idx[None, :]

    def forward(self, embeddings: Tensor, action_emb: Tensor) -> Tensor:
        """Predict the next frame embedding for every frame in the context.

        Args:
            embeddings: Frame embeddings of shape ``(B, T, input_dim)``.
            action_emb:
                Action embeddings of shape ``(B, T, input_dim)``.

                ``action_emb[:, t]`` is the action taken **at** frame ``t``,
                which leads to frame ``t + 1``. A shape check cannot catch a
                phase error here, so a training step reads::

                    predicted = predictor(emb[:, :-1], action_emb=act[:, :-1])
                    target = emb[:, 1:]

        Returns:
            Predicted embeddings of shape ``(B, T, output_dim)``. Entry ``t``
            predicts the frame embedding at ``t + 1``, so the last entry has no
            target inside the clip.
        """
        if embeddings.ndim != 3:
            raise ValueError(
                "embeddings must have shape (B, T, input_dim), got "
                f"{tuple(embeddings.shape)}."
            )
        batch_size, num_frames = embeddings.shape[:2]
        if num_frames > self.num_frames:
            raise ValueError(
                f"embeddings have {num_frames} frames, but the predictor "
                f"context window is {self.num_frames}."
            )
        if action_emb.shape[:2] != (batch_size, num_frames):
            raise ValueError(
                "action_emb must have shape (B, T, D) matching embeddings, "
                f"got {tuple(action_emb.shape)} and {tuple(embeddings.shape)}."
            )

        x = self.input_proj(embeddings) + self.frame_pos_embed[:, :num_frames]
        cond = self.action_proj(action_emb)

        attn_mask = self._causal_mask(num_frames, x.device)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask, cond=cond)
        out: Tensor = self.output_proj(self.norm(x))
        return out

    def rollout(
        self,
        embeddings: Tensor,
        action_emb: Tensor,
        steps: int = 1,
    ) -> Tensor:
        """Roll the dynamics forward in latent space.

        The predictor consumes ``embeddings`` as context and then feeds its own
        predictions back as input for ``steps`` further frames. The context
        window slides, so the predictor never sees more than ``num_frames``
        frames at a time. This is the operation that a planner calls to score
        candidate action sequences.

        Args:
            embeddings: Context embeddings of shape ``(B, T_ctx, input_dim)``.
            action_emb:
                Action embeddings of shape ``(B, T_ctx + steps - 1, input_dim)``.
            steps: Number of frames to predict beyond the context.

        Returns:
            The predicted embeddings of shape ``(B, steps, output_dim)``.
        """
        if steps < 1:
            raise ValueError("steps must be a positive integer.")
        num_context = embeddings.size(1)
        if action_emb.size(1) < num_context + steps - 1:
            raise ValueError(
                f"action_emb must cover {num_context + steps - 1} frames, got "
                f"{action_emb.size(1)}."
            )

        frames = list(embeddings.unbind(dim=1))
        predictions = []
        for step in range(steps):
            end = num_context + step
            start = max(0, end - self.num_frames)
            context = torch.stack(frames[start:end], dim=1)
            next_frame = self(context, action_emb=action_emb[:, start:end])[:, -1]
            predictions.append(next_frame)
            frames.append(next_frame)
        return torch.stack(predictions, dim=1)
