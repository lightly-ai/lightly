"""Encoders that turn actions into conditioning embeddings.

A world model conditions its prediction on what the agent did. The action
arrives as a low-dimensional vector, which is what a robot control task gives.
"""

from __future__ import annotations

from torch import Tensor, nn


class ActionEncoder(nn.Module):
    """Maps continuous action vectors to conditioning embeddings.

    The module applies the same MLP to every timestep of a batch of action
    sequences. Concatenate the action with the proprioceptive state before
    calling it to condition on both.

    Attributes:
        action_dim: Dimension of a single action vector.
        output_dim: Dimension of the conditioning embedding.
        mlp: The multi-layer perceptron applied to each action vector.

    Examples:
        >>> action_encoder = ActionEncoder(action_dim=2, output_dim=192)
        >>> actions = torch.randn(8, 4, 2)  # (batch, time, action_dim)
        >>> action_emb = action_encoder(actions)  # (8, 4, 192)
    """

    def __init__(
        self,
        *,
        action_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
    ) -> None:
        """Initialize the action encoder.

        All arguments are keyword-only.

        Args:
            action_dim: Dimension of a single action vector.
            output_dim: Dimension of the conditioning embedding.
            hidden_dim:
                Width of the hidden layer. Defaults to ``4 * output_dim``.
        """
        super().__init__()
        if action_dim <= 0:
            raise ValueError("action_dim must be a positive integer.")
        if output_dim <= 0:
            raise ValueError("output_dim must be a positive integer.")
        hidden_dim = hidden_dim if hidden_dim is not None else 4 * output_dim
        self.action_dim = action_dim
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, actions: Tensor) -> Tensor:
        """Encode a batch of action sequences.

        Args:
            actions: Actions of shape ``(B, T, action_dim)``.

        Returns:
            Action embeddings of shape ``(B, T, output_dim)``.
        """
        if actions.ndim != 3:
            raise ValueError(
                f"actions must have shape (B, T, action_dim), got {actions.shape}."
            )
        out: Tensor = self.mlp(actions)
        return out
