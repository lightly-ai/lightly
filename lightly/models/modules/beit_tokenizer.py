# Copyright (c) 2024. Lightly AG and its affiliates.
# All Rights Reserved

# The image tokenizer follows the discrete VAE architecture from DALL-E [0],
# which is the tokenizer used in the BEIT paper [1].
#
# -  [0] Reference implementation: https://github.com/openai/DALL-E
#  - [1]: BEIT, 2021, https://arxiv.org/abs/2106.08254

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _EncoderBlock(nn.Module):
    """Residual block used inside the DALL-E encoder.

    Uses a bottleneck residual path with four convolutions and a learned
    post-gain of ``1 / n_layers²`` to stabilise deep networks.
    """

    def __init__(self, n_in: int, n_out: int, n_layers: int) -> None:
        super().__init__()
        n_hid = n_out // 4
        self.post_gain = 1.0 / (n_layers**2)
        self.id_path = nn.Conv2d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.res_path = nn.Sequential(
            OrderedDict(
                [
                    ("relu_1", nn.ReLU()),
                    ("conv_1", nn.Conv2d(n_in, n_hid, 3, padding=1)),
                    ("relu_2", nn.ReLU()),
                    ("conv_2", nn.Conv2d(n_hid, n_hid, 3, padding=1)),
                    ("relu_3", nn.ReLU()),
                    ("conv_3", nn.Conv2d(n_hid, n_hid, 3, padding=1)),
                    ("relu_4", nn.ReLU()),
                    ("conv_4", nn.Conv2d(n_hid, n_out, 1)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


class _DecoderBlock(nn.Module):
    """Residual block used inside the DALL-E decoder.

    Mirrors ``_EncoderBlock`` but the first bottleneck conv uses a 1×1 kernel
    (matching the DALL-E reference implementation).
    """

    def __init__(self, n_in: int, n_out: int, n_layers: int) -> None:
        super().__init__()
        n_hid = n_out // 4
        self.post_gain = 1.0 / (n_layers**2)
        self.id_path = nn.Conv2d(n_in, n_out, 1) if n_in != n_out else nn.Identity()
        self.res_path = nn.Sequential(
            OrderedDict(
                [
                    ("relu_1", nn.ReLU()),
                    ("conv_1", nn.Conv2d(n_in, n_hid, 1)),  # 1×1 in decoder
                    ("relu_2", nn.ReLU()),
                    ("conv_2", nn.Conv2d(n_hid, n_hid, 3, padding=1)),
                    ("relu_3", nn.ReLU()),
                    ("conv_3", nn.Conv2d(n_hid, n_hid, 3, padding=1)),
                    ("relu_4", nn.ReLU()),
                    ("conv_4", nn.Conv2d(n_hid, n_out, 3, padding=1)),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.id_path(x) + self.post_gain * self.res_path(x)


class _DALLEEncoder(nn.Module):
    """DALL-E discrete VAE encoder  (q_φ(z|x)).

    Four groups of residual blocks with max-pool downsampling between groups
    (×8 total spatial reduction), followed by a 1×1 conv to ``vocab_size``
    logits.  Architecture exactly matches the OpenAI DALL-E reference.

    Args:
        n_hid:
            Base number of hidden channels (default 256). Each group doubles
            this: 1×, 2×, 4×, 8×.
        n_blk_per_group:
            Number of residual blocks per group (default 2).
        input_channels:
            Number of input image channels (default 3).
        vocab_size:
            Codebook / vocabulary size (default 8192).
    """

    group_count: int = 4

    def __init__(
        self,
        n_hid: int = 256,
        n_blk_per_group: int = 2,
        input_channels: int = 3,
        vocab_size: int = 8192,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size

        blk_range = range(n_blk_per_group)
        n_layers = self.group_count * n_blk_per_group
        make_blk = partial(_EncoderBlock, n_layers=n_layers)

        self.blocks = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Conv2d(input_channels, 1 * n_hid, 7, padding=3)),
                    (
                        "group_1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(1 * n_hid, 1 * n_hid),
                                        )
                                        for i in blk_range
                                    ],
                                    ("pool", nn.MaxPool2d(kernel_size=2)),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                1 * n_hid if i == 0 else 2 * n_hid,
                                                2 * n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    ("pool", nn.MaxPool2d(kernel_size=2)),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                2 * n_hid if i == 0 else 4 * n_hid,
                                                4 * n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    ("pool", nn.MaxPool2d(kernel_size=2)),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_4",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                4 * n_hid if i == 0 else 8 * n_hid,
                                                8 * n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                ]
                            )
                        ),
                    ),
                    (
                        "output",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("relu", nn.ReLU()),
                                    ("conv", nn.Conv2d(8 * n_hid, vocab_size, 1)),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes an image into logits over the visual vocabulary.

        Args:
            x:
                Input image tensor of shape ``(B, C, H, W)``, dtype
                ``torch.float32``.

        Returns:
            Logits tensor of shape ``(B, vocab_size, H/8, W/8)``.
        """
        if x.ndim != 4:
            raise ValueError(f"input shape {x.shape} is not 4d")
        if x.dtype != torch.float32:
            raise ValueError("input must have dtype torch.float32")
        return self.blocks(x)


class _DALLEDecoder(nn.Module):
    """DALL-E discrete VAE decoder  (p_ψ(x|z)).

    Four groups of residual blocks with nearest-neighbour upsampling between
    groups (×8 total spatial increase), followed by a 1×1 conv to
    ``2 * output_channels``.  The output is split into mean and log-variance
    over colour channels.  Architecture matches the OpenAI DALL-E reference.

    Args:
        n_init:
            Initial projection width from the vocabulary embedding (default 128).
        n_hid:
            Base number of hidden channels (default 256).
        n_blk_per_group:
            Number of residual blocks per group (default 2).
        output_channels:
            Number of output image channels (default 3).
        vocab_size:
            Codebook / vocabulary size (default 8192).
    """

    group_count: int = 4

    def __init__(
        self,
        n_init: int = 128,
        n_hid: int = 256,
        n_blk_per_group: int = 2,
        output_channels: int = 3,
        vocab_size: int = 8192,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.output_channels = output_channels

        blk_range = range(n_blk_per_group)
        n_layers = self.group_count * n_blk_per_group
        make_blk = partial(_DecoderBlock, n_layers=n_layers)

        self.blocks = nn.Sequential(
            OrderedDict(
                [
                    ("input", nn.Conv2d(vocab_size, n_init, 1)),
                    (
                        "group_1",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                n_init if i == 0 else 8 * n_hid,
                                                8 * n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    (
                                        "upsample",
                                        nn.Upsample(scale_factor=2, mode="nearest"),
                                    ),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_2",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                8 * n_hid if i == 0 else 4 * n_hid,
                                                4 * n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    (
                                        "upsample",
                                        nn.Upsample(scale_factor=2, mode="nearest"),
                                    ),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_3",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                4 * n_hid if i == 0 else 2 * n_hid,
                                                2 * n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                    (
                                        "upsample",
                                        nn.Upsample(scale_factor=2, mode="nearest"),
                                    ),
                                ]
                            )
                        ),
                    ),
                    (
                        "group_4",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    *[
                                        (
                                            f"block_{i + 1}",
                                            make_blk(
                                                2 * n_hid if i == 0 else 1 * n_hid,
                                                1 * n_hid,
                                            ),
                                        )
                                        for i in blk_range
                                    ],
                                ]
                            )
                        ),
                    ),
                    (
                        "output",
                        nn.Sequential(
                            OrderedDict(
                                [
                                    ("relu", nn.ReLU()),
                                    (
                                        "conv",
                                        nn.Conv2d(1 * n_hid, 2 * output_channels, 1),
                                    ),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes a soft token map into a reconstructed image.

        Args:
            x:
                Soft one-hot token map of shape ``(B, vocab_size, h, w)``,
                dtype ``torch.float32``.

        Returns:
            Reconstructed image tensor of shape ``(B, C, H, W)`` where
            ``C = output_channels``.  The decoder output has
            ``2 * output_channels`` channels; only the first half (the mean)
            is returned — the second half (log-variance) is discarded during
            pre-training, consistent with the DALL-E reference.
        """
        if x.ndim != 4:
            raise ValueError(f"input shape {x.shape} is not 4d")
        if x.shape[1] != self.vocab_size:
            raise ValueError(
                f"input has {x.shape[1]} channels but model built for {self.vocab_size}"
            )
        if x.dtype != torch.float32:
            raise ValueError("input must have dtype torch.float32")
        out = self.blocks(x)
        return out[:, : self.output_channels]


class ImageTokenizer(nn.Module):
    """Discrete VAE image tokenizer following the DALL-E architecture .

    Used in BEIT  to convert images into discrete visual tokens that serve
    as pre-training targets for masked image modelling.  The tokenizer is
    pre-trained separately (Stage 1) and kept **frozen** during BEIT
    pre-training (Stage 2).

    The encoder down-samples by 8× so a 224×224 image becomes a
    14×14 grid of 8192-way token logits.


    Attributes:
        vocab_size:
            Size of the visual codebook (default 8192).
        temperature:
            Gumbel-softmax temperature used during tokenizer training
            (default 1.0).  Has no effect when calling ``tokenize()``.

    Examples:
        >>> import torch
        >>> from lightly.models.modules import ImageTokenizer
        >>>
        >>> tokenizer = ImageTokenizer()
        >>> images = torch.randn(2, 3, 224, 224)
        >>>
        >>> # During BEIT pre-training: get hard token ids (frozen tokenizer)
        >>> token_ids = tokenizer.tokenize(images)  # (2, 196)
        >>>
        >>> # During tokenizer training: get logits + reconstruction
        >>> logits, recon = tokenizer(images)  # (2, 8192, 28, 28), (2, 3, 224, 224)
    """

    def __init__(
        self,
        input_channels: int = 3,
        vocab_size: int = 8192,
        n_hid: int = 256,
        n_blk_per_group: int = 2,
        n_init: int = 128,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.temperature = temperature

        self.encoder = _DALLEEncoder(
            n_hid=n_hid,
            n_blk_per_group=n_blk_per_group,
            input_channels=input_channels,
            vocab_size=vocab_size,
        )
        self.decoder = _DALLEDecoder(
            n_init=n_init,
            n_hid=n_hid,
            n_blk_per_group=n_blk_per_group,
            output_channels=input_channels,
            vocab_size=vocab_size,
        )

    @torch.no_grad()
    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Converts images into discrete visual token indices.

        Called with ``torch.no_grad()`` — the tokenizer is frozen during
        BEIT pre-training.

        Args:
            x:
                Input image tensor of shape ``(B, C, H, W)``,
                dtype ``torch.float32``.

        Returns:
            Token indices of shape ``(B, h*w)`` as a ``torch.long`` tensor,
            where ``h = H / 8`` and ``w = W / 8``.
        """
        logits = self.encoder(x)
        token_ids = logits.argmax(dim=1)
        B, h, w = token_ids.shape
        return token_ids.view(B, h * w)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encodes and decodes an image for tokenizer training.

        Uses Gumbel-softmax to allow gradients to flow through the discrete
        bottleneck .

        Args:
            x:
                Input image tensor of shape ``(B, C, H, W)``,
                dtype ``torch.float32``.

        Returns:
            A tuple of:
                - **logits** ``(B, vocab_size, h, w)`` — encoder output, used
                  to compute the codebook / reconstruction loss.
                - **recon** ``(B, C, H, W)`` — pixel-space reconstruction
                  from the decoder.
        """
        logits = self.encoder(x)
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=False)
        recon = self.decoder(soft_one_hot)
        return logits, recon
