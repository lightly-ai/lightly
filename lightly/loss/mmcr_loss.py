import torch
import torch.nn as nn
from torch.linalg import svd


class MMCRLoss(nn.Module):
    """Implementation of the loss function from MMCR [0] using Manifold Capacity.
    All hyperparameters are set to the default values from the paper for ImageNet.

    - [0]: Efficient Coding of Natural Images using Maximum Manifold Capacity
    Representations, 2023, https://arxiv.org/pdf/2303.03307.pdf

    Examples:
        >>> # initialize loss function
        >>> loss_fn = MMCRLoss()
        >>> transform = MMCRTransform(k=2)
        >>>
        >>> # transform images, then feed through encoder and projector
        >>> x = transform(x)
        >>> online = online_network(x)
        >>> momentum = momentum_network(x)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(online, momentum)
    """

    def __init__(self, lmda: float = 5e-3):
        """Initializes the MMCRLoss module with the specified lambda parameter.

        Args:
            lmda: The regularization parameter.

        Raises:
            ValueError: If lmda is less than 0.
        """
        super().__init__()
        if lmda < 0:
            raise ValueError("lmda must be greater than or equal to 0")

        self.lmda = lmda

    def forward(self, online: torch.Tensor, momentum: torch.Tensor) -> torch.Tensor:
        """Computes the MMCR loss for the online and momentum network outputs.

        Args:
            online:
                Output of the online network for the current batch. Expected to be
                of shape (batch_size, k, embedding_size), where k represents the
                number of randomly augmented views for each sample.
            momentum:
                Output of the momentum network for the current batch. Expected to be
                of shape (batch_size, k, embedding_size), where k represents the
                number of randomly augmented views for each sample.

        Returns:
            The computed loss value.
        """
        assert (
            online.shape == momentum.shape
        ), "online and momentum need to have the same shape"

        B = online.shape[0]

        # Concatenate and calculate centroid
        z = torch.cat([online, momentum], dim=1)
        c = torch.mean(z, dim=1)  # B x D

        # Calculate singular values
        _, S_z, _ = svd(z)
        _, S_c, _ = svd(c)

        # Calculate loss
        loss = -1.0 * torch.sum(S_c) + self.lmda * torch.sum(S_z) / B

        return loss
