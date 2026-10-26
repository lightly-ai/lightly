"""DSE regularizer."""

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module


class DSERegularizer(Module):
    """Dense Structure Estimator regularizer for dense representations.

    This experimental regularizer is based on the training regularizer variant from "Exploring
    Structural Degradation in Dense Representations for Self-supervised
    Learning", 2025, https://arxiv.org/abs/2510.17299.

    The regularizer expects dense representations of shape ``(B, N, D)``, where
    ``B`` is the batch size, ``N`` is the number of dense locations such as
    patches or spatial positions, and ``D`` is the feature dimension. Higher DSE
    values indicate better dense structure, so this module returns
    ``-weight * dse`` and can be added to an existing SSL loss.

    Examples:
        >>> regularizer = DSERegularizer(weight=0.1)
        >>> ssl_loss = torch.tensor(1.0)
        >>>
        >>> # Transformer output: patch_tokens has shape (B, N, D).
        >>> patch_tokens = torch.randn(8, 196, 384)
        >>> loss = ssl_loss + regularizer(patch_tokens)
        >>>
        >>> # Convolutional output: features has shape (B, C, H, W).
        >>> features = torch.randn(8, 384, 14, 14)
        >>> dense_features = features.flatten(2).transpose(1, 2)
        >>> loss = ssl_loss + regularizer(dense_features)

    Attributes:
        weight:
            Weight of the regularization term.
        normalize:
            If True, L2-normalize representations along the feature dimension.
        local_clusters:
            Number of clusters for patches within each image.
        global_clusters:
            Number of clusters for patches across image groups.
        max_kmeans_iters:
            Maximum number of k-means iterations.
        tol:
            Early stopping threshold for centroid shift in k-means.
    """

    def __init__(
        self,
        weight: float = 1.0,
        normalize: bool = True,
        local_clusters: int = 3,
        global_clusters: int = 24,
        max_kmeans_iters: int = 20,
        tol: float = 1e-4,
    ):
        """Initializes the DSERegularizer with the specified parameters.

        Args:
            weight:
                Weight of the regularization term. Must be >= 0.
            normalize:
                If True, L2-normalize representations along the feature dimension
                before computing the DSE score.
            local_clusters:
                Number of clusters for patches within each image. Must be >= 1.
            global_clusters:
                Number of clusters for patches across image groups. Must be >= 1.
            max_kmeans_iters:
                Maximum number of k-means iterations. Must be >= 1.
            tol:
                Early stopping threshold for centroid shift in k-means. Must be > 0.
        """
        super().__init__()
        if weight < 0:
            raise ValueError(f"weight must be >= 0, got {weight}.")
        if local_clusters < 1:
            raise ValueError(f"local_clusters must be >= 1, got {local_clusters}.")
        if global_clusters < 1:
            raise ValueError(f"global_clusters must be >= 1, got {global_clusters}.")
        if max_kmeans_iters < 1:
            raise ValueError(f"max_kmeans_iters must be >= 1, got {max_kmeans_iters}.")
        if tol <= 0:
            raise ValueError(f"tol must be > 0, got {tol}.")

        self.weight = weight
        self.normalize = normalize
        self.local_clusters = local_clusters
        self.global_clusters = global_clusters
        self.max_kmeans_iters = max_kmeans_iters
        self.tol = tol

    def forward(self, representations: Tensor) -> Tensor:
        """Computes the DSE regularization loss.

        Args:
            representations:
                Dense representations of shape ``(B, N, D)``.

        Returns:
            Scalar DSE regularization loss.
        """
        if representations.ndim != 3:
            raise ValueError(
                "representations must have shape (batch_size, num_tokens, dim), "
                f"got {representations.shape}."
            )
        if min(representations.shape) == 0:
            raise ValueError(
                "representations must have non-empty batch, token, and feature "
                f"dimensions, got {representations.shape}."
            )

        if self.weight == 0:
            return representations.new_zeros(())

        if self.normalize:
            representations = F.normalize(representations, dim=-1)

        batch_size, num_tokens, dim = representations.shape
        centered = representations - representations.mean(dim=(0, 1), keepdim=True)
        avg_l2 = torch.norm(centered, dim=-1, p=2).mean()

        intra_image = self._intra_image_structure(representations)
        inter_batch, intra_batch = self._batch_structure(representations)
        effective_dimensionality = self._effective_dimensionality(
            representations.reshape(batch_size * num_tokens, dim)
        )

        dse = (
            inter_batch / (avg_l2 + 1e-12)
            + effective_dimensionality
            - 0.5 * (intra_batch / (avg_l2 + 1e-12) + intra_image / (avg_l2 + 1e-12))
        )
        loss: Tensor = -self.weight * dse
        return loss

    def _intra_image_structure(self, representations: Tensor) -> Tensor:
        batch_size = representations.shape[0]
        terms = []
        for batch_index in range(batch_size):
            image_representations = representations[batch_index]
            _, labels = self._kmeans(
                image_representations.detach(), self.local_clusters
            )
            ranks = []
            for cluster_index in torch.unique(labels):
                cluster = image_representations[labels == cluster_index]
                if cluster.shape[0] > 1:
                    ranks.append(self._centered_singular_sum(cluster))
            if ranks:
                terms.append(torch.stack(ranks).mean())

        if not terms:
            return representations.new_zeros(())
        return torch.stack(terms).mean()

    def _batch_structure(self, representations: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, _, dim = representations.shape
        inter_class_images = max(self.global_clusters // self.local_clusters, 1)
        inter_terms = []
        intra_terms = []

        for start in range(0, batch_size, inter_class_images):
            group = representations[start : start + inter_class_images].reshape(-1, dim)
            centroids, labels = self._kmeans(group.detach(), self.global_clusters)
            if centroids.shape[0] > 1:
                distances = torch.cdist(group, centroids)
                assigned_mask = torch.zeros_like(distances, dtype=torch.bool)
                assigned_mask[
                    torch.arange(group.shape[0], device=group.device), labels
                ] = True
                other_distances = distances.masked_fill(assigned_mask, float("inf"))
                inter_terms.append(other_distances.min(dim=1).values.mean())

            for cluster_index in torch.unique(labels):
                cluster = group[labels == cluster_index]
                if cluster.shape[0] > 1:
                    intra_terms.append(self._centered_singular_sum(cluster))

        if inter_terms:
            inter_batch = torch.stack(inter_terms).mean()
        else:
            inter_batch = representations.new_zeros(())

        if intra_terms:
            intra_batch = torch.stack(intra_terms).mean()
        else:
            intra_batch = representations.new_zeros(())

        return inter_batch, intra_batch

    def _effective_dimensionality(self, representations: Tensor) -> Tensor:
        singular_values = torch.linalg.svdvals(representations)
        probabilities = singular_values / (singular_values.sum() + 1e-12)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-12))
        return torch.exp(entropy) / min(representations.shape)

    def _centered_singular_sum(self, representations: Tensor) -> Tensor:
        centered = representations - representations.mean(dim=0, keepdim=True)
        singular_values: Tensor = torch.linalg.svdvals(centered)
        denominator = math.sqrt(
            max(representations.shape[0] - 1, representations.shape[1])
        )
        return singular_values.sum() / denominator

    def _kmeans(self, x: Tensor, num_clusters: int) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            num_points = x.shape[0]
            num_clusters = min(num_clusters, num_points)
            centroids = self._kmeans_plus_plus_init(x, num_clusters)

            for _ in range(self.max_kmeans_iters):
                distances = torch.cdist(x, centroids)
                labels = torch.argmin(distances, dim=1)
                new_centroids = []
                for cluster_index in range(num_clusters):
                    cluster = x[labels == cluster_index]
                    if cluster.shape[0] > 0:
                        new_centroids.append(cluster.mean(dim=0))
                    else:
                        index = torch.randint(num_points, (1,), device=x.device)
                        new_centroids.append(x[index].squeeze(0))
                new_centroids_tensor = torch.stack(new_centroids)
                shift = (new_centroids_tensor - centroids).pow(2).sum().sqrt()
                centroids = new_centroids_tensor
                if shift < self.tol:
                    break

            distances = torch.cdist(x, centroids)
            labels = torch.argmin(distances, dim=1)
            return centroids.detach(), labels.detach()

    def _kmeans_plus_plus_init(self, x: Tensor, num_clusters: int) -> Tensor:
        num_points = x.shape[0]
        first_index = torch.randint(num_points, (1,), device=x.device)
        centroids = x[first_index]

        for _ in range(num_clusters - 1):
            distances = torch.cdist(x, centroids).min(dim=1).values
            # Standard k-means++ uses D^2 sampling for the squared Euclidean
            # k-means objective. This differs from the paper (maybe a bug!?).
            probabilities = distances.square()
            probabilities = probabilities / (probabilities.sum() + 1e-12)
            if torch.sum(probabilities) == 0:
                probabilities = torch.ones_like(probabilities) / probabilities.numel()
            next_index = torch.multinomial(probabilities, 1)
            centroids = torch.cat((centroids, x[next_index]), dim=0)

        return centroids
