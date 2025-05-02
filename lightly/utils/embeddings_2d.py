"""Transforms embeddings to two-dimensional space for visualization."""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, cast

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from numpy.typing import NDArray
else:
    NDArray = Any


class PCA(object):
    """Handmade PCA to bypass sklearn dependency.

    Attributes:
        n_components:
            Number of principal components to keep.
        eps:
            Epsilon for numerical stability.
        mean:
            Mean of the data.
        w:
            Eigenvectors of the covariance matrix.

    """

    def __init__(self, n_components: int = 2, eps: float = 1e-10):
        self.n_components = n_components
        self.eps = eps
        # We only care about array shape for typing, not dtype
        self.mean: Optional[NDArray[Any]] = None
        self.w: Optional[NDArray[Any]] = None

    def fit(self, X: NDArray[np.float32]) -> PCA:
        """Fits PCA to data in X.

        Args:
            X:
                Datapoints stored in numpy array of size n x d.

        Returns:
            PCA: The fitted PCA object to transform data points.

        """
        X = X.astype(np.float32)
        self.mean = X.mean(axis=0)
        assert self.mean is not None
        X = X - self.mean + self.eps
        cov = np.cov(X.T) / X.shape[0]
        v, w = np.linalg.eig(cov)
        idx = v.argsort()[::-1]  # Sort eigenvalues in descending order
        v, w = v[idx], w[:, idx]
        # At runtime we cast to float32 for ML performance,
        # but MyPy only checks shape (dtype is `Any`).
        self.w = w.astype(np.float32)
        return self

    def transform(self, X: NDArray[np.float32]) -> NDArray[Any]:
        """Uses PCA to transform data in X.

        Args:
            X:
                Datapoints stored in numpy array of size n x d.

        Returns:
            Numpy array of n x p datapoints where p <= d.

        Raises:
            ValueError:
                If PCA is not fitted before calling this method.

        """
        if self.mean is None or self.w is None:
            raise ValueError("PCA not fitted yet. Call fit() before transform().")

        X = X.astype(np.float32)
        X = X - self.mean + self.eps
        result = X.dot(self.w)[:, : self.n_components].astype(np.float32)
        return cast(NDArray[Any], result)


def fit_pca(
    embeddings: NDArray[np.float32],
    n_components: int = 2,
    fraction: Optional[float] = None,
) -> PCA:
    """Fits PCA to a randomly selected subset of embeddings.

    For large datasets, it can be unfeasible to perform PCA on the whole data.
    This method can fit a PCA on a fraction of the embeddings in order to save
    computational resources.

    Args:
        embeddings:
            Datapoints stored in numpy array of size n x d.
        n_components:
            Number of principal components to keep.
        fraction:
            Fraction of the dataset to fit PCA on.

    Returns:
        A transformer which can be used to transform embeddings
        to lower dimensions.

    Raises:
        ValueError: If fraction < 0 or fraction > 1.

    """
    if fraction is not None:
        if fraction < 0.0 or fraction > 1.0:
            raise ValueError(f"fraction must be in [0, 1] but was {fraction}.")

    N = embeddings.shape[0]
    n = N if fraction is None else min(N, int(N * fraction))
    X = embeddings[np.random.permutation(N)][:n]
    return PCA(n_components=n_components).fit(X)
