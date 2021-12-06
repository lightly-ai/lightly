""" Transform embeddings to two-dimensional space for visualization. """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import numpy as np


class PCA(object):
    """Handmade PCA to bypass sklearn dependency.

    Attributes:
        n_components:
            Number of principal components to keep.
        eps:
            Epsilon for numerical stability.
    """

    def __init__(self, n_components: int = 2, eps: float = 1e-10):
        self.n_components = n_components
        self.mean = None
        self.w = None
        self.eps = eps

    def fit(self, X: np.ndarray):
        """Fits PCA to data in X.

        Args:
            X:
                Datapoints stored in numpy array of size n x d.

        Returns:
            PCA object to transform datapoints.

        """
        X = X.astype(np.float32)
        self.mean = X.mean(axis=0)
        X = X - self.mean + self.eps
        cov = np.cov(X.T) / X.shape[0]
        v, w = np.linalg.eig(cov)
        idx = v.argsort()[::-1]
        v, w = v[idx], w[:, idx]
        self.w = w
        return self

    def transform(self, X: np.ndarray):
        """Uses PCA to transform data in X.

        Args:
            X:
                Datapoints stored in numpy array of size n x d.

        Returns:
            Numpy array of n x p datapoints where p <= d.

        """
        X = X.astype(np.float32)
        X = X - self.mean + self.eps
        return X.dot(self.w)[:, :self.n_components]


def fit_pca(embeddings: np.ndarray, n_components: int = 2, fraction: float = None):
    """Fits PCA to randomly selected subset of embeddings.

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
        if fraction < 0. or fraction > 1.:
            msg = f'fraction must be in [0, 1] but was {fraction}.'
            raise ValueError(msg)

    N = embeddings.shape[0]
    n = N if fraction is None else min(N, int(N * fraction))
    X = embeddings[np.random.permutation(N)][:n]
    return PCA(n_components=n_components).fit(X)
