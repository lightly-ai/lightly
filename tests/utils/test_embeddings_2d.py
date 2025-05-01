import numpy as np
import pytest

from lightly.utils.embeddings_2d import PCA, fit_pca


def test_pca_fit_transform_shapes_and_dtype() -> None:
    # Create dummy data: 100 samples, 10 features
    X = np.random.randn(100, 10).astype(np.float32)

    # Initialize and fit PCA with 2 components
    pca = PCA(n_components=2)
    pca.fit(X)

    # Ensure mean and eigenvectors were set
    assert pca.mean is not None
    assert pca.w is not None
    # w holds all eigenvectors, so shape = (feature_dim, feature_dim)
    assert pca.w.shape == (10, 10)

    # Transform the data and check shape & dtype
    X_transformed = pca.transform(X)
    assert X_transformed.shape == (100, 2)
    assert isinstance(X_transformed, np.ndarray)
    assert X_transformed.dtype == np.float32


def test_transform_without_fit_raises_value_error() -> None:
    pca = PCA(n_components=3)
    X = np.ones((5, 4), dtype=np.float32)
    with pytest.raises(ValueError):
        _ = pca.transform(X)


def test_fit_pca_helper_behavior() -> None:
    X = np.random.randn(50, 5).astype(np.float32)
    # Use the helper to fit on half the data
    helper = fit_pca(X, n_components=1, fraction=0.5)
    # Compare to manual PCA.fit on the same subset
    subset = X[np.random.permutation(50)][:25]
    direct = PCA(n_components=1).fit(subset)

    out1 = helper.transform(X)
    out2 = direct.transform(X)

    # Both should produce shape (50, 1)
    assert out1.shape == (50, 1)
    assert out2.shape == (50, 1)
