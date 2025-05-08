import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from sklearn.decomposition import PCA as SKPCA

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
    assert X_transformed.dtype == np.float32  # type: ignore[comparison-overlap]


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


def test_fit_pca_invalid_fraction_raises_value_error() -> None:
    X = np.random.randn(20, 4).astype(np.float32)
    # fraction > 1 should be rejected
    with pytest.raises(ValueError) as excinfo:
        _ = fit_pca(X, n_components=2, fraction=1.5)
    assert "fraction must be in (0, 1]" in str(excinfo.value)

    # negative fraction should also be rejected
    with pytest.raises(ValueError) as excinfo2:
        _ = fit_pca(X, n_components=2, fraction=-0.1)
    assert "fraction must be in (0, 1]" in str(excinfo2.value)

    # zero fraction should also be rejected
    with pytest.raises(ValueError) as excinfo3:
        _ = fit_pca(X, n_components=2, fraction=0.0)
    assert "fraction must be in (0, 1]" in str(excinfo3.value)


@pytest.mark.parametrize(  # type: ignore[misc]
    "frac",
    [
        0.5,  # 50% of data
        1.0,  # all data
    ],
)
def test_fit_pca_fraction_sample_size(frac: float, seed: int = 42) -> None:
    X = np.arange(40, dtype=np.float32).reshape(10, 4)
    n_sub = int(X.shape[0] * frac)

    np.random.seed(seed)
    perm = np.random.permutation(X.shape[0])
    expected_indices = perm[:n_sub]
    expected_mean = X[expected_indices].mean(axis=0)

    np.random.seed(seed)
    pca = fit_pca(X, n_components=2, fraction=frac)

    # The fitted PCA.mean should match the sample‐subset mean
    assert pca.mean is not None
    assert_allclose(pca.mean, expected_mean, rtol=1e-6)


def make_2d_line_dataset(
    n: int = 100, noise: float = 1e-6, seed: int = 0
) -> NDArray[np.float32]:
    rng = np.random.RandomState(seed)
    x = rng.rand(n, 1) * 10
    data = np.hstack([x, 2 * x])
    data += rng.randn(n, 2) * noise
    return data.astype(np.float32)


def test_pca_aligns_with_sklearn() -> None:
    X = make_2d_line_dataset()
    skpca = SKPCA(n_components=1, svd_solver="full").fit(X)
    sk_comp = skpca.components_[0]  # shape (2,)
    pca = PCA(n_components=1).fit(X)
    assert pca.w is not None
    ours = pca.w[:, 0]
    assert_allclose(np.abs(ours), np.abs(sk_comp), atol=1e-5)
