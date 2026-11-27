import torch

from lightly.models import utils


def test_add_stochastic_positional_noise_disabled() -> None:
    projection = torch.nn.Linear(8, 4)
    pos_embeddings = torch.randn(2, 3, 4)

    out = utils.add_stochastic_positional_noise(
        pos_embeddings=pos_embeddings,
        projection_weight=projection.weight,
        noise_dim=8,
        noise_std=0.0,
    )

    assert torch.equal(out, pos_embeddings)


def test_add_stochastic_positional_noise_enabled() -> None:
    projection = torch.nn.Linear(8, 4)
    pos_embeddings = torch.randn(2, 3, 4)

    out = utils.add_stochastic_positional_noise(
        pos_embeddings=pos_embeddings,
        projection_weight=projection.weight,
        noise_dim=8,
        noise_std=0.25,
    )

    assert out.shape == pos_embeddings.shape
    assert not torch.equal(out, pos_embeddings)
