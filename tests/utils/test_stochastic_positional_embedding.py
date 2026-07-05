import torch

from lightly.models import utils


def test_add_stochastic_positional_noise_disabled() -> None:
    projection = torch.nn.Linear(8, 4)
    pos_embeddings = torch.randn(2, 3, 4)

    out = utils.add_stochastic_positional_noise(
        pos_embeddings=pos_embeddings,
        projection=projection,
        noise_dim=8,
        enabled=False,
    )

    assert torch.equal(out, pos_embeddings)


def test_add_stochastic_positional_noise_enabled_shape() -> None:
    projection = torch.nn.Linear(8, 4)
    pos_embeddings = torch.randn(2, 3, 4)

    out = utils.add_stochastic_positional_noise(
        pos_embeddings=pos_embeddings,
        projection=projection,
        noise_dim=8,
        enabled=True,
    )

    assert out.shape == pos_embeddings.shape
