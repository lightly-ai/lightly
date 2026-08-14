import pytest
import torch.nn as nn

from lightly.models import BarlowTwins


class TestModelsBarlowTwins:
    def test_deprecation_warning(self) -> None:
        with pytest.warns(FutureWarning, match="deprecated"):
            BarlowTwins(nn.Identity(), num_ftrs=8, proj_hidden_dim=8, out_dim=8)
