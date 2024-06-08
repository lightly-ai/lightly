import pytest


def test_override_torchvision_transform():
    # CutMix is only available in torchvision.transforms.v2, so it should not be
    # available within the torchvision.transforms module.
    try:
        from torchvision.transforms import CutMix
    except ImportError:
        pytest.fail("CutMix could not be imported from torchvision.transforms")
