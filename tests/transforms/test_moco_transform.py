from PIL import Image

from lightly.transforms.moco_transform import MoCoV1Transform, MoCoV2Transform

from .. import helpers


def test_moco_v1_multi_view_on_pil_image() -> None:
    multi_view_transform = MoCoV1Transform(input_size=32)
    sample = Image.new("RGB", (100, 100))
    output = helpers.assert_list_view(multi_view_transform(sample))
    assert len(output) == 2
    assert output[0].data.shape == (3, 32, 32)
    assert output[1].data.shape == (3, 32, 32)


def test_moco_v2_multi_view_on_pil_image() -> None:
    multi_view_transform = MoCoV2Transform(input_size=32)
    sample = Image.new("RGB", (100, 100))
    output = helpers.assert_list_view(multi_view_transform(sample))
    assert len(output) == 2
    assert output[0].data.shape == (3, 32, 32)
    assert output[1].data.shape == (3, 32, 32)
