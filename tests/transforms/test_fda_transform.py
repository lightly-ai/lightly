from PIL import Image

from lightly.transforms.fda_transform import (
    FDATransform,
    FDAView1Transform,
    FDAView2Transform,
)

from .. import helpers


def test_view_on_pil_image() -> None:
    single_view_transform = FDAView1Transform(input_size=32)
    sample = Image.new("RGB", (100, 100))
    output = single_view_transform(sample)
    assert output.shape == (3, 32, 32)


def test_multi_view_on_pil_image() -> None:
    multi_view_transform = FDATransform(
        view_1_transform=FDAView1Transform(input_size=32),
        view_2_transform=FDAView2Transform(input_size=32),
    )
    sample = Image.new("RGB", (100, 100))
    output = helpers.assert_list_tensor(multi_view_transform(sample))
    assert len(output) == 2
    assert output[0].shape == (3, 32, 32)
    assert output[1].shape == (3, 32, 32)
