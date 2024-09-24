from PIL import Image

from lightly.transforms.pirl_transform import PIRLTransform
from .. import helpers


def test_multi_view_on_pil_image() -> None:
    multi_view_transform = PIRLTransform(input_size=32)
    sample = Image.new("RGB", (100, 100))
    output = helpers.assert_list_tensor(multi_view_transform(sample))
    assert len(output) == 2
    assert output[0].shape == (3, 32, 32)
    assert output[1].shape == (9, 3, 10, 10)
