from typing import List, Tuple, Union

from PIL import Image

from lightly.transforms.rotation import (
    RandomRotate,
    RandomRotateDegrees,
    random_rotation_transform,
)


def test_RandomRotate_on_pil_image() -> None:
    random_rotate = RandomRotate()
    sample = Image.new("RGB", (100, 100))
    random_rotate(sample)


def test_RandomRotateDegrees_on_pil_image() -> None:
    all_degrees: List[Union[float, Tuple[float, float]]] = [0, 1, 45, (0, 0), (-15, 30)]
    for degrees in all_degrees:
        random_rotate = RandomRotateDegrees(prob=0.5, degrees=degrees)
        sample = Image.new("RGB", (100, 100))
        random_rotate(sample)


def test_random_rotation_transform() -> None:
    transform = random_rotation_transform(rr_prob=1.0, rr_degrees=None)
    assert isinstance(transform, RandomRotate)
    transform = random_rotation_transform(rr_prob=1.0, rr_degrees=45)
    assert isinstance(transform, RandomRotateDegrees)
    transform = random_rotation_transform(rr_prob=1.0, rr_degrees=(30, 45))
    assert isinstance(transform, RandomRotateDegrees)
