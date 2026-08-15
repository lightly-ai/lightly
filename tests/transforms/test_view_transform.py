from PIL import Image

from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.transforms.view_transform import ViewTransform

from .. import helpers


def test_every_transform_makes_one_view() -> None:
    transform = ViewTransform(
        [T.RandomHorizontalFlip(p=0.1), T.RandomVerticalFlip(p=0.5), T.ToTensor()]
    )
    views = transform(Image.new("RGB", (10, 10)))
    assert len(views) == 3


def test_a_view_carries_the_default_role_and_stream() -> None:
    transform = ViewTransform([T.ToTensor(), T.ToTensor()])
    views = helpers.assert_list_view(transform(Image.new("RGB", (10, 10))))
    assert [view.role for view in views] == ["view", "view"]
    assert [view.stream for view in views] == ["image", "image"]
    assert all(view.extras == {} for view in views)


def test_extra_arguments_reach_every_transform() -> None:
    def take_two(image: str, mask: str) -> str:
        return image + mask

    transform = ViewTransform([take_two, take_two])
    views = transform("image", "mask")
    assert [view.data for view in views] == ["imagemask", "imagemask"]
