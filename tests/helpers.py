from typing import Any, List

from torch import Tensor

from lightly.data.sample import View


def assert_list_tensor(items: Any) -> List[Tensor]:
    """Makes sure that the input is a list of tensors.

    Should be used in tests where functions return Union[List[Tensor], List[Image]] and
    we want to make sure that the output is a list of tensors.

    Example:
        >>> output: Union[List[Tensor], List[Image]] = transform(images)
        >>> tensors: List[Tensor] = assert_list_tensor(output)

    """
    assert isinstance(items, list)
    assert all(isinstance(item, Tensor) for item in items)
    return items


def assert_list_view(items: Any) -> List[View]:
    """Makes sure that the input is a list of views holding tensors.

    Example:
        >>> views: List[View] = assert_list_view(transform(image))
        >>> views[0].data.shape
        torch.Size([3, 32, 32])

    """
    assert isinstance(items, list)
    assert all(isinstance(item, View) for item in items)
    assert all(isinstance(item.data, Tensor) for item in items)
    return items
