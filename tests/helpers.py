from typing import Any, List

from torch import Tensor


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
