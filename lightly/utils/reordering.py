from typing import List, Sized


def sort_items_by_keys(
    keys: List[any], 
    items: List[any], 
    sorted_keys: List[any]
):
    """Sorts the items in the same order as the sorted keys.

    Args:
        keys:
            Keys by which items can be identified.
        items:
            Items to sort.
        sorted_keys:
            Keys in sorted order.

    Returns:
        The list of sorted items.

    Examples:
        >>> keys = [3, 2, 1]
        >>> items = ['!', 'world', 'hello']
        >>> sorted_keys = [1, 2, 3]
        >>> sorted_items = sort_items_by_keys(
        >>>     keys,
        >>>     items,
        >>>     sorted_keys,
        >>> )
        >>> print(sorted_items)
        >>> > ['hello', 'world', '!']

    """
    if len(keys) != len(items) or len(keys) != len(sorted_keys):
        raise ValueError(f"All inputs (keys,  items and sorted_keys) "
                         f"must have the same length, "
                         f"but their lengths are: ({len(keys)},"
                         f"{len(items)} and {len(sorted_keys)}).")
    lookup = {key_: item_ for key_, item_ in zip(keys, items)}
    sorted_ = [lookup[key_] for key_ in sorted_keys]
    return sorted_
