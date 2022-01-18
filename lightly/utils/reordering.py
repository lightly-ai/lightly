from typing import List, Sized


def sort_items_by_keys(
    keys: List[any], 
    items: List[any], 
    sorted_keys: List[any]
):
    if len(keys) != len(items) or len(keys) != len(sorted_keys):
        raise ValueError(f"All inputs (keys,  items and sorted_keys) "
                         f"must have the same length, "
                         f"but their lengths are: ({len(keys)},"
                         f"{len(items)} and {len(sorted_keys)}).")
    lookup = {key_: item_ for key_, item_ in zip(keys, items)}
    sorted_ = [lookup[key_] for key_ in sorted_keys]
    return sorted_
