from typing import List, Tuple

from lightly.active_learning.utils import BoundingBox


def read_yolo_label_file(filepath: str, separator: str = ' ') -> Tuple[List[int], List[BoundingBox]]:
    """

    Args:
        filepath:
        separator:

    Returns:

    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    class_indices = []
    bounding_boxes = []
    for line in lines:
        values = line.split(sep=separator)
        class_id, x_norm, y_norm, w_norm, h_norm = (float(val) for val in values)
        class_id = int(class_id)
        class_indices.append(class_id)
        bbox = BoundingBox.from_yolo(x_norm, y_norm, w_norm, h_norm)
        bounding_boxes.append(bbox)
    return class_indices, bounding_boxes
