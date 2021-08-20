from typing import List, Tuple

from lightly.active_learning.utils import BoundingBox


def read_yolo_label_file(filepath: str, padding: float, separator: str = ' ') -> Tuple[List[int], List[BoundingBox]]:
    """Reads a file in the yolo file format

    Args:
        filepath:
            The path to the yolo file, usually a .txt file.
        padding:
            The relative passepartout / padding to be added around the bounding box
        separator:
            The separator character between the 5 values (class id, x, y, w, h) per row in the label file.

    Returns:
        Two lists, each with one element per row in the label file:
            The class indices.
            The bounding boxes.

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

        w_norm *= 1+padding
        h_norm *= 1+padding
        bbox = BoundingBox.from_yolo_label(x_norm, y_norm, w_norm, h_norm)
        bounding_boxes.append(bbox)
    return class_indices, bounding_boxes
