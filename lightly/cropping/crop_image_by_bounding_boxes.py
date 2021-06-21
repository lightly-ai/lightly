from typing import List

from PIL import Image

from lightly.active_learning.utils import BoundingBox


def crop_image_by_bounding_boxes(image_filepath: str, bounding_boxes: List[BoundingBox]) -> List[Image.Image]:
    image = Image.open(image_filepath)
    cropped_images = []
    for bbox in bounding_boxes:
        w, h = image.size
        crop_box = (w * bbox.x0, h * bbox.y0, w * bbox.x1, h * bbox.y1)
        crop_box = tuple(int(i) for i in crop_box)
        cropped_image = image.crop(crop_box)
        cropped_images.append(cropped_image)
    return cropped_images
