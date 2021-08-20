import os.path
import warnings
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm

from lightly.active_learning.utils import BoundingBox
from lightly.data import LightlyDataset


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


def crop_dataset_by_bounding_boxes_and_save(dataset: LightlyDataset,
                                            output_dir: str,
                                            bounding_boxes_list_list: List[List[BoundingBox]],
                                            class_indices_list_list: List[List[int]],
                                            class_names: List[str] = None
                                            ) -> List[List[str]]:
    """Crops all images in a dataset by the bounding boxes and saves them in the output dir

    Args:
        dataset:
            The dataset with the images to be cropped. Must contain M images.
        output_dir:
            The output directory to saved the cropped images to.
        bounding_boxes_list_list:
            The bounding boxes of the detections for each image. Must have M sublists, one for each image.
            Each sublist contains the bounding boxes for each detection, thus N_m elements.
        class_indices_list_list:
            The object class ids of the detections for each image. Must have M sublists, one for each image.
            Each sublist contains the bounding boxes for each detection, thus N_m elements.
        class_names:
            The names of the classes, used to map the class id to the class name.


    Returns:
        The filepaths to all saved cropped images. Has M sublists, one for each image.
        Each sublist contains the filepath of the crop each detection, thus N_m elements.

    """
    filenames_images = dataset.get_filenames()
    if len(filenames_images) != len(bounding_boxes_list_list) or len(filenames_images) != len(class_indices_list_list):
        raise ValueError("There must be one bounding box and class index list for each image in the datasets,"
                         "but the lengths dont align.")

    cropped_image_filepath_list_list: List[List[Image]] = []


    print(f"Cropping objects out of {len(filenames_images)} images...")
    for filename_image, class_indices, bounding_boxes in \
            tqdm(zip(filenames_images, class_indices_list_list, bounding_boxes_list_list)):

        if not len(class_indices) == len(bounding_boxes):
            warnings.warn(UserWarning(f"Length of class indices ({len(class_indices)} does not equal length of bounding boxes"
                          f"({len(bounding_boxes)}. This is an error in the input arguments. "
                          f"Skipping this image {filename_image}."))
            continue

        filepath_image = dataset.get_filepath_from_filename(filename_image)
        filepath_image_base, image_extension = os.path.splitext(filepath_image)

        filepath_out_dir = os.path.join(output_dir, filename_image).replace(image_extension, '')
        Path(filepath_out_dir).mkdir(parents=True, exist_ok=True)

        cropped_images = crop_image_by_bounding_boxes(filepath_image, bounding_boxes)
        cropped_images_filepaths = []
        for index, (class_index, cropped_image) in enumerate((zip(class_indices, cropped_images))):
            if class_names:
                class_name = class_names[class_index]
            else:
                class_name = f"class{class_index}"
            cropped_image_last_filename = f'{index}_{class_name}{image_extension}'
            cropped_image_filepath = os.path.join(filepath_out_dir, cropped_image_last_filename)
            cropped_image.save(cropped_image_filepath)

            cropped_image_filename = os.path.join(filename_image.replace(image_extension, ''), cropped_image_last_filename)
            cropped_images_filepaths.append(cropped_image_filename)

        cropped_image_filepath_list_list.append(cropped_images_filepaths)

    return cropped_image_filepath_list_list
