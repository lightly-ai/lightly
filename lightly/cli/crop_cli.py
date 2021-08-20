# -*- coding: utf-8 -*-
"""**Lightly Train:** Train a self-supervised model from the command-line.

This module contains the entrypoint for the **lightly-train**
command-line interface.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
import os.path
from typing import List

import hydra
import yaml
from torch.utils.hipify.hipify_python import bcolors

from lightly.active_learning.utils import BoundingBox
from lightly.cli._helpers import fix_input_path
from lightly.data import LightlyDataset
from lightly.utils.cropping.crop_image_by_bounding_boxes import crop_dataset_by_bounding_boxes_and_save
from lightly.utils.cropping.read_yolo_label_file import read_yolo_label_file


def _crop_cli(cfg, is_cli_call=True):
    input_dir = cfg['input_dir']
    if input_dir and is_cli_call:
        input_dir = fix_input_path(input_dir)
    output_dir = cfg['output_dir']
    if output_dir and is_cli_call:
        output_dir = fix_input_path(output_dir)
    label_dir = cfg['label_dir']
    if label_dir and is_cli_call:
        label_dir = fix_input_path(label_dir)
    label_names_file = cfg['label_names_file']
    if label_names_file and len(label_names_file) > 0:
        if is_cli_call:
            label_names_file = fix_input_path(label_names_file)
        with open(label_names_file, 'r') as file:
            label_names_file_dict = yaml.full_load(file)
        class_names = label_names_file_dict['names']
    else:
        class_names = None


    dataset = LightlyDataset(input_dir)


    class_indices_list_list: List[List[int]] = []
    bounding_boxes_list_list: List[List[BoundingBox]] = []

    # YOLO-Specific
    for filename_image in dataset.get_filenames():
        filepath_image_base, image_extension = os.path.splitext(filename_image)
        filepath_label = os.path.join(label_dir, filename_image).replace(image_extension, '.txt')
        class_indices, bounding_boxes = read_yolo_label_file(filepath_label, float(cfg['crop_padding']))
        class_indices_list_list.append(class_indices)
        bounding_boxes_list_list.append(bounding_boxes)

    cropped_images_list_list = \
        crop_dataset_by_bounding_boxes_and_save(dataset, output_dir, bounding_boxes_list_list, class_indices_list_list, class_names)

    print(f'Cropped images are stored at: {bcolors.OKBLUE}{output_dir}{bcolors.ENDC}')
    return cropped_images_list_list


@hydra.main(config_path="config", config_name="config")
def crop_cli(cfg):
    """Crops images into one sub-image for each object.

    Args:
        cfg:
            The default configs are loaded from the config file.
            To overwrite them please see the section on the config file
            (.config.config.yaml).

    Command-Line Args:
        input_dir:
            Path to the input directory where images are stored.
        labels_dir:
            Path to the directory where the labels are stored. There must be one label file for each image.
            The label file must have the same name as the image file, but the extension .txt.
            For example, img_123.txt for img_123.jpg. The label file must be in YOLO format.
        output_dir:
            Path to the directory where the cropped images are stored. They are stored in one directory per input image.
        crop_padding: Optional
            The additonal padding about the bounding box. This makes the crops include the context of the object.
            The padding is relative and added to the width and height.
        label_names_file: Optional
            A yaml file including the names of the classes. If it is given, the filenames of the cropped images include
            the class names instead of the class id. This file is usually included when having a dataset in yolo format.
            Example contents of such a label_names_file.yaml: "names: ['class_name_a', 'class_name_b']"


    Examples:
        >>> # Crop images and set the crop to be 20% around the bounding box
        >>> lightly-crop input_dir=data/images label_dir=data/labels output_dir=data/cropped_images crop_padding=0.2

        >>> # Crop images and use the class names in the filename
        >>> lightly-crop input_dir=data/images label_dir=data/labels output_dir=data/cropped_images label_names_file=data/data.yaml

    """
    return _crop_cli(cfg)


def entry():
    crop_cli()
