# -*- coding: utf-8 -*-
"""**Lightly Train:** Train a self-supervised model from the command-line.

This module contains the entrypoint for the **lightly-train**
command-line interface.
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved
import os.path
from pathlib import Path
from typing import List

import hydra
from PIL.Image import Image
from torch.utils.hipify.hipify_python import bcolors

from lightly.cli._helpers import fix_input_path
from lightly.cropping.crop_image_by_bounding_boxes import crop_image_by_bounding_boxes
from lightly.cropping.read_yolo_label_file import read_yolo_label_file
from lightly.data import LightlyDataset


def _crop_cli(cfg, is_cli_call=True):
    input_dir = cfg['input_dir']
    if input_dir and is_cli_call:
        input_dir = fix_input_path(input_dir)
    output_dir = cfg['output_dir']
    if output_dir and is_cli_call:
        output_dir = fix_input_path(output_dir)


    dataset = LightlyDataset(input_dir)
    filenames_images = dataset.get_filenames()
    cropped_images_list_list: List[List[Image]] = []
    for filename_image in filenames_images:
        filepath_image = dataset.get_filepath_from_filename(filename_image)
        filepath_label = filepath_image.replace('jpg', 'txt')
        filepath_out_dir = os.path.join(output_dir, filepath_image).replace('jpg','')
        Path(filepath_out_dir).mkdir(parents=True, exist_ok=True)

        class_indices, bounding_boxes = read_yolo_label_file(filepath_label)
        cropped_images = crop_image_by_bounding_boxes(filepath_image, bounding_boxes)
        cropped_images_list_list.append(cropped_images)
        for index, class_index, cropped_image in enumerate((zip(class_indices, cropped_images))):
            cropped_image_filename = os.path.join(filepath_out_dir, f'{index}_{class_index}')
            cropped_image.save(cropped_image_filename)


    print(f'Cropped images are stored at: {bcolors.OKBLUE}{output_dir}{bcolors.ENDC}')
    return cropped_images_list_list



@hydra.main(config_path="config", config_name="config")
def crop_cli(cfg):
    """TODO

    Args:
        cfg:
            The default configs are loaded from the config file.
            To overwrite them please see the section on the config file
            (.config.config.yaml).

    Command-Line Args:
        input_dir:
            Path to the input directory where images are stored.

    Examples:
        TODO
        >>> # train model with default settings
        >>> lightly-crop input_dir=data_root/images labels_dir=data_root/labels output_dir=data_root/images_cropped padding=0.1
    """
    return _crop_cli(cfg)


def entry():
    crop_cli()
