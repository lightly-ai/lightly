""" Check for Corrupt Images """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import os
from typing import *
from PIL import Image
import tqdm.contrib.concurrent as concurrent
from lightly.data import LightlyDataset


def check_images(data_dir: str) -> Tuple[List[str], List[str]]:
    '''Iterate through a directory of images and find corrupt images

    Args:
        data_dir: Path to the directory containing the images

    Returns:
        (healthy_images, corrupt_images)
    '''
    dataset = LightlyDataset(input_dir=data_dir)
    filenames = dataset.get_filenames()

    def _is_corrupt(filename):
        image = Image.open(
            os.path.join(data_dir, filename)
        )
        try:
            image.load()
        except IOError:
            return True
        else:
            return False

    mapped = concurrent.thread_map(
        _is_corrupt,
        filenames,
        chunksize=min(32, len(filenames))
    )
    healthy_images = [f for f, is_corrupt
                      in zip(filenames, mapped) if not is_corrupt]
    corrupt_images = [f for f, is_corrupt
                      in zip(filenames, mapped) if is_corrupt]
    return healthy_images, corrupt_images
