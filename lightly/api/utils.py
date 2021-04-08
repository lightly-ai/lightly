""" Communication Utility """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import io
import os
import time
import random

import numpy as np
from PIL import Image, ImageFilter
# the following two lines are needed because
# PIL misidentifies certain jpeg images as MPOs
from PIL import JpegImagePlugin

JpegImagePlugin._getmp = lambda: None

LEGAL_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'tiff', 'bmp']
MAXIMUM_FILENAME_LENGTH = 80
MAX_WIDTH, MAX_HEIGHT = 2048, 2048


def retry(func, *args, **kwargs):
    """Repeats a function until it completes successfully or fails too often.

    Args:
        func:
            The function call to repeat.
        args:
            The arguments which are passed to the function.
        kwargs:
            Key-word arguments which are passed to the function.

    Returns:
        What func returns.

    Exceptions:
        RuntimeError when number of retries has been exceeded.

    """

    # config
    backoff = 1. + random.random() * 0.1
    max_backoff = 32
    max_retries = 5

    # try to make the request
    for i in range(max_retries):
        try:
            # return on success
            return func(*args, **kwargs)
        except Exception:
            # sleep on failure
            time.sleep(backoff)
            backoff = 2 * backoff if backoff < max_backoff else backoff
        
    # max retries exceeded
    raise RuntimeError('The connection to the server timed out.')


def getenv(key: str, default: str):
    """Return the value of the environment variable key if it exists,
       or default if it doesn’t.

    """
    try:
        return os.getenvb(key.encode(), default.encode()).decode()
    except Exception:
        pass
    try:
        return os.getenv(key, default)
    except Exception:
        pass
    return default


def PIL_to_bytes(img, ext: str = 'png', quality: int = None):
    """Return the PIL image as byte stream. Useful to send image via requests.

    """
    bytes_io = io.BytesIO()
    if quality is not None:
        img.save(bytes_io, format=ext, quality=quality)
    else:
        subsampling = -1 if ext.lower() in ['jpg', 'jpeg'] else 0
        img.save(bytes_io, format=ext, quality=100, subsampling=subsampling)
    bytes_io.seek(0)
    return bytes_io


def image_mean(np_img: np.ndarray):
    """Return mean of each channel.

    """
    return np_img.mean(axis=(0, 1))


def image_std(np_img: np.ndarray):
    """Return standard deviation of each channel.

    """
    return np_img.std(axis=(0, 1))


def sum_of_values(np_img: np.ndarray):
    """Return the sum of the pixel values of each channel.

    """
    return np_img.sum(axis=(0, 1))


def sum_of_squares(np_img: np.ndarray):
    """Return the sum of the squared pixel values of each channel.

    """
    return (np_img ** 2).sum(axis=(0, 1))


def signal_to_noise_ratio(img, axis: int = None, ddof: int = 0):
    """Calculate the signal to noise ratio of the image.

    """
    np_img = np.asanyarray(img)
    mean = np_img.mean(axis=axis)
    std = np_img.std(axis=axis, ddof=ddof)
    return np.where(std == 0, 0, mean / std)


def sharpness(img):
    """Calculate the sharpness of the image using a Laplacian Kernel.

    """
    img_bw = img.convert('L')
    filtered = img_bw.filter(
        ImageFilter.Kernel(
            (3, 3),
            # Laplacian Kernel:
            (-1, -1, -1, -1, 8, -1, -1, -1, -1),
            1,
            0,
        )
    )
    return np.std(filtered)


def size_in_bytes(img):
    """Return the size of the image in bytes.

    """
    img_file = io.BytesIO()
    img.save(img_file, format='png')
    return img_file.tell()


def shape(np_img: np.ndarray):
    """Shape of the image as np.ndarray.

    """
    return np_img.shape


def get_meta_from_img(img):
    """Calculates metadata from PIL image.

        - Mean
        - Standard Deviation
        - Signal To Noise
        - Sharpness
        - Size in Bytes
        - Shape
        - Sum of Values
        - Sum of Squares

    Args:
        PIL Image.

    Returns:
        A dictionary containing the metadata of the image.
    """

    np_img = np.array(img) / 255.0
    metadata = {
        'mean': image_mean(np_img).tolist(),
        'std': image_std(np_img).tolist(),
        'snr': float(signal_to_noise_ratio(img)),
        'sharpness': float(sharpness(img)),
        'sizeInBytes': size_in_bytes(img),
        'shape': list(shape(np_img)),
        'sumOfValues': sum_of_values(np_img).tolist(),
        'sumOfSquares': sum_of_squares(np_img).tolist(),
    }
    return metadata


def get_thumbnail_from_img(img, size: int = 128):
    """Compute the thumbnail of the image.

    Args:
        img:
            PIL Image.
        size:
            Size of the thumbnail.

    Returns:
        Thumbnail as PIL Image.

    """
    thumbnail = img.copy()
    thumbnail.thumbnail((size, size), Image.LANCZOS)
    return thumbnail


def resize_image(image, max_width: int, max_height: int):
    """Resize the image if it is too large for the web-app.

    """
    width = image.width
    height = image.height
    new_image = image.copy()
    new_image.format = image.format

    width_factor = max_width / width
    height_factor = max_height / height
    factor = min(width_factor, height_factor)
    new_image.resize(
        (
            np.floor(factor * width).astype(int),
            np.floor(factor * height).astype(int)
        )
    )
    return new_image


def check_filename(basename):
    """Checks the length of the filename.

    Args:
        basename:
            Basename of the file.

    """
    return len(basename) <= MAXIMUM_FILENAME_LENGTH


def check_image(image):
    """Checks whether an image is corrupted or not.

    The function reports the metadata, and opens the file to check whether
    it is corrupt or not.

    Args:
        image:
            PIL image from which metadata will be computed.

    Returns:
        A dictionary of metadata of the image.
    """

    # try to load the image to see whether it's corrupted or not
    try:
        image.load()
        is_corrupted = False
        corruption = ''
    except IOError as e:
        is_corrupted = True
        corruption = e

    # calculate metadata from image
    if is_corrupted:
        metadata = { 'corruption': corruption }
    else:
        metadata = get_meta_from_img(image)
        metadata['corruption'] = ''

    metadata['is_corrupted'] = is_corrupted
    return metadata
