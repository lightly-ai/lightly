""" Communication Utility """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import io
import os
import time
import random

import numpy as np
import requests
import warnings
from PIL import Image, ImageFilter

# the following two lines are needed because
# PIL misidentifies certain jpeg images as MPOs
from PIL import JpegImagePlugin
JpegImagePlugin._getmp = lambda: None

LEGAL_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'tiff', 'bmp']
MAXIMUM_FILENAME_LENGTH = 80
MAX_WIDTH, MAX_HEIGHT = 2048, 2048


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


def post_request(dst_url, data=None, json=None,
                 max_backoff=32, max_retries=5):
    """Makes a POST request with random offset retries.

    Args:
        dst_url:
            Url the GET request is made to.
        data:
            POST data.
        json:
            POST json.
        max_backoff:
            Maximal backoff before throwing timing out.
        max_retries:
            Maximum number of retries before timing out.

    Returns:
        The server response.

    Raises:
        RuntimeError if the request fails.

    """
    counter = 0
    backoff = 1. + random.random() * 0.1
    success = False
    while not success:

        response = requests.post(dst_url, data=data, json=json)
        success = (response.status_code == 200)

        # exponential backoff
        if response.status_code in [500, 502]:
            time.sleep(backoff)
            backoff = 2*backoff if backoff < max_backoff else backoff
        elif response.status_code in [402]:
            msg = 'Dataset limit reached. Failed to upload samples. '
            msg += 'Contact your account manager to upgrade your subscription'
            raise ConnectionRefusedError(msg)
        # something went wrong
        elif not success:
            msg = f'Failed POST request to {dst_url} with status_code '
            msg += f'{response.status_code}.'
            raise RuntimeError(msg)

        counter += 1
        if counter >= max_retries:
            break

    if not success:
        msg = f'The connection to the server at {dst_url} timed out. '
        raise RuntimeError(msg)

    return response


def get_request(dst_url, params=None,
                max_backoff=32, max_retries=5):
    """Makes a GET request with random offset retries.

    Args:
        dst_url:
            Url the GET request is made to.
        params:
            GET parameters.
        max_backoff:
            Maximal backoff before throwing timing out.
        max_retries:
            Maximum number of retries before timing out.

    Returns:
        The server response.

    Raises:
        RuntimeError if the request fails.

    """
    counter = 0
    backoff = 1. + random.random() * 0.1
    success = False
    while not success:

        response = requests.get(dst_url, params=params)
        success = (response.status_code == 200)

        # exponential backoff
        if response.status_code in [500, 502]:
            time.sleep(backoff)
            backoff = 2*backoff if backoff < max_backoff else backoff
        # something went wrong
        elif not success:
            msg = f'Failed GET request to {dst_url} with status_code '
            msg += f'{response.status_code}.'
            raise RuntimeError(msg)

        counter += 1
        if counter >= max_retries:
            break

    if not success:
        msg = f'The connection to the server at {dst_url} timed out. '
        raise RuntimeError(msg)

    return response


def put_request(dst_url, data=None, params=None, json=None,
                max_backoff=32, max_retries=5):
    """Makes a PUT request with random offset retries.

    Args:
        dst_url:
            Url the PUT request is made to.
        data:
            PUT data.
        params:
            PUT parameters.
        json:
            PUT json.
        max_backoff:
            Maximal backoff before throwing timing out.
        max_retries:
            Maximum number of retries before timing out.

    Returns:
        The server response.

    Raises:
        RuntimeError if the request fails.

    """
    counter = 0
    backoff = 1. + random.random() * 0.1
    success = False
    while not success:

        response = requests.put(dst_url, data=data, json=json, params=params)
        success = (response.status_code == 200)

        # exponential backoff
        if response.status_code in [500, 502]:
            time.sleep(backoff)
            backoff = 2*backoff if backoff < max_backoff else backoff
        # something went wrong
        elif not success:
            msg = f'Failed PUT request to {dst_url} with status_code '
            msg += f'{response.status_code}.'
            raise RuntimeError(msg)

        counter += 1
        if counter >= max_retries:
            break

    if not success:
        msg = f'The connection to the server at {dst_url} timed out. '
        raise RuntimeError(msg)

    return response
