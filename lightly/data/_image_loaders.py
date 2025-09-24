"""Module for handling image loading in torchvision-compatible format.

This module provides image loading functionality similar to torchvision's implementation
(see https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html)
"""

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from PIL import Image


def pil_loader(path: str) -> Image.Image:
    """Loads an image using PIL.

    Args:
        path: Path to the image file.

    Returns:
        A PIL Image in RGB format.
    """
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path: str) -> Image.Image:
    """Loads an image using the accimage library for faster loading.

    Falls back to PIL loader if accimage fails to load the image.

    Args:
        path: Path to the image file.

    Returns:
        An image loaded either by accimage or PIL in case of failure.
    """
    try:
        import accimage

        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Image.Image:
    """Loads an image using the default backend specified in torchvision.

    Uses accimage if available and configured as the backend, otherwise falls back to PIL.

    Args:
        path: Path to the image file.

    Returns:
        An image loaded by either accimage or PIL depending on the backend.
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
