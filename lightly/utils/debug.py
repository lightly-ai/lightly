""" Bounding Box Utils """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import List, Union

import torch
import torchvision
from PIL import Image

from lightly.data.collate import BaseCollateFunction, MultiViewCollateFunction

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = ModuleNotFoundError(
        "Matplotlib is not installed on your system. Please install it to use the "
        "plotting functionalities. You can install it with "
        "'pip install lightly[matplotlib]'."
    )
except ImportError as ex:
    # Matplotlib import can fail if an incompatible dateutil version is installed.
    plt = ex


def _check_matplotlib_available() -> None:
    """Raises an error if matplotlib is not available."""
    if isinstance(plt, Exception):
        raise plt


@torch.no_grad()
def std_of_l2_normalized(z: torch.Tensor) -> torch.Tensor:
    """Calculates the mean of the standard deviation of l2-normalized tensor.

    This function calculates the standard deviation of the l2-normalized tensor 
    along each dimension. It is used to assess whether learned representations 
    have collapsed to a constant vector.

    Args:
        z: 
            A torch tensor of shape (batch_size, dimension).

    Returns:
        The mean of the standard deviation of the l2-normalized tensor along 
        each dimension.

    Raises:
        ValueError: If the input tensor does not have exactly two dimensions.

    Example:
        >>> z = torch.randn(32, 128)
        >>> std = std_of_l2_normalized(z)
        >>> print(std)
    """
    if len(z.shape) != 2:
        raise ValueError(
            f"Input tensor must have two dimensions but has {len(z.shape)}!"
        )

    z_norm = torch.nn.functional.normalize(z, dim=1)
    return torch.std(z_norm, dim=0).mean()


def apply_transform_without_normalize(
    image: Image.Image,
    transform,
) -> Image.Image:
    """Applies the given transform to the image, skipping ToTensor and Normalize.

    This function applies the provided transformations to the image, but skips 
    the `ToTensor` and `Normalize` transformations for visualization purposes.

    Args:
        image:
            The input image (PIL Image) to which the transform will be applied.
        transform:
            A torchvision transform or a composition of transforms.

    Returns:
        The transformed image, with ToTensor and Normalize skipped.

    Example:
        >>> from torchvision import transforms
        >>> image = Image.open("example.jpg")
        >>> transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        >>> result_image = apply_transform_without_normalize(image, transform)
    """
    skippable_transforms = (
        torchvision.transforms.ToTensor,
        torchvision.transforms.Normalize,
    )
    if isinstance(transform, torchvision.transforms.Compose):
        for transform_ in transform.transforms:
            image = apply_transform_without_normalize(image, transform_)
    elif not isinstance(transform, skippable_transforms):
        image = transform(image)
    return image


def generate_grid_of_augmented_images(
    input_images: List[Image.Image],
    collate_function: Union[BaseCollateFunction, MultiViewCollateFunction],
) -> List[List[Image.Image]]:
    """Generates a grid of augmented images.

    This function creates a grid of augmented images where images in the same 
    column belong together. It skips the `ToTensor` and `Normalize` transformations 
    to facilitate visualization.

    Args:
        input_images:
            A list of PIL images for which the augmentations should be plotted.
        collate_function:
            The collate function of the self-supervised learning algorithm, must 
            be an instance of BaseCollateFunction or MultiViewCollateFunction.

    Returns:
        A grid of augmented images, where images in the same column belong 
        together.

    Raises:
        ValueError: If collate_function is not an instance of 
                    BaseCollateFunction or MultiViewCollateFunction.

    Example:
        >>> input_images = [Image.open("image1.jpg"), Image.open("image2.jpg")]
        >>> collate_fn = BaseCollateFunction()
        >>> grid = generate_grid_of_augmented_images(input_images, collate_fn)
    """
    grid = []
    if isinstance(collate_function, BaseCollateFunction):
        for _ in range(2):
            grid.append(
                [
                    apply_transform_without_normalize(image, collate_function.transform)
                    for image in input_images
                ]
            )
    elif isinstance(collate_function, MultiViewCollateFunction):
        for transform in collate_function.transforms:
            grid.append(
                [
                    apply_transform_without_normalize(image, transform)
                    for image in input_images
                ]
            )
    else:
        raise ValueError(
            "Collate function must be one of (BaseCollateFunction, MultiViewCollateFunction) "
            f"but is {type(collate_function)}."
        )
    return grid


def plot_augmented_images(
    input_images: List[Image.Image],
    collate_function: Union[BaseCollateFunction, MultiViewCollateFunction],
):
    """Plots original and augmented images.

    This function creates a figure showing the original images in the left column 
    and augmented images in the subsequent columns. It skips the `ToTensor` and 
    `Normalize` transformations for visualization purposes.

    Args:
        input_images:
            A list of PIL images for which the augmentations should be plotted.
        collate_function:
            The collate function of the self-supervised learning algorithm, must 
            be an instance of BaseCollateFunction or MultiViewCollateFunction.

    Returns:
        A matplotlib figure displaying the original and augmented images in a grid.

    Raises:
        ValueError: If no input images are provided.
        ModuleNotFoundError: If matplotlib is not installed.

    Example:
        >>> input_images = [Image.open("image1.jpg"), Image.open("image2.jpg")]
        >>> collate_fn = BaseCollateFunction()
        >>> fig = plot_augmented_images(input_images, collate_fn)
        >>> plt.show()
    """
    _check_matplotlib_available()

    if len(input_images) == 0:
        raise ValueError("There must be at least one input image.")

    grid = generate_grid_of_augmented_images(input_images, collate_function)
    grid.insert(0, input_images)
    nrows = len(input_images)
    ncols = len(grid)

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))

    for i in range(nrows):
        for j in range(ncols):
            ax = axs[i][j] if len(input_images) > 1 else axs[j]
            img = grid[j][i]
            ax.imshow(img)
            ax.set_axis_off()

    ax_top_left = axs[0, 0] if len(input_images) > 1 else axs[0]
    ax_top_left.set(title="Original images")
    ax_top_left.title.set_size(8)
    ax_top_next = axs[0, 1] if len(input_images) > 1 else axs[1]
    ax_top_next.set(title="Augmented images")
    ax_top_next.title.set_size(8)
    fig.tight_layout()

    return fig
