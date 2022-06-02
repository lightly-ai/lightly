from typing import List, Union
from PIL import Image

import torch
import torchvision

from lightly.data.collate import BaseCollateFunction, MultiViewCollateFunction, DINOCollateFunction, SimCLRCollateFunction

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = ModuleNotFoundError(
        "Matplotlib is not installed on your system. Please install it to use the plotting"
        "functionalities. See https://matplotlib.org/ for installation instructions."
    )

def _check_matplotlib_available() -> None:
    if isinstance(plt, Exception):
        raise plt


@torch.no_grad()
def std_of_l2_normalized(z: torch.Tensor):
    """Calculates the mean of the standard deviation of z along each dimension.

    This measure was used by [0] to determine the level of collapse of the
    learned representations. If the returned number is 0., the outputs z have
    collapsed to a constant vector. "If the output z has a zero-mean isotropic
    Gaussian distribution" [0], the returned number should be close to 1/sqrt(d)
    where d is the dimensionality of the output.

    [0]: https://arxiv.org/abs/2011.10566

    Args:
        z:
            A torch tensor of shape batch_size x dimension.

    Returns:
        The mean of the standard deviation of the l2 normalized tensor z along
        each dimension.
    
    """

    if len(z.shape) != 2:
        raise ValueError(
            f'Input tensor must have two dimensions but has {len(z.shape)}!'
        )

    _, d = z.shape

    z_norm = torch.nn.functional.normalize(z, dim=1)
    return torch.std(z_norm, dim=0).mean()


def apply_transform_without_normalize(
    image: Image.Image,
    transform,
):
    """Applies the transform to the image but skips ToTensor and Normalize.

    """
    if isinstance(transform, torchvision.transforms.Compose):
        for transform_ in transform.transforms:
            if isinstance(transform_, torchvision.transforms.Normalize):
                continue
            elif isinstance(transform_, torchvision.transforms.ToTensor):
                continue
            elif isinstance(transform_, torchvision.transforms.Compose):
                image = apply_transform_without_normalize(image, transform_)
            else:
                image = transform_(image)
    else:
        image = transform(image)
    return image


def generate_grid_of_augmented_images(
    input_images: List[Image.Image],
    collate_function: Union[BaseCollateFunction, MultiViewCollateFunction],
):
    """Returns a grid of augmented images. Images in a column belong together.

    This function ignores the transforms ToTensor and Normalize for visualization purposes.

    Args:
        input_images:
            List of PIL images for which the augmentations should be plotted.
        collate_function:
            The collate function of the self-supervised learning algorithm.
            Must be of type BaseCollateFunction or MultiViewCollateFunction.

    Returns:
        A grid of augmented images. Images in a column belong together.

    """
    grid = []
    if isinstance(collate_function, BaseCollateFunction):
        for _ in range(2):
            grid.append([
                apply_transform_without_normalize(image, collate_function.transform)
                for image in input_images
            ])
    elif isinstance(collate_function, MultiViewCollateFunction):
        for transform in collate_function.transforms:
            grid.append([
                apply_transform_without_normalize(image, transform)
                for image in input_images
            ])
    else:
        raise ValueError(
            'Collate function must be one of '
            '(BaseCollateFunction, MultiViewCollateFunction) '
            f'but is {type(collate_function)}.'
        )
    return grid


def plot_augmented_images(
    input_images: List[Image.Image],
    collate_function: Union[BaseCollateFunction, MultiViewCollateFunction],
):
    """Returns a figure showing original images in the left column and augmented images to their right.

    This function ignores the transforms ToTensor and Normalize for visualization purposes.

    Args:
        input_images:
            List of PIL images for which the augmentations should be plotted.
        collate_function:
            The collate function of the self-supervised learning algorithm.
            Must be of type BaseCollateFunction or MultiViewCollateFunction.

    Returns:
        A figure showing the original images in the left column and the augmented
        images to their right. If the collate_function is an instance of the
        BaseCollateFunction, two example augmentations are shown. For
        MultiViewCollateFunctions all the generated views are shown.

    """

    _check_matplotlib_available()

    if len(input_images) == 0:
        raise ValueError('There must be at least one input image.')

    grid = generate_grid_of_augmented_images(input_images, collate_function)
    nrows = len(input_images)
    ncols = len(grid) + 1 # extra column for the original images

    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols * 1.5, nrows * 1.5))

    grid.insert(0, input_images)
    for i in range(nrows):
        for j in range(ncols):
            ax = axs[i][j]
            img = grid[j][i]
            ax.imshow(img)
            ax.set_axis_off()

    axs[0, 0].set(title='Original images')
    axs[0, 0].title.set_size(8)
    axs[0, 1].set(title='Augmented images')
    axs[0, 1].title.set_size(8)
    fig.tight_layout()

    return fig


if __name__ == '__main__':

    import numpy
    from PIL import Image

    input_images_ = []
    for i in range(2):
        imarray = numpy.random.rand(100,100,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        input_images_.append(im)

    collate_function = SimCLRCollateFunction()

    fig = plot_augmented_images(input_images_, collate_function)
    fig.savefig('hello.png')