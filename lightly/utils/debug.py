from typing import List, Union
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision

from lightly.data.collate import BaseCollateFunction, MultiViewCollateFunction, SimCLRCollateFunction, DINOCollateFunction, SwaVCollateFunction


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
    transform #.TODO: typehint: transforms.Transform,
):
    """TODO"""
    # TODO: change this
    if isinstance(transform, torchvision.transforms.Compose):
        for transform_ in transform.transforms:
            if isinstance(transform_, torchvision.transforms.Compose):
                image = apply_transform_without_normalize(image, transform_)
                continue
            if isinstance(transform_, torchvision.transforms.Normalize):
                continue
            if isinstance(transform_, torchvision.transforms.ToTensor):
                continue
            print(transform_)
            image = transform_(image)
    else:
        image = transform_(image)
    return image



def generate_grid_of_augmented_images(
    input_images: List[Image.Image],
    collate_function: Union[BaseCollateFunction, MultiViewCollateFunction],
):
    """TODO"""

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
        raise ValueError('TODO')
    return grid


def plot_augmented_images(
    input_images: List[Image.Image],
    collate_function: Union[BaseCollateFunction, MultiViewCollateFunction],
):
    """TODO"""

    grid = generate_grid_of_augmented_images(input_images, collate_function)
    nrows = len(grid) + 1 # extra row for the original images
    ncols = len(input_images)

    fig, axs = plt.subplots(nrows, ncols)

    grid.insert(0, input_images)
    for i in range(nrows):
        for ax, img in zip(axs[i], grid[i]):
            ax.imshow(img)
            ax.set_axis_off()

    fig.tight_layout()
    return fig


if __name__ == '__main__':

    import numpy
    from PIL import Image

    input_images = []
    for i in range(5):
        imarray = numpy.random.rand(100,100,3) * 255
        im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
        input_images.append(im)

    collate_function = DINOCollateFunction()

    fig = plot_augmented_images(input_images, collate_function)
    fig.savefig('hello.png')