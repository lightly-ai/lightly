""" Collate Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import math
from multiprocessing import Value
from typing import List, Optional, Tuple, Union
from warnings import warn

import torch
import torch.nn as nn
import torchvision
from PIL import Image

from lightly.transforms import GaussianBlur, Jigsaw, RandomSolarization
from lightly.transforms.random_crop_and_flip_with_grid import RandomResizedCropAndFlip
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE

imagenet_normalize = IMAGENET_NORMALIZE
# Kept for backwards compatibility


class BaseCollateFunction(nn.Module):
    """Base class for other collate implementations.

    Takes a batch of images as input and transforms each image into two
    different augmentations with the help of random transforms. The images are
    then concatenated such that the output batch is exactly twice the length
    of the input batch.

    Attributes:
        transform:
            A set of torchvision transforms which are randomly applied to
            each image.

    """

    def __init__(self, transform: T.Compose):
        _deprecation_warning_collate_functions()
        super(BaseCollateFunction, self).__init__()
        self.transform = transform

    def forward(self, batch: List[Tuple[Image.Image, int, str]]):
        """Turns a batch of tuples into a tuple of batches.

        Args:
            batch:
                A batch of tuples of images, labels, and filenames which
                is automatically provided if the dataloader is built from
                a LightlyDataset.

        Returns:
            A tuple of images, labels, and filenames. The images consist of
            two batches corresponding to the two transformations of the
            input images.

        Examples:
            >>> # define a random transformation and the collate function
            >>> transform = ... # some random augmentations
            >>> collate_fn = BaseCollateFunction(transform)
            >>>
            >>> # input is a batch of tuples (here, batch_size = 1)
            >>> input = [(img, 0, 'my-image.png')]
            >>> output = collate_fn(input)
            >>>
            >>> # output consists of two random transforms of the images,
            >>> # the labels, and the filenames in the batch
            >>> (img_t0, img_t1), label, filename = output

        """
        batch_size = len(batch)

        # list of transformed images
        transforms = [
            self.transform(batch[i % batch_size][0]).unsqueeze_(0)
            for i in range(2 * batch_size)
        ]
        # list of labels
        labels = torch.LongTensor([item[1] for item in batch])
        # list of filenames
        fnames = [item[2] for item in batch]

        # tuple of transforms
        transforms = (
            torch.cat(transforms[:batch_size], 0),
            torch.cat(transforms[batch_size:], 0),
        )

        return transforms, labels, fnames


class ImageCollateFunction(BaseCollateFunction):
    """Implementation of a collate function for images.

    This is an implementation of the BaseCollateFunction with a concrete
    set of transforms.

    The set of transforms is inspired by the SimCLR paper as it has shown
    to produce powerful embeddings.

    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        input_size: int = 64,
        cj_prob: float = 0.8,
        cj_bright: float = 0.7,
        cj_contrast: float = 0.7,
        cj_sat: float = 0.7,
        cj_hue: float = 0.2,
        min_scale: float = 0.15,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.2, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: dict = imagenet_normalize,
    ):
        if isinstance(input_size, tuple):
            input_size_ = max(input_size)
        else:
            input_size_ = input_size

        color_jitter = T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            T.ToTensor(),
        ]

        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]

        transform = T.Compose(transform)

        super(ImageCollateFunction, self).__init__(transform)


class MultiViewCollateFunction(nn.Module):
    """Generates multiple views for each image in the batch.

    Attributes:
        transforms:
            List of transformation functions. Each function is used to generate
            one view of the back.

    """

    def __init__(self, transforms: List[T.Compose]):
        _deprecation_warning_collate_functions()
        super().__init__()
        self.transforms = transforms

    def forward(self, batch: List[tuple]):
        """Turns a batch of tuples into a tuple of batches.

        Args:
            batch:
                The input batch.

        Returns:
            A (views, labels, fnames) tuple where views is a list of tensors
            with each tensor containing one view of the batch.

        """
        views = []
        for transform in self.transforms:
            view = torch.stack([transform(img) for img, _, _ in batch])
            views.append(view)
        # list of labels
        labels = torch.LongTensor([label for _, label, _ in batch])
        # list of filenames
        fnames = [fname for _, _, fname in batch]
        return views, labels, fnames


class SimCLRCollateFunction(ImageCollateFunction):
    """Implements the transformations for SimCLR.

    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    Examples:

        >>> # SimCLR for ImageNet
        >>> collate_fn = SimCLRCollateFunction()
        >>>
        >>> # SimCLR for CIFAR-10
        >>> collate_fn = SimCLRCollateFunction(
        >>>     input_size=32,
        >>>     gaussian_blur=0.,
        >>> )

    """

    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.2, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: dict = imagenet_normalize,
    ):
        super(SimCLRCollateFunction, self).__init__(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_bright=cj_strength * 0.8,
            cj_contrast=cj_strength * 0.8,
            cj_sat=cj_strength * 0.8,
            cj_hue=cj_strength * 0.2,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            normalize=normalize,
        )


class MoCoCollateFunction(ImageCollateFunction):
    """Implements the transformations for MoCo v1.

    For MoCo v2, simply use the SimCLR settings.

    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    Examples:

        >>> # MoCo v1 for ImageNet
        >>> collate_fn = MoCoCollateFunction()
        >>>
        >>> # MoCo v1 for CIFAR-10
        >>> collate_fn = MoCoCollateFunction(
        >>>     input_size=32,
        >>> )

    """

    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 0.4,
        min_scale: float = 0.2,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.0,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.2, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: dict = imagenet_normalize,
    ):
        super(MoCoCollateFunction, self).__init__(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_bright=cj_strength,
            cj_contrast=cj_strength,
            cj_sat=cj_strength,
            cj_hue=cj_strength,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            kernel_size=kernel_size,
            sigmas=sigmas,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            normalize=normalize,
        )


class MultiCropCollateFunction(MultiViewCollateFunction):
    """Implements the multi-crop transformations for SwaV.

    Attributes:
        crop_sizes:
            Size of the input image in pixels for each crop category.
        crop_counts:
            Number of crops for each crop category.
        crop_min_scales:
            Min scales for each crop category.
        crop_max_scales:
            Max_scales for each crop category.
        transforms:
            Transforms which are applied to all crops.

    """

    def __init__(
        self,
        crop_sizes: List[int],
        crop_counts: List[int],
        crop_min_scales: List[float],
        crop_max_scales: List[float],
        transforms: T.Compose,
    ):
        if len(crop_sizes) != len(crop_counts):
            raise ValueError(
                "Length of crop_sizes and crop_counts must be equal but are"
                f" {len(crop_sizes)} and {len(crop_counts)}."
            )
        if len(crop_sizes) != len(crop_min_scales):
            raise ValueError(
                "Length of crop_sizes and crop_min_scales must be equal but are"
                f" {len(crop_sizes)} and {len(crop_min_scales)}."
            )
        if len(crop_sizes) != len(crop_min_scales):
            raise ValueError(
                "Length of crop_sizes and crop_max_scales must be equal but are"
                f" {len(crop_sizes)} and {len(crop_min_scales)}."
            )

        crop_transforms = []
        for i in range(len(crop_sizes)):
            random_resized_crop = T.RandomResizedCrop(
                crop_sizes[i], scale=(crop_min_scales[i], crop_max_scales[i])
            )

            crop_transforms.extend(
                [
                    T.Compose(
                        [
                            random_resized_crop,
                            transforms,
                        ]
                    )
                ]
                * crop_counts[i]
            )
        super().__init__(crop_transforms)


class SwaVCollateFunction(MultiCropCollateFunction):
    """Implements the multi-crop transformations for SwaV.

    Attributes:
        crop_sizes:
            Size of the input image in pixels for each crop category.
        crop_counts:
            Number of crops for each crop category.
        crop_min_scales:
            Min scales for each crop category.
        crop_max_scales:
            Max_scales for each crop category.
        hf_prob:
            Probability that horizontal flip is applied.
        vf_prob:
            Probability that vertical flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    Examples:

        >>> # SwaV for Imagenet
        >>> collate_fn = SwaVCollateFunction()
        >>>
        >>> # SwaV w/ 2x160 and 4x96 crops
        >>> collate_fn = SwaVCollateFunction(
        >>>     crop_sizes=[160, 96],
        >>>     crop_counts=[2, 4],
        >>> )

    """

    def __init__(
        self,
        crop_sizes: List[int] = [224, 96],
        crop_counts: List[int] = [2, 6],
        crop_min_scales: List[float] = [0.14, 0.05],
        crop_max_scales: List[float] = [1.0, 0.14],
        hf_prob: float = 0.5,
        vf_prob: float = 0.0,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        cj_prob: float = 0.8,
        cj_strength: float = 0.8,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.2, 2),
        normalize: dict = imagenet_normalize,
    ):
        color_jitter = T.ColorJitter(
            cj_strength,
            cj_strength,
            cj_strength,
            cj_strength / 4.0,
        )

        transforms = T.Compose(
            [
                T.RandomHorizontalFlip(p=hf_prob),
                T.RandomVerticalFlip(p=vf_prob),
                random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
                T.ColorJitter(),
                T.RandomApply([color_jitter], p=cj_prob),
                T.RandomGrayscale(p=random_gray_scale),
                GaussianBlur(
                    kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur
                ),
                T.ToTensor(),
                T.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

        super(SwaVCollateFunction, self).__init__(
            crop_sizes=crop_sizes,
            crop_counts=crop_counts,
            crop_min_scales=crop_min_scales,
            crop_max_scales=crop_max_scales,
            transforms=transforms,
        )


class DINOCollateFunction(MultiViewCollateFunction):
    """Implements the global and local view augmentations for DINO [0].

    This class generates two global and a user defined number of local views
    for each image in a batch. The code is adapted from [1].

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: https://github.com/facebookresearch/dino

    Attributes:
        global_crop_size:
            Crop size of the global views.
        global_crop_scale:
            Tuple of min and max scales relative to global_crop_size.
        local_crop_size:
            Crop size of the local views.
        local_crop_scale:
            Tuple of min and max scales relative to local_crop_size.
        n_local_views:
            Number of generated local views.
        hf_prob:
            Probability that horizontal flip is applied.
        vf_prob:
            Probability that vertical flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        cj_prob:
            Probability that color jitter is applied.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Tuple of probabilities to apply gaussian blur on the different
            views. The input is ordered as follows:
            (global_view_0, global_view_1, local_views)
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        kernel_scale:
            Old argument. Value is deprecated in favor of sigmas. If set, the old behavior applies and `sigmas` is ignored.
            Used to scale the `kernel_size` of a factor of `kernel_scale`
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        solarization:
            Probability to apply solarization on the second global view.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        global_crop_size=224,
        global_crop_scale=(0.4, 1.0),
        local_crop_size=96,
        local_crop_scale=(0.05, 0.4),
        n_local_views=6,
        hf_prob=0.5,
        vf_prob=0,
        rr_prob=0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        cj_prob=0.8,
        cj_bright=0.4,
        cj_contrast=0.4,
        cj_sat=0.2,
        cj_hue=0.1,
        random_gray_scale=0.2,
        gaussian_blur=(1.0, 0.1, 0.5),
        kernel_size: Optional[float] = None,
        kernel_scale: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 2),
        solarization_prob=0.2,
        normalize=imagenet_normalize,
    ):
        flip_and_color_jitter = T.Compose(
            [
                T.RandomHorizontalFlip(p=hf_prob),
                T.RandomVerticalFlip(p=vf_prob),
                random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
                T.RandomApply(
                    [
                        T.ColorJitter(
                            brightness=cj_bright,
                            contrast=cj_contrast,
                            saturation=cj_sat,
                            hue=cj_hue,
                        )
                    ],
                    p=cj_prob,
                ),
                T.RandomGrayscale(p=random_gray_scale),
            ]
        )
        normalize = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )
        global_crop = T.RandomResizedCrop(
            global_crop_size,
            scale=global_crop_scale,
            interpolation=Image.BICUBIC,
        )

        # first global crop
        global_transform_0 = T.Compose(
            [
                global_crop,
                flip_and_color_jitter,
                GaussianBlur(
                    kernel_size=kernel_size,
                    scale=kernel_scale,
                    sigmas=sigmas,
                    prob=gaussian_blur[0],
                ),
                normalize,
            ]
        )

        # second global crop
        global_transform_1 = T.Compose(
            [
                global_crop,
                flip_and_color_jitter,
                GaussianBlur(
                    kernel_size=kernel_size,
                    scale=kernel_scale,
                    sigmas=sigmas,
                    prob=gaussian_blur[1],
                ),
                RandomSolarization(prob=solarization_prob),
                normalize,
            ]
        )

        # transformation for the local small crops
        local_transform = T.Compose(
            [
                T.RandomResizedCrop(
                    local_crop_size, scale=local_crop_scale, interpolation=Image.BICUBIC
                ),
                flip_and_color_jitter,
                GaussianBlur(
                    kernel_size=kernel_size,
                    scale=kernel_scale,
                    sigmas=sigmas,
                    prob=gaussian_blur[2],
                ),
                normalize,
            ]
        )
        local_transforms = [local_transform] * n_local_views

        transforms = [global_transform_0, global_transform_1]
        transforms.extend(local_transforms)
        super().__init__(transforms)


class MAECollateFunction(MultiViewCollateFunction):
    """Implements the view augmentation for MAE [0].

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377

    Attributes:
        input_size:
            Size of the input image in pixels.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        input_size: Union[int, Tuple[int, int]] = 224,
        min_scale: float = 0.2,
        normalize: dict = imagenet_normalize,
    ):
        transforms = [
            T.RandomResizedCrop(
                input_size, scale=(min_scale, 1.0), interpolation=3
            ),  # 3 is bicubic
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]

        if normalize:
            transforms.append(T.Normalize(mean=normalize["mean"], std=normalize["std"]))
        super().__init__([T.Compose(transforms)])

    def forward(self, batch: List[tuple]):
        views, labels, fnames = super().forward(batch)
        # Return only first view as MAE needs only a single view per image.
        return views[0], labels, fnames


class PIRLCollateFunction(nn.Module):
    """Implements the transformations for PIRL [0]. The jigsaw augmentation
    is applied during the forward pass.

    - [0] PIRL, 2019: https://arxiv.org/abs/1912.01991

    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        hf_prob:
            Probability that horizontal flip is applied.
        n_grid:
            Sqrt of the number of grids in the jigsaw image.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    Examples:

        >>> # PIRL for ImageNet
        >>> collate_fn = PIRLCollateFunction()
        >>>
        >>> # PIRL for CIFAR-10
        >>> collate_fn = PIRLCollateFunction(
        >>>     input_size=32,
        >>> )

    """

    def __init__(
        self,
        input_size: int = 64,
        cj_prob: float = 0.8,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.4,
        cj_hue: float = 0.4,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        hf_prob: float = 0.5,
        n_grid: int = 3,
        normalize: dict = imagenet_normalize,
    ):
        _deprecation_warning_collate_functions()
        super(PIRLCollateFunction, self).__init__()

        if isinstance(input_size, tuple):
            input_size_ = max(input_size)
        else:
            input_size_ = input_size

        color_jitter = T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)

        # Transform for transformed jigsaw image
        transform = [
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            T.ToTensor(),
        ]

        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]

        # Cropping and normalisation for untransformed image
        self.no_augment = T.Compose(
            [
                T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
                T.ToTensor(),
                T.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )
        self.jigsaw = Jigsaw(
            n_grid=n_grid,
            img_size=input_size_,
            crop_size=int(input_size_ // n_grid),
            transform=T.Compose(transform),
        )

    def forward(self, batch: List[tuple]):
        """Overriding the BaseCollateFunction class's forward method because
        for PIRL we need only one augmented batch, as opposed to both, which the
        BaseCollateFunction creates."""
        batch_size = len(batch)

        # list of transformed images
        img_transforms = [
            self.jigsaw(batch[i][0]).unsqueeze_(0) for i in range(batch_size)
        ]
        img = [self.no_augment(batch[i][0]).unsqueeze_(0) for i in range(batch_size)]
        # list of labels
        labels = torch.LongTensor([item[1] for item in batch])
        # list of filenames
        fnames = [item[2] for item in batch]

        # tuple of transforms
        transforms = (torch.cat(img, 0), torch.cat(img_transforms, 0))

        return transforms, labels, fnames


class MSNCollateFunction(MultiViewCollateFunction):
    """Implements the transformations for MSN [0].

    Generates a set of random and focal views for each input image. The generated output
    is (views, target, filenames) where views is list with the following entries:
    [random_views_0, random_views_1, ..., focal_views_0, focal_views_1, ...].

    - [0]: Masked Siamese Networks, 2022: https://arxiv.org/abs/2204.07141

    Attributes:
        random_size:
            Size of the random image views in pixels.
        focal_size:
            Size of the focal image views in pixels.
        random_views:
            Number of random views to generate.
        focal_views:
            Number of focal views to generate.
        random_crop_scale:
            Minimum and maximum size of the randomized crops for the relative to random_size.
        focal_crop_scale:
            Minimum and maximum size of the randomized crops relative to focal_size.
        cj_prob:
            Probability that color jittering is applied.
        cj_strength:
            Strength of the color jitter.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        random_gray_scale:
            Probability of conversion to grayscale.
        hf_prob:
            Probability that horizontal flip is applied.
        vf_prob:
            Probability that vertical flip is applied.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.
    """

    def __init__(
        self,
        random_size: int = 224,
        focal_size: int = 96,
        random_views: int = 2,
        focal_views: int = 10,
        random_crop_scale: Tuple[float, float] = (0.3, 1.0),
        focal_crop_scale: Tuple[float, float] = (0.05, 0.3),
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.2, 2),
        random_gray_scale: float = 0.2,
        hf_prob: float = 0.5,
        vf_prob: float = 0.0,
        normalize: dict = imagenet_normalize,
    ) -> None:
        color_jitter = T.ColorJitter(
            brightness=0.8 * cj_strength,
            contrast=0.8 * cj_strength,
            saturation=0.8 * cj_strength,
            hue=0.2 * cj_strength,
        )
        transform = T.Compose(
            [
                T.RandomResizedCrop(size=random_size, scale=random_crop_scale),
                T.RandomHorizontalFlip(p=hf_prob),
                T.RandomVerticalFlip(p=vf_prob),
                T.RandomApply([color_jitter], p=cj_prob),
                T.RandomGrayscale(p=random_gray_scale),
                GaussianBlur(
                    kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur
                ),
                T.ToTensor(),
                T.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )
        focal_transform = T.Compose(
            [
                T.RandomResizedCrop(size=focal_size, scale=focal_crop_scale),
                T.RandomHorizontalFlip(p=hf_prob),
                T.RandomVerticalFlip(p=vf_prob),
                T.RandomApply([color_jitter], p=cj_prob),
                T.RandomGrayscale(p=random_gray_scale),
                GaussianBlur(
                    kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur
                ),
                T.ToTensor(),
                T.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )
        transforms = [transform] * random_views
        transforms += [focal_transform] * focal_views
        super().__init__(transforms=transforms)


class SMoGCollateFunction(MultiViewCollateFunction):
    """Implements the transformations for SMoG.

    Attributes:
        crop_sizes:
            Size of the input image in pixels for each crop category.
        crop_counts:
            Number of crops for each crop category.
        crop_min_scales:
            Min scales for each crop category.
        crop_max_scales:
            Max_scales for each crop category.
        gaussian_blur_probs:
            Probability of Gaussian blur for each crop category.
        gaussian_blur_kernel_sizes:
            Deprecated values in favour of sigmas.
        gaussian_blur_sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
        solarize_probs:
            Probability of solarization for each crop category.
        hf_prob:
            Probability that horizontal flip is applied.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter.
        random_gray_scale:
            Probability of conversion to grayscale.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        crop_sizes: List[int] = [224, 96],
        crop_counts: List[int] = [4, 4],
        crop_min_scales: List[float] = [0.2, 0.05],
        crop_max_scales: List[float] = [1.0, 0.2],
        gaussian_blur_probs: List[float] = [0.5, 0.1],
        gaussian_blur_kernel_sizes: Optional[List[float]] = [None, None],
        gaussian_blur_sigmas: Tuple[float, float] = (0.2, 2),
        solarize_probs: List[float] = [0.0, 0.2],
        hf_prob: float = 0.5,
        cj_prob: float = 1.0,
        cj_strength: float = 0.5,
        random_gray_scale: float = 0.2,
        normalize: dict = imagenet_normalize,
    ):
        transforms = []
        for i in range(len(crop_sizes)):
            random_resized_crop = T.RandomResizedCrop(
                crop_sizes[i], scale=(crop_min_scales[i], crop_max_scales[i])
            )

            color_jitter = T.ColorJitter(
                0.8 * cj_strength,
                0.8 * cj_strength,
                0.4 * cj_strength,
                0.2 * cj_strength,
            )

            transforms.extend(
                [
                    T.Compose(
                        [
                            random_resized_crop,
                            T.RandomHorizontalFlip(p=hf_prob),
                            T.RandomApply([color_jitter], p=cj_prob),
                            T.RandomGrayscale(p=random_gray_scale),
                            GaussianBlur(
                                kernel_size=gaussian_blur_kernel_sizes[i],
                                prob=gaussian_blur_probs[i],
                                sigmas=gaussian_blur_sigmas,
                            ),  # TODO
                            RandomSolarization(prob=solarize_probs[i]),
                            T.ToTensor(),
                            T.Normalize(mean=normalize["mean"], std=normalize["std"]),
                        ]
                    )
                ]
                * crop_counts[i]
            )

        super().__init__(transforms)


class VICRegCollateFunction(BaseCollateFunction):
    """Implementation of a collate function for images.

    This is an implementation of the BaseCollateFunction with a concrete
    set of transforms.

    The set of transforms is inspired by the SimCLR paper as it has shown
    to produce powerful embeddings.

    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_bright:
            How much to jitter brightness.
        cj_contrast:
            How much to jitter constrast.
        cj_sat:
            How much to jitter saturation.
        cj_hue:
            How much to jitter hue.
        min_scale:
            Minimum size of the randomized crop relative to the input_size.
        random_gray_scale:
            Probability of conversion to grayscale.
        solarize_prob:
            Probability of solarization.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Will be deprecated in favor of `sigmas` argument. If set, the old behavior applies and `sigmas` is ignored.
            Used to calculate sigma of gaussian blur with kernel_size * input_size.
        sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `kernel_size` is set.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random rotation is applied.
        rr_degrees:
            Range of degrees to select from for random rotation. If rr_degrees is None,
            images are rotated by 90 degrees. If rr_degrees is a (min, max) tuple,
            images are rotated by a random angle in [min, max]. If rr_degrees is a
            single number, images are rotated by a random angle in
            [-rr_degrees, +rr_degrees]. All rotations are counter-clockwise.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_bright: float = 0.4,
        cj_contrast: float = 0.4,
        cj_sat: float = 0.2,
        cj_hue: float = 0.1,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        solarize_prob: float = 0.1,
        gaussian_blur: float = 0.5,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.2, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: dict = imagenet_normalize,
    ):
        if isinstance(input_size, tuple):
            input_size_ = max(input_size)
        else:
            input_size_ = input_size

        color_jitter = T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            RandomSolarization(prob=solarize_prob),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            T.ToTensor(),
        ]

        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]

        transform = T.Compose(transform)

        super(VICRegCollateFunction, self).__init__(transform)


class VICRegLCollateFunction(nn.Module):
    """Transforms images for VICRegL.

    Attributes:
        global_crop_size:
            Size of the input image in pixels for the global crop category.
        local_crop_size:
            Size of the input image in pixels for the local crop category.
        global_crop_scale:
            Min and max scales for the global crop category.
        local_crop_scale:
            Min and max scales for the local crop category.
        global_grid_size:
            Grid size for the global crop category.
        local_grid_size:
            Grid size for the local crop category.
        global_gaussian_blur_prob:
            Probability of Gaussian blur for the global crop category.
        local_gaussian_blur_prob:
            Probability of Gaussian blur for the local crop category.
        global_gaussian_blur_kernel_size:
            Will be deprecated in favor of `global_gaussian_blur_sigmas` argument. If set, the old behavior applies and `global_gaussian_blur_sigmas` is ignored.
            Used to calculate sigma of gaussian blur with global_gaussian_blur_kernel_size * input_size. Applied to global crop category.
        local_gaussian_blur_kernel_size:
            Will be deprecated in favor of `local_gaussian_blur_sigmas` argument. If set, the old behavior applies and `local_gaussian_blur_sigmas` is ignored.
            Used to calculate sigma of gaussian blur with local_gaussian_blur_kernel_size * input_size. Applied to local crop category.
        global_gaussian_blur_sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `global_gaussian_blur_kernel_size` is set. Applied to global crop category.
        local_gaussian_blur_sigmas:
            Tuple of min and max value from which the std of the gaussian kernel is sampled.
            Is ignored if `local_gaussian_blur_kernel_size` is set. Applied to local crop category.
        global_solarize_prob:
            Probability of solarization for the global crop category.
        local_solarize_prob:
            Probability of solarization for the local crop category.
        hf_prob:
            Probability that horizontal flip is applied.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter.
        random_gray_scale:
            Probability of conversion to grayscale.
        normalize:
            Dictionary with mean and standard deviation for normalization.
    """

    def __init__(
        self,
        global_crop_size: int = 224,
        local_crop_size: int = 96,
        global_crop_scale: Tuple[int] = (0.2, 1.0),
        local_crop_scale: Tuple[int] = (0.05, 0.2),
        global_grid_size: int = 7,
        local_grid_size: int = 3,
        global_gaussian_blur_prob: float = 0.5,
        local_gaussian_blur_prob: float = 0.1,
        global_gaussian_blur_kernel_size: Optional[float] = None,
        local_gaussian_blur_kernel_size: Optional[float] = None,
        global_gaussian_blur_sigmas: Tuple[float, float] = (0.2, 2),
        local_gaussian_blur_sigmas: Tuple[float, float] = (0.2, 2),
        global_solarize_prob: float = 0.0,
        local_solarize_prob: float = 0.2,
        hf_prob: float = 0.5,
        cj_prob: float = 1.0,
        cj_strength: float = 0.5,
        random_gray_scale: float = 0.2,
        normalize: dict = imagenet_normalize,
    ):
        _deprecation_warning_collate_functions()
        super().__init__()
        self.global_crop_and_flip = RandomResizedCropAndFlip(
            crop_size=global_crop_size,
            crop_min_scale=global_crop_scale[0],
            crop_max_scale=global_crop_scale[1],
            hf_prob=hf_prob,
            grid_size=global_grid_size,
        )
        self.local_crop_and_flip = RandomResizedCropAndFlip(
            crop_size=local_crop_size,
            crop_min_scale=local_crop_scale[0],
            crop_max_scale=local_crop_scale[1],
            hf_prob=hf_prob,
            grid_size=local_grid_size,
        )

        color_jitter = T.ColorJitter(
            0.8 * cj_strength,
            0.8 * cj_strength,
            0.4 * cj_strength,
            0.2 * cj_strength,
        )
        self.global_transform = T.Compose(
            [
                T.RandomApply([color_jitter], p=cj_prob),
                T.RandomGrayscale(p=random_gray_scale),
                GaussianBlur(
                    kernel_size=global_gaussian_blur_kernel_size,
                    prob=global_gaussian_blur_prob,
                    sigmas=global_gaussian_blur_sigmas,
                ),
                RandomSolarization(prob=global_solarize_prob),
                T.ToTensor(),
                T.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

        self.local_transform = T.Compose(
            [
                T.RandomApply([color_jitter], p=cj_prob),
                T.RandomGrayscale(p=random_gray_scale),
                GaussianBlur(
                    kernel_size=local_gaussian_blur_kernel_size,
                    prob=local_gaussian_blur_prob,
                    sigmas=local_gaussian_blur_sigmas,
                ),
                RandomSolarization(prob=local_solarize_prob),
                T.ToTensor(),
                T.Normalize(mean=normalize["mean"], std=normalize["std"]),
            ]
        )

    def forward(
        self, batch: List[Tuple[Image.Image, int, str]]
    ) -> Tuple[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Applies transforms to images in the input batch.

        Args:
            batch:
                A list of tuples containing an image (as a PIL Image),
                a label (int), and a filename (str).

        Returns:
            A tuple of transformed images (as a 4-tuple of torch.Tensors containing view_global, view_local, grid_global, grid_local),
            labels (as torch.Tensor), and filenames (as torch.Tensor).

        """

        views_global = []
        views_local = []
        grids_global = []
        grids_local = []
        labels = []
        fnames = []

        for image, label, filename in batch:
            view_global, grid_global = self.global_crop_and_flip.forward(image)
            view_local, grid_local = self.local_crop_and_flip.forward(image)
            views_global.append(self.global_transform(view_global))
            views_local.append(self.local_transform(view_local))
            grids_global.append(grid_global)
            grids_local.append(grid_local)
            labels.append(torch.LongTensor(label))
            fnames.append(filename)

        views_global = torch.stack(views_global)
        views_local = torch.stack(views_local)
        grids_global = torch.stack(grids_global)
        grids_local = torch.stack(grids_local)

        return (views_global, views_local, grids_global, grids_local), labels, fnames


class IJEPAMaskCollator:
    """Collator for IJEPA model [0].

    Experimental: Support for I-JEPA is experimental, there might be breaking changes
    in the future.

    Code inspired by [1].

    - [0]: Joint-Embedding Predictive Architecture, 2023, https://arxiv.org/abs/2301.08243
    - [1]: https://github.com/facebookresearch/ijepa
    """

    def __init__(
        self,
        input_size=(224, 224),
        patch_size=16,
        enc_mask_scale=(0.2, 0.8),
        pred_mask_scale=(0.2, 0.8),
        aspect_ratio=(0.3, 3.0),
        nenc=1,
        npred=2,
        min_keep=4,
        allow_overlap=False,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.patch_size = patch_size
        self.height, self.width = (
            input_size[0] // patch_size,
            input_size[1] // patch_size,
        )
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = (
            allow_overlap  # whether to allow overlap b/w enc and pred masks
        )
        self._itr_counter = Value("i", -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    def _sample_block_mask(self, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """Helper to restrict given mask to a set of acceptable regions"""
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top : top + h, left : left + w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top : top + h, left : left + w] = 0
        # --
        return mask, mask_complement

    def __call__(self, batch):
        """
        Create encoder and predictor masks when collating imgs into a batch
        # 1. sample enc block (size + location) using seed
        # 2. sample pred block (size) using seed
        # 3. sample several enc block locations for each image (w/o seed)
        # 4. sample several pred block locations for each image (w/o seed)
        # 5. return enc mask and pred mask
        """
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio,
        )
        e_size = self._sample_block_size(
            generator=g, scale=self.enc_mask_scale, aspect_ratio_scale=(1.0, 1.0)
        )

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):
            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C

            if self.allow_overlap:
                acceptable_regions = None

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(
                    e_size, acceptable_regions=acceptable_regions
                )
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        collated_masks_pred = [
            [cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred
        ]
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
        # --
        collated_masks_enc = [
            [cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc
        ]
        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)

        return collated_batch, collated_masks_enc, collated_masks_pred


def _deprecation_warning_collate_functions() -> None:
    warn(
        "Collate functions are deprecated and will be removed in favor of transforms in v1.4.0.\n"
        "See https://docs.lightly.ai/self-supervised-learning/examples/models.html for examples.",
        category=DeprecationWarning,
    )
