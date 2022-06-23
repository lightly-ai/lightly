""" Collate Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn

from typing import Dict, List, Optional, Tuple, Union

from PIL import Image
import torchvision
import torchvision.transforms as T

from lightly.transforms import GaussianBlur
from lightly.transforms import Jigsaw
from lightly.transforms import RandomRotate
from lightly.transforms import RandomSolarization

imagenet_normalize = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
}


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

    def __init__(self, transform: torchvision.transforms.Compose):

        super(BaseCollateFunction, self).__init__()
        self.transform = transform

    def forward(self, batch: List[tuple]):
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
        transforms = [self.transform(batch[i % batch_size][0]).unsqueeze_(0)
                      for i in range(2 * batch_size)]
        # list of labels
        labels = torch.LongTensor([item[1] for item in batch])
        # list of filenames
        fnames = [item[2] for item in batch]

        # tuple of transforms
        transforms = (
            torch.cat(transforms[:batch_size], 0),
            torch.cat(transforms[batch_size:], 0)
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
            Sigma of gaussian blur is kernel_size * input_size.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random (+90 degree) rotation is applied.
        normalize:
            Dictionary with 'mean' and 'std' for torchvision.transforms.Normalize.

    """

    def __init__(self,
                 input_size: int = 64,
                 cj_prob: float = 0.8,
                 cj_bright: float = 0.7,
                 cj_contrast: float = 0.7,
                 cj_sat: float = 0.7,
                 cj_hue: float = 0.2,
                 min_scale: float = 0.15,
                 random_gray_scale: float = 0.2,
                 gaussian_blur: float = 0.5,
                 kernel_size: float = 0.1,
                 vf_prob: float = 0.0,
                 hf_prob: float = 0.5,
                 rr_prob: float = 0.0,
                 normalize: dict = imagenet_normalize):

        if isinstance(input_size, tuple):
            input_size_ = max(input_size)
        else:
            input_size_ = input_size

        color_jitter = T.ColorJitter(
            cj_bright, cj_contrast, cj_sat, cj_hue
        )

        transform = [T.RandomResizedCrop(size=input_size,
                                         scale=(min_scale, 1.0)),
             RandomRotate(prob=rr_prob),
             T.RandomHorizontalFlip(p=hf_prob),
             T.RandomVerticalFlip(p=vf_prob),
             T.RandomApply([color_jitter], p=cj_prob),
             T.RandomGrayscale(p=random_gray_scale),
             GaussianBlur(
                 kernel_size=kernel_size * input_size_,
                 prob=gaussian_blur),
             T.ToTensor()
        ]

        if normalize:
            transform += [
             T.Normalize(
                mean=normalize['mean'],
                std=normalize['std'])
             ]
           
        transform = T.Compose(transform)

        super(ImageCollateFunction, self).__init__(transform)


class MultiViewCollateFunction(nn.Module):
    """Generates multiple views for each image in the batch.

    Attributes:
        transforms:
            List of transformation functions. Each function is used to generate
            one view of the back.
    
    """
    def __init__(self, transforms: List[torchvision.transforms.Compose]):
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
            Sigma of gaussian blur is kernel_size * input_size.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random (+90 degree) rotation is applied.
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

    def __init__(self,
                 input_size: int = 224,
                 cj_prob: float = 0.8,
                 cj_strength: float = 0.5,
                 min_scale: float = 0.08,
                 random_gray_scale: float = 0.2,
                 gaussian_blur: float = 0.5,
                 kernel_size: float = 0.1,
                 vf_prob: float = 0.0,
                 hf_prob: float = 0.5,
                 rr_prob: float = 0.0,
                 normalize: dict = imagenet_normalize):

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
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
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
            Sigma of gaussian blur is kernel_size * input_size.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random (+90 degree) rotation is applied.
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

    def __init__(self,
                 input_size: int = 224,
                 cj_prob: float = 0.8,
                 cj_strength: float = 0.4,
                 min_scale: float = 0.2,
                 random_gray_scale: float = 0.2,
                 gaussian_blur: float = 0.,
                 kernel_size: float = 0.1,
                 vf_prob: float = 0.0,
                 hf_prob: float = 0.5,
                 rr_prob: float = 0.0,
                 normalize: dict = imagenet_normalize):

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
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
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


    def __init__(self,
                 crop_sizes: List[int],
                 crop_counts: List[int],
                 crop_min_scales: List[float],
                 crop_max_scales: List[float],
                 transforms: T.Compose):

        if len(crop_sizes) != len(crop_counts):
            raise ValueError(
                'Length of crop_sizes and crop_counts must be equal but are'
                f' {len(crop_sizes)} and {len(crop_counts)}.'
            )
        if len(crop_sizes) != len(crop_min_scales):
            raise ValueError(
                'Length of crop_sizes and crop_min_scales must be equal but are'
                f' {len(crop_sizes)} and {len(crop_min_scales)}.'
            )
        if len(crop_sizes) != len(crop_min_scales):
            raise ValueError(
                'Length of crop_sizes and crop_max_scales must be equal but are'
                f' {len(crop_sizes)} and {len(crop_min_scales)}.'
            )
        
        crop_transforms = []
        for i in range(len(crop_sizes)):
            
            random_resized_crop = T.RandomResizedCrop(
                crop_sizes[i],
                scale=(crop_min_scales[i], crop_max_scales[i])
            )

            crop_transforms.extend([
                T.Compose([
                    random_resized_crop,
                    transforms,
                ])
            ] * crop_counts[i])
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
            Probability that random (+90 degree) rotation is applied.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter.
        random_gray_scale:
            Probability of conversion to grayscale.
        gaussian_blur:
            Probability of Gaussian blur.
        kernel_size:
            Sigma of gaussian blur is kernel_size * input_size.
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

    def __init__(self,
                 crop_sizes: List[int] = [224, 96],
                 crop_counts: List[int] = [2, 6],
                 crop_min_scales: List[float] = [0.14, 0.05],
                 crop_max_scales: List[float] = [1.0, 0.14],
                 hf_prob: float = 0.5,
                 vf_prob: float = 0.0,
                 rr_prob: float = 0.0,
                 cj_prob: float = 0.8,
                 cj_strength: float = 0.8,
                 random_gray_scale: float = 0.2,
                 gaussian_blur: float = 0.,
                 kernel_size: float = 1.0,
                 normalize: dict = imagenet_normalize):

        color_jitter = T.ColorJitter(
            cj_strength, cj_strength, cj_strength, cj_strength / 4.,
        )

        transforms = T.Compose([
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            RandomRotate(prob=rr_prob),
            T.ColorJitter(),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size, prob=gaussian_blur),
            T.ToTensor(),
            T.Normalize(mean=normalize['mean'], std=normalize['std'])
        ])

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
            Probability that random (+90 degree) rotation is applied.
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
            Sigma of gaussian blur is kernel_size * input_size.
        kernel_scale:
            Fraction of the kernel size which is used for upper and lower
            limits of the randomized kernel size.
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
        cj_prob=0.8,
        cj_bright=0.4,
        cj_contrast=0.4,
        cj_sat=0.2,
        cj_hue=0.1,
        random_gray_scale=0.2,
        gaussian_blur=(1.0, 0.1, 0.5),
        kernel_size=1.4,
        kernel_scale=0.6,
        solarization_prob=0.2,
        normalize=imagenet_normalize,
    ):

        flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            RandomRotate(prob=rr_prob),
            T.RandomApply(
                [T.ColorJitter(
                    brightness=cj_bright, 
                    contrast=cj_contrast, 
                    saturation=cj_sat, 
                    hue=cj_hue
                )],
                p=cj_prob,
            ),
            T.RandomGrayscale(p=random_gray_scale),
        ])
        normalize = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=normalize['mean'], std=normalize['std']),
        ])
        global_crop = T.RandomResizedCrop(
            global_crop_size, 
            scale=global_crop_scale, 
            interpolation=Image.BICUBIC,
        )

        # first global crop
        global_transform_0 = T.Compose([
            global_crop,
            flip_and_color_jitter,
            GaussianBlur(
                kernel_size=kernel_size, 
                prob=gaussian_blur[0], 
                scale=kernel_scale
            ),
            normalize,
        ])
        
        # second global crop
        global_transform_1 = T.Compose([
            global_crop,
            flip_and_color_jitter,
            GaussianBlur(
                kernel_size=kernel_size, 
                prob=gaussian_blur[1], 
                scale=kernel_scale
            ),
            RandomSolarization(prob=solarization_prob),
            normalize,
        ])

        # transformation for the local small crops
        local_transform = T.Compose([
            T.RandomResizedCrop(
                local_crop_size, 
                scale=local_crop_scale, 
                interpolation=Image.BICUBIC
            ),
            flip_and_color_jitter,
            GaussianBlur(
                kernel_size=kernel_size, 
                prob=gaussian_blur[2], 
                scale=kernel_scale
            ),
            normalize,
        ])
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
            T.RandomResizedCrop(input_size, scale=(min_scale, 1.0), interpolation=3),  # 3 is bicubic
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]

        if normalize:
            transforms.append(
                T.Normalize(
                    mean=normalize['mean'],
                    std=normalize['std']
                )
            )
        super().__init__([T.Compose(transforms)])

    def forward(self, batch: List[tuple]):
        views, labels, fnames = super().forward(batch)
        # Return only first view as MAE needs only a single view per image.
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

    def __init__(self,
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
                 normalize: dict = imagenet_normalize
        ):
        super(PIRLCollateFunction, self).__init__()

        if isinstance(input_size, tuple):
            input_size_ = max(input_size)
        else:
            input_size_ = input_size

        color_jitter = T.ColorJitter(
            cj_bright, cj_contrast, cj_sat, cj_hue
        )
        
        # Transform for transformed jigsaw image
        transform = [
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            T.ToTensor()
            ]

        if normalize:
            transform += [
             T.Normalize(
                mean=normalize['mean'],
                std=normalize['std'])
             ]

        # Cropping and normalisation for untransformed image
        self.no_augment = T.Compose([
            T.RandomResizedCrop(size=input_size,
                                scale=(min_scale,1.0)),
            T.ToTensor(),
            T.Normalize(
                mean=normalize['mean'],
                std=normalize['std'])
        ])
        self.jigsaw = Jigsaw(n_grid=n_grid,
                             img_size=input_size_,
                             crop_size=int(input_size_//n_grid),
                             transform=T.Compose(transform))
    
    def forward(self, batch: List[tuple]):
        """Overriding the BaseCollateFunction class's forward method because
        for PIRL we need only one augmented batch, as opposed to both, which the
        BaseCollateFunction creates."""
        batch_size = len(batch)

        # list of transformed images
        img_transforms = [self.jigsaw(batch[i][0]).unsqueeze_(0)
                      for i in range(batch_size)]
        img = [self.no_augment(batch[i][0]).unsqueeze_(0)
                for i in range(batch_size)]
        # list of labels
        labels = torch.LongTensor([item[1] for item in batch])
        # list of filenames
        fnames = [item[2] for item in batch]

        # tuple of transforms
        transforms = (
            torch.cat(img, 0),
            torch.cat(img_transforms, 0)
        )

        return transforms, labels, fnames