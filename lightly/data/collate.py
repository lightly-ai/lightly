""" Collate Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn

from typing import List

import torchvision
import torchvision.transforms as transforms
from lightly.transforms import GaussianBlur
from lightly.transforms import RandomRotate


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
                A tuple of batches of images, labels, and filenames. All are 
                exactly twice as long as the input batch size. For the labels
                and filenames, the second half of the new batch is a copy of 
                the first half. For the images, the two halves represent 
                different transforms of the original images.
        """
        batch_a = torch.cat(
            [self.transform(item[0]).unsqueeze_(0) for item in batch], 0
        )
        batch_b = torch.cat(
            [self.transform(item[0]).unsqueeze_(0) for item in batch], 0
        )
        labels = torch.tensor([item[1] for item in batch]).long()
        fnames = [item[2] for item in batch]
        return (torch.cat((batch_a, batch_b), 0),
                torch.cat((labels, labels), 0),
                fnames)


class ImageCollateFunction(BaseCollateFunction):
    """Implementation of a collate function for SimCLR.

    This is an implementation of the BaseCollateFunction with a concrete
    set of transforms.

    The set of transforms is determined by the SimCLR paper as it has shown
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
        kernel_size:
            Sigma of gaussian blur is kernel_size * input_size.
        vf_prob:
            Probability that vertical flip is applied.
        hf_prob:
            Probability that horizontal flip is applied.
        rr_prob:
            Probability that random rotation is applied.

    """

    def __init__(self,
                 input_size: int = 32,
                 cj_prob: float = 0.8,
                 cj_bright: float = 0.7,
                 cj_contrast: float = 0.7,
                 cj_sat: float = 0.7,
                 cj_hue: float = 0.2,
                 min_scale: float = 0.08,
                 random_gray_scale: float = 0.2,
                 kernel_size: float = 0.1,
                 vf_prob: float = 0.0,
                 hf_prob: float = 0.0,
                 rr_prob: float = 0.0):

        color_jitter = transforms.ColorJitter(
            cj_bright, cj_contrast, cj_sat, cj_hue
        )

        transform = transforms.Compose(
            [transforms.RandomResizedCrop(size=input_size,
                                          scale=(min_scale, 1.0)),
             RandomRotate(prob=rr_prob),
             transforms.RandomHorizontalFlip(p=hf_prob),
             transforms.RandomVerticalFlip(p=vf_prob),
             transforms.RandomApply([color_jitter], p=cj_prob),
             transforms.RandomGrayscale(p=random_gray_scale),
             GaussianBlur(kernel_size=kernel_size * input_size),
             transforms.ToTensor(),
             transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
             ]
        )
        super(ImageCollateFunction, self).__init__(transform)
