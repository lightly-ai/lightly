from typing import Dict, List, Optional, Tuple, Union
from PIL import Image,ImageFilter
from torch import Tensor
import torchvision.transforms.functional as F

from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE

import numpy as np
import random
class RedColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2):
        self.brightness = brightness
        self.contrast = contrast
        
    def __call__(self, img):
        # 转换为HSV色域
        img_hsv = img.convert('HSV')
        h, s, v = img_hsv.split()
        
        # 微调亮度和对比度（红色在HSV中对应特定的H值范围）
        if np.random.random() < 0.5:
            v = v.point(lambda x: x + np.random.uniform(-self.brightness*255, self.brightness*255))
        if np.random.random() < 0.5:
            factor = np.random.uniform(1-self.contrast, 1+self.contrast)
            v = v.point(lambda x: x * factor)
            
        # 限制S和V值范围，避免颜色偏离红色色相
        s = s.point(lambda x: min(max(x, 50), 200))  # 限制饱和度
        v = v.point(lambda x: min(max(x, 100), 255))  # 限制亮度
        
        img_hsv = Image.merge('HSV', (h, s, v))
        return img_hsv.convert('RGB')

class EdgeEnhanceTransform:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            if isinstance(img, Image.Image):
                return img.filter(ImageFilter.EDGE_ENHANCE)
            else:
                # 对Tensor做Sobel边缘检测
                kernel = torch.tensor([[-1, -2, -1],
                                      [ 0,  0,  0],
                                      [ 1,  2,  1]], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                edge = torch.nn.functional.conv2d(img, kernel, padding=1)
                return img + edge
        return img

class TextureEnhanceTransform:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            if isinstance(img, Image.Image):
                return img.filter(ImageFilter.DETAIL)
            else:
                # 对Tensor做高通滤波
                kernel = torch.tensor([[-1, -1, -1],
                                      [-1,  8, -1],
                                      [-1, -1, -1]], dtype=img.dtype, device=img.device).unsqueeze(0).unsqueeze(0)
                if img.dim() == 3:
                    img = img.unsqueeze(0)
                texture = torch.nn.functional.conv2d(img, kernel, padding=1)
                return img + texture
        return img
class SimCLRTransform(MultiViewTransform):
    """Implements the transformations for SimCLR [0, 1].

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of Tensor of length 2.

    Applies the following augmentations by default:
        - Random resized crop
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - ImageNet normalization

    Note that SimCLR v1 and v2 use the same data augmentations.

    - [0]: SimCLR v1, 2020, https://arxiv.org/abs/2002.05709
    - [1]: SimCLR v2, 2020, https://arxiv.org/abs/2006.10029

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of [tensor, tensor].

    Attributes:
        input_size:
            Size of the input image in pixels.
        cj_prob:
            Probability that color jitter is applied.
        cj_strength:
            Strength of the color jitter. `cj_bright`, `cj_contrast`, `cj_sat`, and
            `cj_hue` are multiplied by this value. For datasets with small images,
            such as CIFAR, it is recommended to set `cj_strenght` to 0.5.
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
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        min_scale: float = 0.9,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.25,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 0.5),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.2,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = 15,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        view_transform = SimCLRViewTransform(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_strength=cj_strength,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
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
        super().__init__(transforms=[view_transform, view_transform])


class SimCLRViewTransform:
    def __init__(
        self,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 1.0,
        cj_bright: float = 0.8,
        cj_contrast: float = 0.8,
        cj_sat: float = 0.8,
        cj_hue: float = 0.2,
        min_scale: float = 0.9,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.25,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.1, 0.5),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.2,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = 15,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            # T.Resize(size=input_size),
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            # T.RandomVerticalFlip(p=vf_prob),        # 随机垂直翻转
            T.RandomApply([RedColorJitter()], p=cj_prob),
            # T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            EdgeEnhanceTransform(prob=0.3),      # 新增
            TextureEnhanceTransform(prob=0.3),   # 新增
            T.ToTensor(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image.Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed
