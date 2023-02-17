from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.imagenet_normalize import imagenet_normalize
from lightly.transforms.rotation import RandomRotationTransform
from lightly.transforms.gaussian_blur import GaussianBlur
from typing import Tuple, Union
import torchvision.transforms as T


class SimCLRTransform(MultiViewTransform):
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
        sigmas: Tuple[float, float] = (0.2, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        normalize: dict = imagenet_normalize,
        to_tensor: bool = True,
    ):

        view_transform = SimCLRViewTransform(
            input_size=input_size,
            cj_prob=cj_prob,
            cj_bright=cj_bright,
            cj_contrast=cj_contrast,
            cj_sat=cj_sat,
            cj_hue=cj_hue,
            min_scale=min_scale,
            random_gray_scale=random_gray_scale,
            gaussian_blur=gaussian_blur,
            sigmas=sigmas,
            vf_prob=vf_prob,
            hf_prob=hf_prob,
            rr_prob=rr_prob,
            rr_degrees=rr_degrees,
            normalize=normalize,
            to_tensor=to_tensor,
        )
        super().__init__(transforms=[view_transform, view_transform])


class SimCLRViewTransform:
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
        sigmas: Tuple[float, float] = (0.2, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        normalize: dict = imagenet_normalize,
        to_tensor: bool = True,
    ):
        color_jitter = T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            RandomRotationTransform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(sigmas=sigmas, prob=gaussian_blur),
        ]
        if to_tensor:
            transform.append(T.ToTensor())
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image):
        return self.transform(image)
