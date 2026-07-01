from typing import Dict, List, Optional, Tuple, Union

from PIL.Image import Image
from torch import Tensor

from lightly.transforms.amplitude_rescale_transform import AmplitudeRescaleTransform
from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.gaussian_mixture_masks_transform import GaussianMixtureMask
from lightly.transforms.irfft2d_transform import IRFFT2DTransform
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.phase_shift_transform import PhaseShiftTransform
from lightly.transforms.random_frequency_mask_transform import (
    RandomFrequencyMaskTransform,
)
from lightly.transforms.rfft2d_transform import RFFT2DTransform
from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms.solarize import RandomSolarization
from lightly.transforms.torchvision_v2_compatibility import torchvision_transforms as T
from lightly.transforms.utils import IMAGENET_NORMALIZE


class FDAView1Transform:
    """Transforms an image into the first view for FDA [0].

    Used by FDATransform to create the first view of an image.

    Input to this transform:
        PIL Image. (Tensor inputs are supported when torchvision transforms v2 are available.)
    Output of this transform:
        Tensor.

    Applies the following augmentations by default:
        - Random resized crop
        - RFFT 2D transform
        - Amplitude rescale transform
        - Phase shift transform
        - Random frequency mask transform
        - Gaussian mixture mask
        - IRFFT 2D transform
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - ImageNet normalization

    - [0]: Disentangling the Effects of Data Augmentation and Format Transform in
        Self-Supervised Learning of Image Representations, 2023,
        https://arxiv.org/pdf/2312.02205

    """

    def __init__(
        self,
        # Random resized crop
        input_size: int = 224,
        min_scale: float = 0.08,
        # Color jitter
        cj_prob: float = 0.8,
        cj_contrast: float = 0.4,
        cj_bright: float = 0.4,
        cj_sat: float = 0.2,
        cj_hue: float = 0.1,
        cj_strength: float = 1.0,
        # Grayscale
        random_gray_scale: float = 0.2,
        # Gaussian blur
        gaussian_blur: float = 1.0,
        sigmas: Tuple[float, float] = (0.1, 2),
        kernel_size: Optional[float] = 23,
        # Amplitude rescale
        ampl_rescale_range: Tuple[float, float] = (0.8, 1.75),
        ampl_rescale_prob: float = 0.2,
        # Phase shift
        phase_shift_range: Tuple[float, float] = (0.4, 0.7),
        phase_shift_prob: float = 0.2,
        # Random frequency mask
        rand_freq_mask_range: Tuple[float, float] = (0.01, 0.1),
        rand_freq_mask_prob: float = 0.5,
        # Gaussian mixture mask
        gmm_num_gaussians: int = 20,
        gmm_std_range: Tuple[float, float] = (10, 15),
        gmm_prob: float = 0.2,
        # Other
        solarization_prob: float = 0.0,
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        """Initializes FDAView1Transform.

        Args:
            input_size: Size of the input image in pixels.
            min_scale: Minimum size of the randomized crop relative to the input_size.
            cj_prob: Probability that color jitter is applied.
            cj_contrast: How much to jitter contrast.
            cj_bright: How much to jitter brightness.
            cj_sat: How much to jitter saturation.
            cj_hue: How much to jitter hue.
            cj_strength: Strength of the color jitter. `cj_bright`, `cj_contrast`,
                `cj_sat`, and `cj_hue` are multiplied by this value.
            random_gray_scale: Probability of conversion to grayscale.
            gaussian_blur: Probability of Gaussian blur.
            sigmas: Tuple of min and max value from which the std of the gaussian
                kernel is sampled. Is ignored if `kernel_size` is set.
            kernel_size: Will be deprecated in favor of `sigmas` argument. If set,
                the old behavior applies and `sigmas` is ignored. Used to calculate
                sigma of gaussian blur with kernel_size * input_size.
            ampl_rescale_range: Range of the amplitude rescaling factor for
                frequency domain augmentation.
            ampl_rescale_prob: Probability of applying amplitude rescaling.
            phase_shift_range: Range of the phase shift for frequency domain
                augmentation.
            phase_shift_prob: Probability of applying phase shifting.
            rand_freq_mask_range: Range for the random frequency mask.
            rand_freq_mask_prob: Probability of applying random frequency masking.
            gmm_num_gaussians: Number of Gaussians in the Gaussian mixture mask.
            gmm_std_range: Range of the standard deviation for the Gaussian mixture
                mask in pixels.
            gmm_prob: Probability of applying the Gaussian mixture mask.
            solarization_prob: Probability of solarization.
            vf_prob: Probability that vertical flip is applied.
            hf_prob: Probability that horizontal flip is applied.
            rr_prob: Probability that random rotation is applied.
            rr_degrees: Range of degrees to select from for random rotation. If
                rr_degrees is None, images are rotated by 90 degrees. If rr_degrees
                is a (min, max) tuple, images are rotated by a random angle in
                [min, max]. If rr_degrees is a single number, images are rotated by
                a random angle in [-rr_degrees, +rr_degrees]. All rotations are
                counter-clockwise.
            normalize: Dictionary with 'mean' and 'std' for
                torchvision.transforms.Normalize.

        """
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            T.ToTensor(),
            RFFT2DTransform(),
            T.RandomApply(
                [AmplitudeRescaleTransform(range=ampl_rescale_range)],
                p=ampl_rescale_prob,
            ),
            T.RandomApply(
                [PhaseShiftTransform(range=phase_shift_range)], p=phase_shift_prob
            ),
            T.RandomApply(
                [RandomFrequencyMaskTransform(k=rand_freq_mask_range)],
                p=rand_freq_mask_prob,
            ),
            T.RandomApply(
                [
                    GaussianMixtureMask(
                        num_gaussians=gmm_num_gaussians, std_range=gmm_std_range
                    )
                ],
                p=gmm_prob,
            ),
            IRFFT2DTransform(shape=(input_size, input_size)),
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            RandomSolarization(prob=solarization_prob),
            T.ToTensor(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed


class FDAView2Transform:
    """Transforms an image into the second view for FDA [0].

    Used by FDATransform to create the second view of an image.

    Input to this transform:
        PIL Image. (Tensor inputs are supported when torchvision transforms v2 are available.)
    Output of this transform:
        Tensor.

    Applies the following augmentations by default:
        - Random resized crop
        - RFFT 2D transform
        - Amplitude rescale transform
        - Phase shift transform
        - Random frequency mask transform
        - IRFFT 2D transform
        - Random horizontal flip
        - Color jitter
        - Random gray scale
        - Gaussian blur
        - ImageNet normalization

    - [0]: Disentangling the Effects of Data Augmentation and Format Transform in
        Self-Supervised Learning of Image Representations, 2023,
        https://arxiv.org/pdf/2312.02205

    """

    def __init__(
        self,
        # Random resized crop
        input_size: int = 224,
        min_scale: float = 0.08,
        # Color jitter
        cj_prob: float = 0.8,
        cj_contrast: float = 0.4,
        cj_bright: float = 0.4,
        cj_sat: float = 0.2,
        cj_hue: float = 0.1,
        cj_strength: float = 1.0,
        # Grayscale
        random_gray_scale: float = 0.2,
        # Gaussian blur
        gaussian_blur: float = 0.1,
        sigmas: Tuple[float, float] = (0.1, 2),
        kernel_size: Optional[float] = 23,
        # Amplitude rescale
        ampl_rescale_range: Tuple[float, float] = (0.8, 1.75),
        ampl_rescale_prob: float = 0.2,
        # Phase shift
        phase_shift_range: Tuple[float, float] = (0.4, 0.7),
        phase_shift_prob: float = 0.2,
        # Random frequency mask
        rand_freq_mask_range: Tuple[float, float] = (0.01, 0.1),
        rand_freq_mask_prob: float = 0.5,
        # Gaussian mixture mask
        gmm_num_gaussians: int = 20,
        gmm_std_range: Tuple[float, float] = (10, 15),
        gmm_prob: float = 0.0,
        # Other
        solarization_prob: float = 0.0,
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Optional[Union[float, Tuple[float, float]]] = None,
        normalize: Union[None, Dict[str, List[float]]] = IMAGENET_NORMALIZE,
    ):
        """Initializes FDAView2Transform.

        Args:
            input_size: Size of the input image in pixels.
            min_scale: Minimum size of the randomized crop relative to the input_size.
            cj_prob: Probability that color jitter is applied.
            cj_contrast: How much to jitter contrast.
            cj_bright: How much to jitter brightness.
            cj_sat: How much to jitter saturation.
            cj_hue: How much to jitter hue.
            cj_strength: Strength of the color jitter. `cj_bright`, `cj_contrast`,
                `cj_sat`, and `cj_hue` are multiplied by this value.
            random_gray_scale: Probability of conversion to grayscale.
            gaussian_blur: Probability of Gaussian blur.
            sigmas: Tuple of min and max value from which the std of the gaussian
                kernel is sampled. Is ignored if `kernel_size` is set.
            kernel_size: Will be deprecated in favor of `sigmas` argument. If set,
                the old behavior applies and `sigmas` is ignored. Used to calculate
                sigma of gaussian blur with kernel_size * input_size.
            ampl_rescale_range: Range of the amplitude rescaling factor for
                frequency domain augmentation.
            ampl_rescale_prob: Probability of applying amplitude rescaling.
            phase_shift_range: Range of the phase shift for frequency domain
                augmentation.
            phase_shift_prob: Probability of applying phase shifting.
            rand_freq_mask_range: Range for the random frequency mask.
            rand_freq_mask_prob: Probability of applying random frequency masking.
            gmm_num_gaussians: Number of Gaussians in the Gaussian mixture mask.
            gmm_std_range: Range of the standard deviation for the Gaussian mixture
                mask in pixels.
            gmm_prob: Probability of applying the Gaussian mixture mask.
            solarization_prob: Probability of solarization.
            vf_prob: Probability that vertical flip is applied.
            hf_prob: Probability that horizontal flip is applied.
            rr_prob: Probability that random rotation is applied.
            rr_degrees: Range of degrees to select from for random rotation. If
                rr_degrees is None, images are rotated by 90 degrees. If rr_degrees
                is a (min, max) tuple, images are rotated by a random angle in
                [min, max]. If rr_degrees is a single number, images are rotated by
                a random angle in [-rr_degrees, +rr_degrees]. All rotations are
                counter-clockwise.
            normalize: Dictionary with 'mean' and 'std' for
                torchvision.transforms.Normalize.

        """
        color_jitter = T.ColorJitter(
            brightness=cj_strength * cj_bright,
            contrast=cj_strength * cj_contrast,
            saturation=cj_strength * cj_sat,
            hue=cj_strength * cj_hue,
        )

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0)),
            T.ToTensor(),
            RFFT2DTransform(),
            T.RandomApply(
                [AmplitudeRescaleTransform(range=ampl_rescale_range)],
                p=ampl_rescale_prob,
            ),
            T.RandomApply(
                [PhaseShiftTransform(range=phase_shift_range)], p=phase_shift_prob
            ),
            T.RandomApply(
                [RandomFrequencyMaskTransform(k=rand_freq_mask_range)],
                p=rand_freq_mask_prob,
            ),
            T.RandomApply(
                [
                    GaussianMixtureMask(
                        num_gaussians=gmm_num_gaussians, std_range=gmm_std_range
                    )
                ],
                p=gmm_prob,
            ),
            IRFFT2DTransform(shape=(input_size, input_size)),
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
            GaussianBlur(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur),
            RandomSolarization(prob=solarization_prob),
            T.ToTensor(),
        ]
        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]
        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed


class FDATransform(MultiViewTransform):
    """Implements the transformations for FDA[0].

        Input to this transform:
            PIL Image or Tensor.

        Output of this transform:
            List of Tensor of length 2.

        Applies the following augmentations by default:

            - Random resized crop
            - RFFT 2D transform
            - Amplitude rescale transform
            - Phase shift transform
            - Random frequency mask transform
            - Gaussian mixture mask
            - IRFFT 2D transform
            - Color jitter
            - Random grayscale
            - Gaussian blur
            - Random solarization
            - Random horizontal flip

    - [0]: Disentangling the Effects of Data Augmentation and Format Transform in
    Self-Supervised Learning of Image Representations, 2023, https://arxiv.org/pdf/2312.02205

    Input to this transform:
        PIL Image or Tensor.

    Output of this transform:
        List of [tensor, tensor].

    Attributes:
        view_1_transform: The transform for the first view.
        view_2_transform: The transform for the second view.
    """

    def __init__(
        self,
        view_1_transform: Optional[FDAView1Transform] = None,
        view_2_transform: Optional[FDAView2Transform] = None,
    ):
        # We need to initialize the transforms here
        view_1_transform = view_1_transform or FDAView1Transform()
        view_2_transform = view_2_transform or FDAView2Transform()
        super().__init__(transforms=[view_1_transform, view_2_transform])
