import functools


@functools.lru_cache(maxsize=1)
def torchvision_vit_available() -> bool:
    """Checks if Vision Transformer (ViT) models are available in torchvision.

    This function checks if the `vision_transformer` module is available in torchvision,
    which requires torchvision version >= 0.12. It also handles exceptions related to
    CUDA version mismatches and installation issues.

    Returns:
        True if the Vision Transformer (ViT) models are available in torchvision,
        otherwise False.
    """
    try:
        import torchvision.models.vision_transformer  # Requires torchvision >=0.12.
    except (
        RuntimeError,  # Different CUDA versions for torch and torchvision.
        OSError,  # Different CUDA versions for torch and torchvision (old).
        ImportError,  # No installation or old version of torchvision.
    ):
        return False
    return True


@functools.lru_cache(maxsize=1)
def timm_vit_available() -> bool:
    """Checks if Vision Transformer (ViT) models are available in the timm library.

    This function checks if the `vision_transformer` module and `LayerType` from timm
    are available, which requires timm version >= 0.3.3 and >= 0.9.9, respectively.

    Returns:
        True if the Vision Transformer (ViT) models are available in timm,
        otherwise False.

    """
    try:
        import timm.models.vision_transformer  # Requires timm >= 0.3.3
        from timm.layers import LayerType  # Requires timm >= 0.9.9
    except ImportError:
        return False
    return True


@functools.lru_cache(maxsize=1)
def torchvision_transforms_v2_available() -> bool:
    """Checks if torchvision supports the v2 transforms API with the `tv_tensors`
    module. Checking for the availability of the `transforms.v2` is not sufficient
    since it is available in torchvision >= 0.15.1, but the `tv_tensors` module is
    only available in torchvision >= 0.16.0.

    Returns:
        True if transforms.v2 are available, False otherwise
    """
    try:
        from torchvision import tv_tensors
    except ImportError:
        return False
    return True
