import functools


@functools.lru_cache(maxsize=1)
def torchvision_vit_available() -> bool:
    try:
        # Requires torchvision >=0.12
        import torchvision.models.vision_transformer
    except (
        RuntimeError,  # Different CUDA versions for torch and torchvision
        OSError,  # Different CUDA versions for torch and torchvision (old)
        ImportError,  # No installation or old version of torchvision
    ):
        return False
    else:
        return True


@functools.lru_cache(maxsize=1)
def timm_vit_available() -> bool:
    try:
        # Requires timm >= 0.9.9
        import timm.models.vision_transformer
    except ImportError:
        return False
    else:
        return True
