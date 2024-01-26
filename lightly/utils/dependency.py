import functools


@functools.lru_cache(maxsize=1)
def torchvision_vit_available() -> bool:
    try:
        import torchvision.models.vision_transformer  # Requires torchvision >=0.12
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
        import timm.models.vision_transformer  # Requires timm >= 0.3.3
        from timm.layers import LayerType  # Requires timm >= 0.9.9
    except ImportError:
        return False
    else:
        return True
