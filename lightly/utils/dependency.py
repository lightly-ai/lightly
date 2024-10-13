""" Dependency Utils """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import functools

@functools.lru_cache(maxsize=1)
def torchvision_vit_available() -> bool:
    """Checks if the Vision Transformer model is available in torchvision.

    This function attempts to import the Vision Transformer model from torchvision, which 
    is available from version 0.12 onward. If the model or specific torchvision version is 
    not available, it catches the ImportError and returns False.

    Returns:
        bool: True if torchvision's Vision Transformer is available, False otherwise.
    
    Raises:
        None: Catches all import-related exceptions internally.

    Example:
        >>> is_vit_available = torchvision_vit_available()
        >>> print(is_vit_available)  # Output: True or False
    """
    try:
        import torchvision.models.vision_transformer  # Requires torchvision >= 0.12
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
    """Checks if the Vision Transformer model is available in timm.

    This function attempts to import the Vision Transformer model and the `LayerType` 
    class from `timm`, which are available from version 0.3.3 and 0.9.9 onward, respectively. 
    If unavailable, the function returns False.

    Returns:
        bool: True if timm's Vision Transformer is available, False otherwise.

    Example:
        >>> is_vit_available = timm_vit_available()
        >>> print(is_vit_available)  # Output: True or False
    """
    try:
        import timm.models.vision_transformer  # Requires timm >= 0.3.3
        from timm.layers import LayerType  # Requires timm >= 0.9.9
    except ImportError:
        return False
    else:
        return True
