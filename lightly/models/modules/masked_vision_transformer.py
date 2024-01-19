from abc import ABC, abstractmethod

import torch


class MaskedVisionTransformer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def patch_embed(self):
        pass

    @abstractmethod
    def add_prefix_tokens(self):
        pass

    @abstractmethod
    def add_pos_embed(self):
        pass
