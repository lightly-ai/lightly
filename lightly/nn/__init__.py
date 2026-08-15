"""Modules that own parameters, buffers or optimiser state, plus the helpers
that drive them.

A pure equation over tensors belongs in ``lightly.functional`` instead.
"""

from lightly.nn.axes import encode

__all__ = ["encode"]
