"""Pure functions over tensors: one equation each, no parameters, no state.

A block that owns parameters, buffers or optimiser state belongs in
``lightly.nn`` instead.
"""

from lightly.functional.ntxent import ntxent

__all__ = ["ntxent"]
