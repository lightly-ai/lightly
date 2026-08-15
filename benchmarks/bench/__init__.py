"""What operates a benchmark run without touching the update.

Loaders, logging and checkpoint plumbing live here. Anything that changes what
the optimiser sees lives in the method's own benchmark file, where a reviewer
reads it beside the equation.
"""

from benchmarks.bench.datamodule import ImageDataModule

__all__ = ["ImageDataModule"]
