"""Building blocks for world models.

A world model predicts how an environment evolves. A latent world model does it
in embedding space: it predicts the embedding of the next frame from the
embeddings of the past frames, optionally conditioned on the action that was
taken.

The modules here are parts that latent world models share.
:class:`ActionEncoder` turns what the agent did into an embedding,
:class:`LatentDynamicsPredictor` maps embeddings and optional actions to future
embeddings, and :class:`PredictorBlock` is its conditioned transformer block,
reusable on its own.

The image encoder is deliberately absent. A predictor reads embeddings, not
pixels, so any backbone in :mod:`lightly.models` works as an encoder, trained
from scratch or frozen and pretrained. Reconstruction decoders and reward, value
or continue heads belong here too when a method trains them jointly with the
dynamics.

Out of scope: environments, trajectory collection and planning solvers. They
need an environment, an episode or a discount factor to be meaningful, which
makes them the job of a control library.

.. warning::

    This subpackage is new. The shapes and signatures here -- including the
    ``conditional`` and ``causal`` options and :class:`PredictorBlock` -- may
    change in a minor release while the remaining world model methods are added.
"""

from lightly.models.modules.world_model.conditioning import ActionEncoder
from lightly.models.modules.world_model.predictor import (
    LatentDynamicsPredictor,
    PredictorBlock,
)

__all__ = [
    "ActionEncoder",
    "LatentDynamicsPredictor",
    "PredictorBlock",
]
