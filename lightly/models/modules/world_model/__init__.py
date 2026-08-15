"""Building blocks for world models.

A world model predicts how an environment evolves. A latent world model does it
in embedding space: it predicts the embedding of the next frame from the
embeddings of the past frames and the action that was taken, and it never
reconstructs pixels.

The modules here are the parts that world models share:
:class:`ActionEncoder` turns what the agent did into an embedding, and
:class:`LatentDynamicsPredictor` maps embeddings and actions to future
embeddings.

The image encoder is deliberately absent. A predictor reads embeddings, not
pixels, so any backbone in :mod:`lightly.models` works as an encoder, and a
frozen pretrained encoder is interchangeable with one trained from scratch.

Environments, trajectory collection and planning solvers are out of scope. They
need an environment, an episode or a discount factor to be meaningful, which
makes them the job of a control library.

.. warning::

    This subpackage is new. The shapes and signatures here may change in a
    minor release while the remaining world model methods are added.
"""

from lightly.models.modules.world_model.conditioning import ActionEncoder
from lightly.models.modules.world_model.predictor import LatentDynamicsPredictor

__all__ = [
    "ActionEncoder",
    "LatentDynamicsPredictor",
]
