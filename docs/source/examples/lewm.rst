.. _lewm:

LeWM
====

LeWM [0]_ is a latent world model. It predicts the embedding of the next frame
from the embeddings of the past frames and the actions that were taken, and it
never reconstructs pixels. The encoder is trained from pixels together with the
predictor. SIGReg [1]_ keeps the embeddings close to an isotropic Gaussian,
which is what prevents collapse, so LeWM needs no stop-gradient, no teacher
network and no exponential moving average.

This is the self-supervised idea with one more axis. An SSL method predicts the
embedding of one view from another view. A world model predicts the embedding
of the next frame from the past ones, given what the agent did.

.. warning::

    LeWM and the ``world_model`` modules are experimental. Their shapes and
    signatures may change in a minor release.

Key Components
--------------

- **Encoder**: Any image encoder maps a frame to an embedding. The example uses
  a ViT-tiny with a :class:`lightly.models.modules.LeWMProjectionHead` on the
  class token.
- **Action encoder**: :class:`lightly.models.modules.ActionEncoder` maps a
  low-dimensional action vector to the width of the predictor.
- **Predictor**: :class:`lightly.models.modules.LatentDynamicsPredictor` is a
  causal transformer over frames. The action at frame ``t`` conditions every
  block through AdaLN-Zero. It is action-conditioned and causal by default;
  ``conditional=False`` gives an actionless predictor and ``causal=False`` a
  bidirectional one, both built from
  :class:`lightly.models.modules.PredictorBlock`.
- **Loss**: :class:`lightly.loss.LeWMLoss` adds a next-embedding prediction
  term and :class:`lightly.loss.SIGReg`. The weight of the SIGReg term is the
  only hyperparameter that needs tuning.

The shape contract
------------------

The predictor reads and returns::

    embeddings   (B, T, D)
    action_emb   (B, T, D)

``B`` is the batch, ``T`` the frames of one clip and ``D`` the width. Entry
``t`` of the output predicts frame ``t + 1``, and ``action_emb[:, t]`` is the
action taken **at** frame ``t``. A shape check cannot catch a phase error, so a
training step reads::

    predicted = predictor(emb[:, :-1], action_emb=action_emb[:, :-1])
    target = emb[:, 1:]

Good to Know
------------

- **Same regularizer as LeJEPA**: LeWM is LeJEPA over time, with actions. Both
  methods use the same :class:`lightly.loss.SIGReg` class.
- **The encoder is interchangeable**: The predictor reads embeddings, not
  images, so any backbone works, and a frozen pretrained encoder is
  interchangeable with one trained from scratch. A frozen encoder also cannot
  collapse, because its targets cannot move.
- **Planning is out of scope**:
  :meth:`lightly.models.modules.LatentDynamicsPredictor.rollout` feeds the
  model its own predictions, which is how a planner scores candidate action
  sequences. The planner itself needs an environment and an episode, so it
  belongs to a control library rather than here.

.. note::

    A projection head that ends in ``BatchNorm`` behaves differently in train
    and eval mode, and planning runs in eval mode. Check that the embeddings
    agree between the two modes before trusting a rollout.

Reference:

    .. [0] `LeWorldModel, 2026 <https://arxiv.org/abs/2603.19312>`_
    .. [1] `LeJEPA, 2025 <https://arxiv.org/abs/2511.08544>`_

.. note::

    LeWM requires `TIMM <https://github.com/huggingface/pytorch-image-models>`_
    to be installed

    .. code-block:: bash

        pip install "lightly[timm]"

.. tabs::
    .. tab:: PyTorch

        .. image:: https://img.shields.io/badge/Open%20in%20Colab-blue?logo=googlecolab&label=%20&labelColor=5c5c5c
            :target: https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/lewm.ipynb

        The example generates its own trajectories, so it runs without an
        environment or a recorded dataset. It can be run from the command line
        with::

            python lightly/examples/pytorch/lewm.py

        .. literalinclude:: ../../../examples/pytorch/lewm.py
