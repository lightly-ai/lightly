.. _load-model-from-checkpoint:

Load Model from Checkpoint
==========================

The Lightly worker can be used to :ref:`train a self-supervised model on your data. <training-a-self-supervised-model>`
Lightly saves the weights of the model after training to a checkpoint file in
:code:`output_dir/lightly_epoch_X.ckpt`. This checkpoint can then be further
used to, for example, train a classifier model on your dataset. The code below
demonstrates how the checkpoint can be loaded:

.. literalinclude:: code_examples/load_model_from_checkpoint.py
