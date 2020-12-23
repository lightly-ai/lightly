lightly.cli
===================

.. automodule:: lightly.cli

.lightly_cli
---------------
.. automodule:: lightly.cli.lightly_cli
   :members:

.train_cli
---------------
.. automodule:: lightly.cli.train_cli
   :members:

.embed_cli
---------------
.. automodule:: lightly.cli.embed_cli
   :members:

.upload_cli
---------------
.. automodule:: lightly.cli.upload_cli
   :members:

.download_cli
---------------
.. automodule:: lightly.cli.download_cli
   :members:

.config.config.yaml
-------------------

The default settings for all command line tools in the lightly Python package are stored in a YAML config file.
The config file is distributed along with the Python package and can be adapted to fit custom requirements.

The arguments are grouped into namespaces. For example, everything related to the embedding model is grouped under
the namespace "model". See the config file listed below for an overview over the different namespaces.

Overwrites
^^^^^^^^^^
The default settings can (and sometimes must) be overwritten. For example, when using any command-line tool,
it is necessary to specify an input directory where images are stored. The default setting of "input_dir" is
and empty string so it must be overwritten:

.. code-block:: bash

   # train the default model on my data
   lightly-train input_dir='path/to/my/data'

An argument which is grouped under a certain namespace can be accessed by specifying the namespace and the argument,
separated by a dot. For example the argument "name" in the namespace "model" can be accessed like so:

.. code-block:: bash

   # train a ResNet-34 on my data
   lightly-train input_dir='path/to/my/data' model.name='resnet-34'

Additional Arguments
^^^^^^^^^^^^^^^^^^^^^
Some of the grouped arguments are passed directly to the constructor of the respective class. For example, all 
arguments under the namespace "optimizer" are passed directly to the PyTorch constructor of the optimizer. If you
take a look at the default settings below, you can see that the momentum of the optimizer is not specified in the
config file. In order to train a self-supervised model with momentum, an additional argument needs to be passed.
This can be done by adding a + right before the argument:

.. code-block:: bash

   # train a ResNet-34 with momentum on my data
   lightly-train input_dir='path/to/my/data' model.name='resnet-34' +optimizer.momentum=0.9


Default Settings
^^^^^^^^^^^^^^^^

.. literalinclude:: ../../lightly/cli/config/config.yaml
   :language: yaml
