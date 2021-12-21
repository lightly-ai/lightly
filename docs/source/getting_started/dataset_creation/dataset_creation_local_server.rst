.. _dataset-creation-local-server:

Create a dataset with a local file server
-----------------------------------------

You can configure a dataset in the Lightly Platform that streams the images
from your local file system. This makes the creation of a dataset much faster,
as no upload of images and thumbnails is needd.


Setting up local file server
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Navigate to the directory your dataset lies.
From within this directory, create a local http server for the files:

.. code-block:: bash

    http://localhost:1234/

If the port 1234 is used, try another.

Configuring Lightly Dataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create and configure a dataset

1. `Create a new dataset <https://app.lightly.ai/dataset/create>`_ in Lightly
2. Edit your dataset and select
To create a new dataset of images with embeddings use the lightly :ref:`lightly-command-line-tool`:

.. code-block:: bash

    lightly-magic trainer.max_epochs=0 token='YOUR_API_TOKEN' new_dataset_name='my-dataset' input_dir='/path/to/my/dataset'

