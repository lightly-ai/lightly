.. _dataset-creation-local-upload:

Create a dataset from a local folder
-------------------------------------

To create a new dataset of images with embeddings use the lightly :ref:`lightly-command-line-tool`:

.. code-block:: bash

    lightly-magic trainer.max_epochs=0 token='YOUR_API_TOKEN' new_dataset_name='my-dataset' input_dir='/path/to/my/dataset'


Images and embeddings can also be uploaded from a Python script. For this, you need to
have a numpy array of image embeddings, the filenames of the images, and categorical pseudo-labels.
You can use the `save_embeddings` function to store them in a lightly-compatible CSV format and
upload them from your Python code:

.. code-block:: python

    from lightly.utils import save_embeddings
    from lightly.api.api_workflow_client import ApiWorkflowClient

    client = ApiWorkflowClient(token='123', dataset_id='xyz')

    # upload the images to the dataset
    # change mode to 'thumbnails' or 'meta' if you're working with sensitive data
    client.upload_dataset('path/to/your/images/', mode='full')

    # store the embeddings in a lightly compatible CSV format
    save_embeddings('embeddings.csv', embeddings, labels, filenames)

    # upload the embeddings.csv file to the platform
    client.upload_embeddings('embeddings.csv', name='my-embeddings')

.. note::

    Check out :ref:`ref-webapp-dataset-id` to see how to get the dataset identifier.
