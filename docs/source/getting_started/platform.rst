.. _lightly-platform:

The Lightly Platform
===================================

The lightly framework itself allows you to use self-supervised learning
in a very simple way and even create embeddings of your dataset.
However, we can do much more than just train and embed datasets. 
Once you have an embedding of an unlabeled dataset you might still require
some labels to train a model. But which samples do you pick for labeling and 
training a model?

This is exactly why we built the 
`Lightly Data Curation Platform <https://app.lightly.ai>`_. 
The platform helps you analyze your dataset and using various methods 
pick the relevant samples for your task.


The video below gives you a quick tour through the platform:


.. raw:: html

    <div style="position: relative; height: 0; 
        overflow: hidden; max-width: 100%; padding-bottom: 20px; height: auto;">
        <iframe width="560" height="315" 
            src="https://www.youtube.com/embed/38kwv0xEIz4" 
            frameborder="0" allow="accelerometer; autoplay; 
            clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
        </iframe>
    </div>

.. |br| raw:: html

    <br />

|br|

.. note:: 

    Head to our :ref:`tutorials <platform-tutorials-label>` to see the many use-cases of the Lightly Platform.


Basic Concepts
-----------------------------------

The Lightly Platform is built around datasets, tags, embeddings, samples and their metadata.

Learn more about the different concepts in our `Glossary <https://app.lightly.ai/glossary>`_.




Create a Dataset
-------------------------


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


.. _platform-custom-metadata:

Custom Metadata
------------------------

With the custom metadata option, you can upload any information about your
images to the Lightly Platform and analyze it there. For example, in autonomous driving, companies
are often interested in different weather scenarios or the number of pedestrians in an image.
The Lightly Platform supports the upload of arbitrary custom metadata as long as it's correctly
formatted.


Upload
^^^^^^^^^^^

You can pass custom metadata when creating a dataset and later configure it for inspection in the web-app.
Simply add the argument `custom_metadata` to the :py:class:`lightly-magic <lightly.cli.lightly_cli>` command.


.. code-block:: bash

    lightly-magic trainer.max_epochs=0 token='YOUR_API_TOKEN' new_dataset_name='my-dataset' input_dir='/path/to/my/dataset' custom_metadata='my-custom-metadata.json'


As with images and embeddings before, it's also possible to upload custom metadata from your Python code:


.. code-block:: python

    import json
    from lightly.api.api_workflow_client import ApiWorkflowClient

    client = ApiWorkflowClient(token='123', dataset_id='xyz')
    with open('my-custom-metadata.json') as f:
        client.upload_custom_metadata(json.load(f))

.. note:: 

    To save the custom metadata in the correct format, use the helpers 
    :py:class:`format_custom_metadata <lightly.utils.io.format_custom_metadata>` and 
    :py:class:`save_custom_metadata <lightly.utils.io.save_custom_metadata>` or learn more
    about the custom metadata format below.

.. note::

    Check out :ref:`ref-webapp-dataset-id` to see how to get the dataset identifier.


Configuration
^^^^^^^^^^^^^^^

In order to use the custom metadata on the Lightly Platform, it must be configured first. For this,
follow these steps:

1. Go to your dataset and click on "Configurator" on the left side.
2. Click on "Add entry" to add a new configuration.
3. Click on "Path". Lightly should now propose different custom metadata keys.
4. Pick the key you are interested in, set the data type, display name, and fallback value.
5. Click on "Save changes" on the bottom.

Done! You can now use the custom metadata in the "Explore" and "Analyze & Filter" screens.

.. figure:: images/custom_metadata_weather_temperature.png
    :align: center
    :alt: Custom metadata weather configuration

    Example of a custom metadata configuration for the key `weather.temperature`.


Format
^^^^^^^^^^^

In order to upload the custom metadata, you need to save it to a `.json` file in a COCO-like format.
The following things are important:

- Information about the images is stored under the key `images`.

- Each image must have a `file_name` and an `id`.

- Custom metadata must be accessed with the `metadata` key.

- Each custom metadata entry must have an `image_id` to match it with the corresponding image.

For the example of an autonomous driving company mentioned above, the custom metadata file would
need to look like this:

.. code-block:: json

    {
        "images": [
            {
                "file_name": "image0.jpg",
                "id": 0,
            },
            {
                "file_name": "image1.jpg",
                "id": 1,
            }
        ],
        "metadata": [
            {
                "image_id": 0,
                "number_of_pedestrians": 3,
                "weather": {
                    "scenario": "cloudy",
                    "temperature": 20.3
                }
            },
            {
                "image_id": 1,
                "number_of_pedestrians": 1,
                "weather": {
                    "scenario": "rainy",
                    "temperature": 15.0
                }
            }
        ]
    }


.. note:: Make sure that the custom metadata is present for every image. The metadata
          must not necessarily include the same keys for all images but it is strongly
          recommended.

.. note:: Lightly supports integers, floats, strings, booleans, and even nested objects for
          custom metadata. Every metadata item must be a valid JSON object.




Sampling
----------------

Before you start sampling make sure you have

#. Created a dataset --> `Create a Dataset`_

#. Uploaded images and embeddings --> `Upload Images`_ & `Upload Embeddings`_

Now, let's get started with sampling!

Follow these steps to sample the most representative images from your dataset:

#. Choose the dataset you want to work on from the *"My Datasets"* 
section by clicking on it.

#. Navigate to *"Analyze & Filter"* → *"Sampling"* through the menu on the left.

#. Choose the embedding and sampling strategy to use for this sampling run.

#. Give a name to your subsampling so that you can later compare 
   the different samplings.

#. Hit "Process" to start sampling the data. Each sample is now assigned an 
   "importance score". The exact meaning of the score depends on the sampler.

    .. figure:: images/webapp_create_sampling.gif
        :align: center
        :alt: Alt text
        :figclass: align-center
        :scale: 150%

        You can create a sampling once you uploaded a dataset and an embedding. 
        Since sampling requires more compute resources it can take a while

#. Move the slider to select the number of images you want to keep and save 
   your selection by creating a new tag, for example like this:

    .. figure:: images/webapp_sampling_new_tag.gif
        :align: center
        :alt: Alt text
        :figclass: align-center
        :scale: 120%

        You can move the slider to change the number of selected samples.


.. _ref-webapp-dataset-id:

Dataset Identifier
-------------------------

Every dataset has a unique identifier called 'Dataset ID'. You find it in the dataset overview page.

.. figure:: images/webapp_dataset_id.jpg
    :align: center
    :alt: Alt text
    :figclass: align-center

    The Dataset ID is a unique identifier.

.. _ref-authentication-token:

Authentication API Token
-----------------------------------

To authenticate yourself on the platform when using the pip package
we provide you with an authentication token. You can retrieve
it when creating a new dataset or when clicking on your 
account (top right)-> preferences on the 
`web application <https://app.lightly.ai>`_.

.. figure:: images/webapp_token.gif
    :align: center
    :alt: Alt text
    :figclass: align-center

    With the API token you can authenticate yourself.

.. warning:: Keep the token for yourself and don't share it. Anyone with the
          token could access your datasets!

