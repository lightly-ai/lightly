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

To use the custom metadata on the Lightly Platform, it must be configured first. For this,
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

To upload the custom metadata, you need to save it to a `.json` file in a COCO-like format.
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

Every dataset has a unique identifier called 'Dataset ID'. You find it on the dataset overview page.

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


How to use S3 with Lightly
------------------------------


Lightly allows you to configure a remote datasource like Amazon S3 (Amazon Simple Storage Service) so that you don't need to upload your data to Lightly and can preserve its privacy.


**What you will learn**


In this guide, we will show you how to setup your S3 bucket, configure your dataset to use said bucket, and only upload metadata to Lightly while preserving the privacy of your data


Setting up Amazon S3
^^^^^^^^^^^^^^^^^^^^^^
For Lightly to be able to create so-called `presigned URLs/read URLs <https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html>`_ to be used for displaying your data in your browser, Lightly needs at minimum to be able to read and list permissions on your bucket. If you want Lightly to create optimal thumbnails for you while uploading the metadata of your images, write permissions are also needed.

Let us assume your bucket is called `datalake`. And let us assume the folder you want to use with Lightly is located at projects/farm-animals/

**Creating an IAM**

1. Go to the `Identity and Access Management IAM page <https://console.aws.amazon.com/iamv2/home?#/users>`_ and create a new user for Lightly.
2. Choose a unique name of your choice and select "Programmatic access" as "Access type". Click next
    
    .. figure:: resources/AWSCreateUser2.png
        :align: center
        :alt: Create AWS User

        Create AWS User

3. We will want to create very restrictive permissions for this new user so that it can't access other resources of your company. Click on "Attach existing policies directly" and then on "Create policy". This will bring you to a new page
    
    .. figure:: resources/AWSCreateUser3.png
        :align: center
        :alt: Setting user permission in AWS

        Setting user permission in AWS

4. As our policy is very simple, we will use the JSON option and enter the following while substituting `datalake` with your bucket and `projects/farm-animals/` with the folder you want to share.
    
    .. code-block:: json

        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "VisualEditor0",
                    "Effect": "Allow",
                    "Action": "s3:ListBucket",
                    "Resource": [
                        "arn:aws:s3:::datalake",
                        "arn:aws:s3:::datalake/projects/farm-animals/*"
                    ]
                },
                {
                    "Sid": "VisualEditor1",
                    "Effect": "Allow",
                    "Action": "s3:*",
                    "Resource": [
                        "arn:aws:s3:::datalake/projects/farm-animals/*"
                    ]
                }
            ]
        }
    .. figure:: resources/AWSCreateUser4.png
        :align: center
        :alt: Permission policy in AWS

        Permission policy in AWS
5. Go to the next page and create tags as you see fit (e.g `external` or `lightly`) and give a name to your new policy before creating it.

    .. figure:: resources/AWSCreateUser5.png
        :align: center
        :alt: Review and name permission policy in AWS

        Review and name permission policy in AWS
6. Return to the previous page as shown in the screenshot below and reload. Now when filtering policies your newly created policy will show up. Select it and continue setting up your new user.
    
    .. figure:: resources/AWSCreateUser6.png
        :align: center
        :alt: Attach permission policy to user in AWS

        Attach permission policy to user in AWS
7. Write down the `Access key ID` and the `Secret access key` in a secure location (such as a password manager) as you will not be able to access this information again (you can generate new keys and revoke old keys under `Security credentials` of a users detail page)
    
    .. figure:: resources/AWSCreateUser7.png
        :align: center
        :alt: Get security credentials (access key id, secret access key) from AWS

        Get security credentials (access key id, secret access key) from AWS

**Preparing your data**


For Lightly to be able to create embeddings and extract metadata from your data, `lightly-magic` needs to be able to access your data. You can either download/sync your data from S3 or you can mount S3 as a drive. We recommend downloading your data from S3 as it makes the overall process faster.

**Downloading from S3 (recommended)**

1. Install AWS cli by following the `guide of Amazon <https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html>`_
2. Run `aws configure` and set the credentials
3. Download/synchronize the folder located on S3 to your current directory `aws s3 sync s3://datalake/projects/farm-animals ./farm`

**Mount S3 as a drive**

Please follow official guidance or for Linux and MacOS use `s3fs-fuse <https://github.com/s3fs-fuse/s3fs-fuse>`_

Uploading your data
^^^^^^^^^^^^^^^^^^^^^^

Create and configure a dataset

1. `Create a new dataset <https://app.lightly.ai/dataset/create>`_ in Lightly
2. Edit your dataset and select S3 as your datasource

    .. figure:: resources/LightlyEdit1.png
        :align: center
        :alt: Get security credentials (access key id, secret access key) from AWS

        Get security credentials (access key id, secret access key) from AWS

3. As the resource path, enter the full S3 URI to your resource eg. `s3://datalake/projects/farm-animals/`
4. Enter the `access key` and the `secret access key` we obtained from creating a new user in the previous step and select the AWS region in which you created your bucket in
5. The thumbnail suffix allows you to configure
   
    - where your thumbnails are stored when you already have generated thumbnails in your S3 bucket
    - where your thumbnails will be stored when you want Lightly to create thumbnails for you. For this to work, the user policy you have created must possess write permissions.
    - when the thumbnail suffix is not defined/empty, we will load the full image even when requesting the thumbnail.
    
    .. figure:: resources/LightlyEdit2.png
        :align: center
        :alt: Lightly S3 connection config
        :width: 60%

        Lightly S3 connection config

6. Press save and ensure that at least the lights for List and Read turn green.


Use Lightly
Use `lightly-magic` and `lightly-upload` just as you always would with the following considerations;

- If you already have generated thumbnails, don't want to see thumbnails or just want to use the full image for a thumbnail (by setting the thumbnail suffix to empty), add `upload=metadata` to the `lightly-magic` command.
- If you want Lightly to create thumbnails for you, you can add `upload=thumbnails` to the `lightly-magic` command.



