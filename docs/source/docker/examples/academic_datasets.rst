ImageNet
========

Let's have a look at how to run the Lightly Worker to analyze and filter the famous
ImageNet dataset. We are assuming here that the ImageNet dataset is located in an S3
bucket under `s3://dataset/imagenet/`. Start by creating a dataset and configuring the datasource

.. note:: For all examples we assume that the Lightly Worker is configured and running. See :ref:`ref-docker-setup` for more information.


.. code-block:: python

  from lightly.api import ApiWorkflowClient
  from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType
  from lightly.openapi_generated.swagger_client.models.datasource_purpose import DatasourcePurpose

  # Create the Lightly client to connect to the API.
  client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")

  # Create a new dataset on the Lightly Platform.
  client.create_new_dataset_with_unique_name(
      'imagenet-example',
      DatasetType.IMAGES,
  )

  ## AWS S3
  # Input bucket
  client.set_s3_config(
      resource_path="s3://dataset/imagenet/",
      region='eu-central-1',
      access_key='S3-ACCESS-KEY',
      secret_access_key='S3-SECRET-ACCESS-KEY',
      thumbnail_suffix=".lightly/thumbnails/[filename]_thumb.[extension]",
      purpose=DatasourcePurpose.INPUT
  )
  # Output bucket
  client.set_s3_config(
      resource_path="s3://output/",
      region='eu-central-1',
      access_key='S3-ACCESS-KEY',
      secret_access_key='S3-SECRET-ACCESS-KEY',
      thumbnail_suffix=".lightly/thumbnails/[filename]_thumb.[extension]",
      purpose=DatasourcePurpose.LIGHTLY
  )



Next, we schedule a job which extracts 500000 frames with the default Coreset strategy which
selects a diverse set of frames:


.. code-block:: python

    client.schedule_compute_worker_run(
        worker_config={
            "enable_corruptness_check": False,
            "remove_exact_duplicates": True,
            "enable_training": False,
            "pretagging": False,
            "pretagging_debug": False,
            "method": "coreset",
            "stopping_condition": {
                "n_samples": 500000,
                "min_distance": -1
            }
        }
    )


The complete **processing time** was **04h 37m 02s**. The machine used for this experiment is a cloud instance with
8 cores, 30GB of RAM, and a V100 GPU. The dataset was stored on S3.

You can also use the direct link for the
`ImageNet <https://uploads-ssl.webflow.com/5f7ac1d59a6fc13a7ce87963/5facf14359b56365e817a773_report_imagenet_500k.pdf>`_ report.



Combining Cityscapes with Kitti
================================

The Lightly Worker's datapool feature allows to update the pool of selected images
whenver new data arrives. This is a common usecase in production systems where new
image data arrives every week. In this example we simulate this process by first
selecting a subset of the Cityscapes dataset and then adding images from Kitti.


We start by creating a dataset and configuring the datasource. We assume here that we
have **only the Cityscapes** dataset stored in our S3 bucket under `s3://dataset/kittiscapes`:

.. code-block:: python

  from lightly.api import ApiWorkflowClient
  from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType
  from lightly.openapi_generated.swagger_client.models.datasource_purpose import DatasourcePurpose

  # Create the Lightly client to connect to the API.
  client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")

  # Create a new dataset on the Lightly Platform.
  client.create_new_dataset_with_unique_name(
      'kittiscapes-example',
      DatasetType.IMAGES,
  )

  ## AWS S3
  # Input bucket
  client.set_s3_config(
      resource_path="s3://dataset/kittiscapes/",
      region='eu-central-1',
      access_key='S3-ACCESS-KEY',
      secret_access_key='S3-SECRET-ACCESS-KEY',
      thumbnail_suffix=".lightly/thumbnails/[filename]_thumb.[extension]",
      purpose=DatasourcePurpose.INPUT
  )
  # Output bucket
  client.set_s3_config(
      resource_path="s3://output/",
      region='eu-central-1',
      access_key='S3-ACCESS-KEY',
      secret_access_key='S3-SECRET-ACCESS-KEY',
      thumbnail_suffix=".lightly/thumbnails/[filename]_thumb.[extension]",
      purpose=DatasourcePurpose.LIGHTLY
  )

The following command schedules a job to select a subset from Cityscapes:

.. code-block:: python

    client.schedule_compute_worker_run(
        worker_config={
            "enable_corruptness_check": False,
            "remove_exact_duplicates": True,
            "enable_training": False,
            "pretagging": False,
            "pretagging_debug": False,
            "method": "coreset",
            "stopping_condition": {
                "n_samples": -1,
                "min_distance": 0.2,
            }
        }
    )


The report for running the command can be found here:
:download:`Cityscapes.pdf <../resources/datapool_example_cityscapes.pdf>` 

Since the Cityscapes dataset has subfolders for the different cities Lightly
worker uses them as weak labels for the embedding plot as shown below.

.. figure:: ../resources/cityscapes_scatter_umap_k_15_no_overlay.png
    :align: center
    :alt: some alt text

    Scatterplot of Cityscapes. Each color represents one of the 18 
    subfolders (cities) of the Cityscapes dataset.


Now we can use the datapool to select the interesting
frames from Kitti and add them to Cityscapes. For this, first **add all images
from Kitti to the S3 bucket** and then simply run the same command as above again.
The Lightly Worker will detect which images have already been processed and only work with
the new images.


.. code-block:: python

    client.schedule_compute_worker_run(
        worker_config={
            "enable_corruptness_check": False,
            "remove_exact_duplicates": True,
            "enable_training": False,
            "pretagging": False,
            "pretagging_debug": False,
            "method": "coreset",
            "stopping_condition": {
                "n_samples": -1,
                "min_distance": 0.2,
            }
        }
    )


The dataset from the beginning will now contain images from both datasets and 
new plots have been generated in the report. The plots show
the embeddings and highlight with blue color the samples which have been added
from the new dataset. In our experiment, we see that Lightly Worker added several 
new samples outside of the previous embedding distribution. This is great, since it
shows that Cityscapes and Kitti have different data and we can combine the two datasets.


.. figure:: ../resources/datapool_umap_scatter_before_threshold_0.2.png
    :align: center
    :alt: An example of the newly selected examples when we use 
          stopping_condition.min_distance=0.2

    An example of the newly selected examples when we use 
    stopping_condition.min_distance=0.2. 7089 samples from Kitti have been added
    to our existing datapool.

.. figure:: ../resources/datapool_umap_scatter_before_threshold_0.05.png
    :align: center
    :alt: An example of the newly selected examples when we use 
          stopping_condition.min_distance=0.05

    An example of the newly selected examples when we use 
    stopping_condition.min_distance=0.05. 3598 samples from Kitti have been added
    to our existing datapool.


The report for running the command can be found here:
:download:`kitti_with_min_distance=0.2.pdf <../resources/datapool_example_kitti_threshold_0.2.pdf>` 

And the report for stopping condition mininum distance of 0.05:
:download:`kitti_with_min_distance=0.05.pdf <../resources/datapool_example_kitti_threshold_0.05.pdf>` 