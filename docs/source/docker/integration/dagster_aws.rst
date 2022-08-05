
.. _docker-integration-aws-dagster:

Data Pre-processing Pipeline on AWS with Dagster
================================================


Introduction
--------------
Data collection and pre-processing pipelines have become more and more automated in the recent years. The Lightly Worker can take on a crucial role
in such a pipeline as it can reliably filter out redundant images and corrupted images with high throughput.

This guide shows how to write a simple automated data pre-processing pipeline which performs the following steps:

1. Download a random video from `Pexels <https://www.pexels.com/>`_.
2. Upload the video to an S3 bucket.
3. Run the Lightly Worker on the video to extract a diverse set of frames for further processing:

Here, the first two steps simulate a data collection process.

.. note::

    The datapool option of the Lightly Worker allows it to remember frames/images it has seen
    in past executions of the pipeline and ignore images which are too similar to already known ones.


Dagster
---------
Dagster is an open-source data orchestrator for machine learning. It enables building, deploying, and
debugging data processing pipelines. Click `here <https://dagster.io/>`__ to learn more.


Setting up the S3 Bucket
--------------------------
If you don't have an S3 bucket already, follow `these <https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html>`_ instructions to create one.
For the purpose of this tutorial, name the bucket `lightly-aws-bucket`. If you want to use a different S3 bucket, remember to replace all occurences
of `lightly-aws-bucket` in the rest of this guide.

.. note::
    Make sure you have access to credentials to provide Lightly with `LIST` and `READ` access to the input bucket and
    with `LIST`, `READ`, and `WRITE` access to the output bucket. See :ref:`dataset-creation-gcloud-bucket`, 
    :ref:`dataset-creation-aws-bucket`, and :ref:`dataset-creation-azure-storage` for help
    with configuring the different roles.

Then, configure a dataset in the Lightly Platform which will represent the state of your datapool:

.. code-block:: python

    from lightly.api import ApiWorkflowClient
    from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType
    from lightly.openapi_generated.swagger_client.models.datasource_purpose import DatasourcePurpose

    # Create the Lightly client to connect to the API.
    client = ApiWorkflowClient(token="YOUR_LIGHTLY_TOKEN")

    # Create a new dataset on the Lightly Platform.
    client.create_new_dataset_with_unique_name(
        'my-datapool-name',
        DatasetType.IMAGES  # can be DatasetType.VIDEOS when working with videos
    )
    print(f'Dataset id: {client.dataset_id}')

   ## AWS S3
   # Input bucket
   client.set_s3_config(
       resource_path="s3://lightly-aws-bucket/pexels",
       region='eu-central-1'
       access_key='S3-ACCESS-KEY',
       secret_access_key='S3-SECRET-ACCESS-KEY',
       purpose=DatasourcePurpose.INPUT
   )
   # Output bucket
   client.set_s3_config(
       resource_path="s3://lightly-aws-bucket/outputs/",
       region='eu-central-1'
       access_key='S3-ACCESS-KEY',
       secret_access_key='S3-SECRET-ACCESS-KEY',
       purpose=DatasourcePurpose.LIGHTLY
    )

Make sure to note the dataset id somewhere safe as you'll need it throughout this tutorial.



Setting up the EC2 Instance
-----------------------------
The next step is to set up the EC2 instance. For the purposes of this tutorial,
it's recommended to pick an instance with a GPU (like the g4dn.xlarge) and the "Deep Learning AMI (Ubuntu 18.04) Version 48.0" AMI.
See `this guide <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html>`_ to get started. Connect to the instance.


Next, the Lightly Worker should be installed on the instance. Please follow the instructions `here <https://docs.lightly.ai/docker/getting_started/setup.html>`__.
Make sure you have the API token and the worker id from the setup steps. Start the worker in waiting mode with the following arguments:

.. code-block:: shell

    # general
    IMAGE=lightly/worker:latest

    OUTPUT_DIR=/home/ubuntu/output_dir/

    # api
    TOKEN=YOUR_LIGHTLY_TOKEN
    WORKER_ID=MY_WORKER_ID

    # run command
    # this makes the Lightly Worker start up and wait for jobs
    docker run --gpus all --rm -it \
        -v ${OUTPUT_DIR}:/home/output_dir \
        lightly/worker:latest \
        token=${TOKEN} \
        worker.worker_id=${WORKER_ID}


Integration
-------------

Before you start, install the following dependencies:


.. code:: console

    pip install pypexels
    pip install boto3
    pip install dagster


Now that everything is setup, begin with building the data processing pipeline. Dagster's pipelines consist of several `solids` which can
be chained one after each other. Put each solid in a separate file and aim for the following directory structure:

.. code:: console

    ./source
    ├── aws_example_pipeline.py
    └── solids
        ├── aws
        │   ├── lightly.py
        │   └── s3.py
        └── pexels.py


The following code is the content of `pexels.py` and represents first solid in the pipeline.
It downloads a random video from `Pexels <https://www.pexels.com/>`_ and saves it in the current
working directory. Don't forget to set the `PEXELS_API_KEY`.


.. code-block:: python

    import os
    import string
    import random
    import requests

    from typing import List

    from pypexels import PyPexels

    from dagster import solid


    PEXELS_API_KEY = 'YOUR_PEXELS_API_KEY'


    class PexelsClient:
        """Pexels client to download a random popular video.
        
        """

        def __init__(self):
            self.api = PyPexels(api_key=PEXELS_API_KEY)


        def random_filename(self, size_: int = 8):
            """Generates a random filename of uppercase letters and digits.
            
            """
            chars = string.ascii_uppercase + string.digits
            return ''.join(random.choice(chars) for _ in range(size_)) + '.mp4'


        def download_video(self, root: str):
            """Downloads a random popular video from pexels and saves it.
            
            """
            popular_videos = self.api.videos_popular(per_page=40)._body['videos']
            video = random.choice(popular_videos)
            video_file = video['video_files'][0]
            video_link = video_file['link']
            
            video = requests.get(video_link)
            
            path = os.path.join(root, self.random_filename())
            with open(path, 'wb') as outfile:
                outfile.write(video._content)

            return path


    @solid
    def download_random_video_from_pexels() -> str:
        """Dagster solid to download a random pexels video to the current directory.

        Returns:
            The path to the downloaded video.

        """

        client = PexelsClient()
        path = client.download_video('./')

        return path


The next solid in the pipeline (`s3.py`) uploads the video to the S3 bucket. It saves the video
in a randomly created subfolder in the S3 bucket.
Set the `BUCKET_NAME` and `REGION_NAME` to your bucket name and region of the EC2 instance. 


.. code-block:: python

    import os
    import string
    import random

    import boto3
    from botocore.exceptions import ClientError

    from dagster import solid


    BUCKET_NAME: str = 'lightly-aws-bucket'
    REGION_NAME: str = 'YOUR_REGION_NAME' # e.g. eu-central-1


    class S3Client:
        """S3 client to upload files to a bucket.
        
        """

        def __init__(self):
            self.s3 = boto3.client('s3', region_name=REGION_NAME)


        def random_subfolder(self, size_: int = 8):
            """Generates a random subfolder name of uppercase letters and digits.
            
            """
            chars = string.ascii_uppercase + string.digits
            return ''.join(random.choice(chars) for _ in range(size_))


        def upload_file(self, filename: str):
            """Uploads the file at filename to the s3 bucket.

            Generates a random subfolder so the file will be stored at:
            >>> BUCKET_NAME/RANDOM_SUBFOLDER/basefilename.mp4
            
            """

            # upload file to lightly-aws-bucket/pexels/RANDOM_STRING/basename.mp4
            object_name = os.path.join(
                'pexels',
                self.random_subfolder(),
                os.path.basename(filename)
            )

            # Upload the file
            try:
                self.s3.upload_file(filename, BUCKET_NAME, object_name)
            except ClientError as e:
                print(e)
                return None

            return object_name


    @solid
    def upload_video_to_s3(filename: str) -> str:
        """Dagster solid to upload a video to an s3 bucket.

        Args:
            filename:
                Path to the video which should be uploaded.

        Returns:
            The name of the object in the s3 bucket.

        """

        s3_client = S3Client()
        object_name = s3_client.upload_file(filename)

        return object_name


Finally, the last solid in the pipeline (`lightly.py`) runs the Lightly Worker on the newly collected videos.
Set the `YOUR_LIGHTLY_TOKEN`, `YOUR_DATASET_ID` accordingly.

.. code-block:: python

    import os
    import time

    from dagster import solid

    TOKEN: str = 'YOUR_LIGHTLY_TOKEN'
    DATASET_ID: str = 'YOUR_DATASET_ID'



    class LightlyClient:
        """Lightly client to run the Lightly Worker.
        
        """

        def __init__(self, token: str, dataset_id: str):
            self.token = token
            self.dataset_id = dataset_id

        def run_lightly_worker():
            """Runs the Lightly Worker on the EC2 instance.
            
            """

            client = ApiWorkflowClient(
                token=self.token,
                dataset_id=self.dataset_id
            )
            client.schedule_compute_worker_run(
                worker_config={
                    "enable_corruptness_check": True,
                    "remove_exact_duplicates": True,
                    "enable_training": False,
                    "pretagging": False,
                    "pretagging_debug": False,
                    "method": "coreset",
                    "stopping_condition": {
                        "n_samples": 0.1,
                        "min_distance": -1
                    }
                }
            )


    @solid
    def run_lightly_worker() -> None:
        """Dagster solid to run Lightly Worker on a remote EC2 instance.

        """

        lightly_client = LightlyClient(TOKEN, DATASET_ID)
        lightly_client.run_lightly_worker()


To put the solids together in a single pipeline, save the following code in `aws_example_pipeline.py`:


.. code-block:: python

    from dagster import pipeline

    from solids.pexels import download_random_video_from_pexels
    from solids.aws.s3 import upload_video_to_s3
    from solids.aws.lightly import run_lightly_onprem


    @pipeline
    def aws_example_pipeline():
        """Example data processing pipeline with Lightly on AWS.

        The pipeline performs the following three steps:
            - Download a random video from pexels
            - Upload the video to an s3 bucket
            - Run the Lightly pre-selection solution on the video and store the
                extracted frames in the s3 bucket
        
        """
        file_name = download_random_video_from_pexels()
        upload_video_to_s3(file_name)
        run_lightly_onprem()


Dagster allows to visualize pipelines in a web interface. The following command
shows the above pipeline on `127.0.0.1:3000`:

.. code-block:: console

    dagit -f aws_example_pipeline.py


Finally, you can execute the pipeline with the following command:


.. code-block:: console

    dagster pipeline execute -f aws_example_pipeline.py

For automatic execution of the pipeline you can install a cronjob, trigger the pipeline
upon certain events, or deploy it to an `AWS EC2 or GCP GCE <https://docs.dagster.io/deployment>`_.