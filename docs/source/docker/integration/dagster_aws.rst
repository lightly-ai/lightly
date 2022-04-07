
.. _ref-docker-integration-aws-dagster:

Data Pre-processing Pipeline on AWS with Dagster
===================================================


Introduction
--------------
Data collection and pre-processing pipelines have become more and more automated in the recent years. The Lightly Docker can take on a crucial role
in such a pipeline as it can reliably filter out redundant images and corrupted images with high throughput.

This guide shows how to write a simple automated data pre-processing pipeline which performs the following steps:

1. Download a random video from `Pexels <https://www.pexels.com/>`_.
2. Upload the video to an S3 bucket.
3. Run the Lightly Docker on the video to extract a diverse set of frames for further processing:
   
   a. Spin up an EC2 instance.
   
   b. Run the Lightly Docker
   
   c. Store the extracted frames in the S3 bucket
   
   d. Stop the EC2 instance

Here, the first two steps simulate a data collection process.

.. note::

    The datapool option of the Lightly Docker allows it to remember frames/images it has seen
    in past executions of the pipeline and ignore images which are too similar to already known ones.


Dagster
---------
Dagster is an open-source data orchestrator for machine learning. It enables building, deploying, and
debugging data processing pipelines. Click `here <https://dagster.io/>`__ to learn more.


Setting up the EC2 Instance
-----------------------------
The first step is to set up the EC2 instance. For the purposes of this tutorial,
it's recommended to pick an instance with a GPU (like the g4dn.xlarge) and the "Deep Learning AMI (Ubuntu 18.04) Version 48.0" AMI.
See `this guide <https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EC2_GetStarted.html>`_ to get started. Connect to the instance.


Next, the Lightly Docker should be installed. Please follow the instructions `here <https://docs.lightly.ai/docker/getting_started/setup.html>`__.
You can test if the installation was successfull like this:

.. code-block:: console

    docker run --rm -it lightly/worker:latest sanity_check=True

To run the docker remotely, it's recommended to write a `run.sh` script with default parameters. The other parameters can then
be changed by passing command line arguments. Use the following as a starting point and adapt it to your needs:

.. code-block:: shell

    # general
    IMAGE=lightly/worker:latest

    INPUT_DIR=$1
    SHARED_DIR=/home/ubuntu/shared_dir
    OUTPUT_DIR=/home/ubuntu/lightly-aws-bucket/output_dir

    # api
    TOKEN=YOUR_LIGHTLY_TOKEN

    # run command
    docker run --gpus all --rm --shm-size="512m" \
            -v ${INPUT_DIR}:/home/input_dir \
            -v ${OUTPUT_DIR}:/home/output_dir \
            -v ${SHARED_DIR}:/home/shared_dir \
            --ipc="host" --network "host" \
            ${IMAGE} token=${TOKEN} \
            lightly.loader.num_workers=0 \
            enable_corruptness_check=True \
            remove_exact_duplicates=True \
            stopping_condition.n_samples=0.1 \
            upload_dataset=True \
            dump_dataset=True \
            datapool.name=lightly-datapool \
            >> /home/ubuntu/log.txt


.. note::

    The above run command samples 10% of the frames for every input. After sampling, it uploads the sampled images to the Lightly Platform
    and saves them to the output directory. The datapool option allows the Lightly Docker to remember already seen frames and adapt decisions based 
    on this knowledge. Learn more about the configuration of the `run.sh` file `here <https://docs.lightly.ai/docker/configuration/configuration.html>`_.


    
Setting up the S3 Bucket
--------------------------
If you don't have an S3 bucket already, follow `these <https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html>`_ instructions to create one.
For the purpose of this tutorial, name the bucket `lightly-aws-bucket`. If you want to use a different S3 bucket, remember to replace all occurences
of `lightly-aws-bucket` in the rest of this guide.


To access the data in the S3 bucket, the S3 bucket must be mounted on the EC2 instance. This can be done with the s3fs library.

First, install the library:

.. code-block:: console

    sudo apt install s3fs


Then, set the `user_allow_other` flag in the `/etc/fuse.conf` file and add the following line to `/etc/fstab`:

.. code-block:: console

    s3fs#lightly-aws-bucket /home/ubuntu/lightly-aws-bucket/ fuse _netdev,allow_other,umask=000,passwd_file=/home/ubuntu/.passwd-s3fs 0 0

Finally, create a password file which contains your AWS credentials and mount the S3 bucket:

.. code-block:: console

    echo "YOUR_AWS_ACCESS_KEY_ID:YOUR_AWS_ACCSESS_KEY" >> ~/.passwd-s3fs
    mkdir ~/lightly-aws-bucket
    sudo mount -a


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
in a randomly created subfolder in the S3 bucket and passes the object name to the next solid.
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

            # upload file to lightly-aws-bucket/input_dir/RANDOM_STRING/basename.mp4
            object_name = os.path.join(
                'input_dir',
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


Finally, the last solid in the pipeline (`lightly.py`) spins up the EC2 instance, runs the Lightly Docker on the object name passed
by the last solid, and then stops the EC2 instance again. Set the `REGION_NAME`, `INSTANCE_ID`, and `MOUNTED_DIR` if 
necessary.


.. code-block:: python

    import os
    import time

    import boto3
    from botocore.exceptions import ClientError

    from dagster import solid


    REGION_NAME: str = 'YOUR_REGION_NAME' # e.g. eu-central-1
    INSTANCE_ID: str = 'YOUR_INSTANCE_ID'
    MOUNTED_DIR: str = '/home/ubuntu/lightly-aws-bucket'


    class EC2Client:
        """EC2 client to start, run, and stop instances.
        
        """

        def __init__(self):
            self.ec2 = boto3.client('ec2', region_name=REGION_NAME)
            self.ssm = boto3.client('ssm', region_name=REGION_NAME)


        def wait(self, client, wait_for: str, **kwargs):
            """Waits for a certain status of the ec2 or ssm client.
            
            """
            waiter = client.get_waiter(wait_for)
            waiter.wait(**kwargs)
            print(f'{wait_for}: OK')


        def start_instance(self, instance_id: str):
            """Starts the EC2 instance with the given id.
            
            """
            # Do a dryrun first to verify permissions
            try:
                self.ec2.start_instances(
                    InstanceIds=[instance_id],
                    DryRun=True
                )
            except ClientError as e:
                if 'DryRunOperation' not in str(e):
                    raise

            # Dry run succeeded, run start_instances without dryrun
            try:
                self.ec2.start_instances(
                    InstanceIds=[instance_id],
                    DryRun=False
                )
            except ClientError as e:
                print(e)

            self.wait(self.ec2, 'instance_exists')
            self.wait(self.ec2, 'instance_running')


        def stop_instance(self, instance_id: str):
            """Stops the EC2 instance with the given id.
            
            """
            # Do a dryrun first to verify permissions
            try:
                self.ec2.stop_instances(
                    InstanceIds=[instance_id],
                    DryRun=True
                )
            except ClientError as e:
                if 'DryRunOperation' not in str(e):
                    raise

            # Dry run succeeded, call stop_instances without dryrun
            try:
                self.ec2.stop_instances(
                    InstanceIds=[instance_id],
                    DryRun=False
                )
            except ClientError as e:
                print(e)

            self.wait(self.ec2, 'instance_stopped')


        def run_command(self, command: str, instance_id: str):
            """Runs the given command on the instance with the given id.
            
            """

            # Make sure the instance is OK
            time.sleep(10)

            response = self.ssm.send_command(
                DocumentName='AWS-RunShellScript',
                Parameters={'commands': [command]},
                InstanceIds=[instance_id]
            )
            command_id = response['Command']['CommandId']

            # Make sure the command is pending
            time.sleep(10)

            try:
                self.wait(
                    self.ssm,
                    'command_executed',
                    CommandId=command_id,
                    InstanceId=INSTANCE_ID,
                    WaiterConfig={
                        'Delay': 5,
                        'MaxAttempts': 1000,
                    }
                )
            except:
                # pretty print error message
                import pprint
                pprint.pprint(
                    self.ssm.get_command_invocation(
                        CommandId=command_id,
                        InstanceId=INSTANCE_ID,
                    )
                )


    @solid
    def run_lightly_onprem(object_name: str) -> None:
        """Dagster solid to run Lightly On-premise on a remote EC2 instance.

        Args:
            object_name:
                S3 object containing the input video(s) for Lightly.

        """

        # object name is of format path/RANDOM_DIR/RANDOM_NAME.mp4
        # so the input directory is the RANDOM_DIR
        input_dir = object_name.split('/')[-2]

        # input dir is mounted_dir/input_dir/batch/
        input_dir = os.path.join(MOUNTED_DIR, 'input_dir', input_dir)

        ec2_client = EC2Client()
        ec2_client.start_instance(INSTANCE_ID)
        ec2_client.run_command(f'/home/ubuntu/run.sh {input_dir}', INSTANCE_ID)
        ec2_client.stop_instance(INSTANCE_ID)


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
        object_name = upload_video_to_s3(file_name)
        run_lightly_onprem(object_name)


Dagster allows to visualize pipelines in a web interface. The following command
shows the above pipeline on `127.0.0.1:3000`:

.. code-block:: console

    dagit -f aws_example_pipeline.py


Finally, you can execute the pipeline with the following command:


.. code-block:: console

    dagster pipeline execute -f aws_example_pipeline.py

For automatic execution of the pipeline you can install a cronjob, trigger the pipeline
upon certain events, or deploy it to an `AWS EC2 or GCP GCE <https://docs.dagster.io/deployment>`_.