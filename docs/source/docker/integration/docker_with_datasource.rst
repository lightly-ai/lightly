
.. _ref-docker-with-datasource:

Using the docker with an S3 bucket as remote datasource.
========================================================

Introduction
--------------
The Lightly Docker can be used with the Lightly Platform to do
the following workloads in one single run:

- stream the files from your AWS S3 bucket to your local machine
- embed all images or video frames
- sample a subset
- compute the metadata of the images
- create a dataset in the Lightly Platform from the sampled subset

It will also handle the download of files from your AWS S3 bucket to your
machine and upload all artifacts. Thus it allows you to do the full
Lightly workflow in one single run with minimal overhead.

Requirements
------------

This tutorial requires that you already have a dataset in the Lightly Platform
configured to use the data in your AWS S3 bucket.

Follow the steps in the `tutorial <https://docs.lightly.ai/getting_started/dataset_creation/dataset_creation_aws_bucket.html>`_
to create such a dataset.

Furthermore, you should have access to a machine running docker.
Ideally, it also has a CUDA-GPU.
A fast GPU will speed up the process significantly,
especially for large datasets.


Download the Lightly Docker
---------------------------------------------
Next, the Lightly Docker should be installed.
Please follow the instructions for the :ref:`ref-docker-setup`.


Run the Lightly Docker with the datasource
------------------------------------------
Head to the :ref:`rst-docker-first-steps` to get a general idea of what the docker
can do.

For running the docker with a remote datasouce, use the parameter `datasource.id=YOUR_DATASET_ID`.
You find the dataset id in the Lightly Platform.
E.g. run the docker with

.. code-block:: console

    docker run --gpus all --rm -it \
        -v OUTPUT_DIR:/home/output_dir \
        lightly/sampling:latest \
        token=YOUR_LIGHTLY_PLATFORM_TOKEN \
        datasource.dataset_id=YOUR_DATASET_ID

View the progress of the Lightly Docker
---------------------------------------

To see the progress of your docker run, go to the Lightly Webapp and
head to "My Docker Runs".

.. image:: ../getting_started/images/docker_runs_overview.png

Use your subsampled dataset
---------------------------

Once the docker run has finished, you can use your subsampled dataset as you like:
E.g. you can analyze it in the embedding and metadata view of the webapp,
subsample it further, or export it for labeling.

Add new samples to your dataset
-------------------------------
You probably get new raw data from time to time and want to add any new samples in
it to your LightlyDataset. The Lightly Platform remembers which raw data in your S3
bucket has already been processed and will ignore it in future docker runs.
This is way you can run the docker with the same command again. It will find
you new raw data in the S3 bucket, download and subsample it and then add it to
your existing dataset.

If you want to start from scratch again and process all data in you S3 bucket instead,
then set `datasource.process_all=True` in your docker run command.