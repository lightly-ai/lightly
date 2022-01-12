
.. _ref-docker-api-worker:

Using the docker as worker for the API for embedding and sampling
=================================================================

Introduction
--------------
The Lightly Docker can be used as a worker for the Lightly Platform to do
all compute-intensive workloads in one single run:
- train an embedding model
- embed all images or video frames
- sample a subset
- compute the metadata of the images

It will also handle the download of filenames from your AWS S3 bucket to your
the machine and upload all artifacts. Thus it allows you to do the full
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
Please follow the instructions `here <https://docs.lightly.ai/docker/getting_started/setup.html>`__.

TODO: Provide the link to the instructions for using docker 3.0 directly in the webapp.

You can test if the installation was successful like this:

.. code-block:: console

    docker run --rm -it lightly/sampling:latest sanity_check=True


Run the Lightly Docker
----------------------
From the Lightly Webapp, copy the command to run the Lightly Docker on your machine.
You can configure the parameters as you like,
e.g. to sample a fixed number of samples or a different ratio.
If you want to use a pretrained embedding model instead of
training one on your dataset, change lightly.trainer.max_epochs to 0.

Then run the command on your machine.

TODO: screenshot of Lightly Webapp showing the command.

View the progress of the Lightly Docker
---------------------------------------

To see the progress of your docker run, go to the Lightly Webapp and
head to "My Docker Runs".

TODO: screenshot of Lightly Webapp showing the progress of the docker.

Use your subsampled dataset
---------------------------

Once the docker run has finished, you can use your subsampled dataset as you like:
E.g. you can analyze it in the embedding and metadata view of the webapp,
subsample it further, or export it for labeling.

Add new samples to your dataset
-------------------------------
You probably get new raw data from time to time and want to add any new samples in
it to your LightlyDataset. This can also be done with the Lightly Docker:

TODO: Define the workflow of doing this.