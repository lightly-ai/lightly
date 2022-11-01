.. _docker-setup:

Setup
=====


Analytics
^^^^^^^^^

The Lightly worker currently reports usage metrics to our analytics software 
(we use mixpanel) which uses https encrypted GET and POST requests to https://api.mixpanel.com. 
The transmitted data includes information about crashes and the number of samples 
that have been filtered. However, **the data does not include input / output samples**, 
filenames, or any other information which can be sensitive to our customers.



Licensing
^^^^^^^^^

The licensing and account management is done through the :ref:`ref-authentication-token`
obtained from the Lightly Platform (https://app.lightly.ai). 

The token will be used to authenticate your account. 
The authentication happens at every run of the worker. Make sure the Lightly worker
has a working internet connection and has access to https://api.lightly.ai.



Download the Python client
^^^^^^^^^^^^^^^^^^^^^^^^^^

We recommend to use the `lightly` Python package to interact with the Lightly API. It offers
helper functions to create and delete datasets, schedule jobs, and access the results:

.. code-block:: console
    
    pip install lightly

See :ref:`rst-installing` for details.


.. _docker-download-and-install:

Docker Installation
^^^^^^^^^^^^^^^^^^^^

Lightly Worker requires docker to run. We highly recommend a docker installation 
that supports using GPUs for hardware acceleration using a Linux operating system.

**Check if docker is installed:**

.. code-block:: console

    sudo docker run --rm --gpus all nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi

You might get an error message like this if docker is installed but without GPU support

.. code-block:: console

    docker could not select device driver with capabilities gpu


If you don't have docker installed or without GPU support we recommend following
our guide about :ref:`rst-docker-known-issues-faq-install-docker`. 

.. note::
   If you use a cloud instance (e.g. on AWS, GCP or Azure) Docker is most likely
   already installed!

Download the Lightly Worker
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In short, installing the Docker image consists of the following steps:

1. Make sure :code:`container-credentials.json` is on your machine where you want to run Lightly Worker. 
    We need to access the private container registry of Lightly. You received 
    a :code:`container-credentials.json` file from your account manager.

2. Authenticate your docker account
    To be able to download docker images of Lightly you need to log in with these credentials. 

    The following command will authenticate yourself to gain access to the Lightly docker images. 
    We assume :code:`container-credentials.json` is in your current directory.

    .. code-block:: console

        cat container-credentials.json | docker login -u _json_key --password-stdin https://eu.gcr.io

    You should see a message stating `Login Succeeded`.

3. Pull the Lightly Worker docker image
    Using the following command you pull the latest image from our European cloud server:

    .. code-block:: console

        docker pull eu.gcr.io/boris-250909/lightly/worker:latest

    In case you experience any issues pulling the docker image after successful
    authentication :ref:`check out our FAQ section<rst-docker-known-issues-faq-pulling-docker>`.

    .. warning::

        Until version 2.1.8 the latest image was named `eu.gcr.io/boris-250909/lightly/sampling:latest` 
        from version 2.2 onwards the image is now called `eu.gcr.io/boris-250909/lightly/worker:latest`.
        Please make sure to update any old Docker run commands to use the new image name.

4. Shorten the name of the docker image using :code:`docker tag`
    The downloaded image has a long name. We can reduce it by making use of *docker tag*. 
    The following experiments are using the following image name 
    *lightly/worker:latest*. 
    Create a new Docker tag using the following command:

    .. code-block:: console

        docker tag eu.gcr.io/boris-250909/lightly/worker:latest lightly/worker:latest


    .. note:: If you do not want to tag the image name you can replace lightly/worker:latest
            by eu.gcr.io/boris-250909/lightly/worker:latest for all commands in this documentation.


Update the Lightly Worker
^^^^^^^^^^^^^^^^^^^^^^^^^

To update the Lightly worker we simply need to pull the latest docker image.

.. code-block:: console

    docker pull eu.gcr.io/boris-250909/lightly/worker:latest

Don't forget to tag the image again after pulling it.

.. code-block:: console

    docker tag eu.gcr.io/boris-250909/lightly/worker:latest lightly/worker:latest


.. note:: You can download a specific version of the Docker image by indicating the version number
          instead of `latest`. We follow semantic versioning standards. 


Furthermore, we always recommend using the latest version of the lightly pip package 
alongside the latest version of the Lightly Worker. You can update the 
pip package using the following command.

.. code-block:: console

    pip install lightly --upgrade

.. _docker-setup-sanity-check:

Sanity Check
^^^^^^^^^^^^

**Next**, verify that the Lightly worker is installed correctly by running the following command:

.. code-block:: console

    docker run --shm-size="1024m" --rm -it lightly/worker:latest sanity_check=True

You should see an output similar to this one:

.. code-block:: console
    
    [2022-05-02 20:37:27] Lightly Docker Solution v2.2.0
    [2022-05-02 20:37:27] Congratulations! It looks like the Lightly container is running!


.. _worker-register:

Register the Lightly Worker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Finally**, start the Lightly worker in waiting mode. In this mode, the worker will long-poll
the Lightly API for new jobs to process. To do so, a worker first needs to be registered.


.. note:: You only have to register each worker once. The registry is required because
    it's possible to have several workers registered at the same time working on different
    jobs in parallel.

.. code-block:: python

    # execute the following code once to get a worker_id
    from lightly.api import ApiWorkflowClient

    client = ApiWorkflowClient(token='MY_AWESOME_TOKEN') # replace this with your token
    worker_id = client.register_compute_worker()
    print(worker_id)

Store the `worker_id` in a secure location and then start the worker with


.. code-block:: console

    docker run --shm-size="1024m" --gpus all --rm -it \
        lightly/worker:latest \
        token=MY_AWESOME_TOKEN \
        worker.worker_id=MY_WORKER_ID


.. note:: All registered workers and their ids can be found under https://app.lightly.ai/compute/workers.

All outputs generated by jobs will be stored in uploaded to the Lightly API as artifacts. Artifacts are explained in more detail in :ref:`docker-first-steps`.


.. code-block:: console

    [2022-06-03 07:57:34] Lightly Docker Solution v2.2.0
    [2022-06-03 07:57:34] You are using docker build: Wed Jun  1 09:51:10 UTC 2022.
    [2022-06-03 07:57:34] Starting worker with id 61f27c8bf2f5d06164071415
    [2022-06-03 07:57:34] Worker started. Waiting for jobs...

.. note:: In case the command fails because docker does not detect your GPU
          you want to make sure `nvidia-docker` is installed.
          You can follow the guide 
          `here <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`_.


Head on to :ref:`docker-first-steps` to see how to schedule a job!
