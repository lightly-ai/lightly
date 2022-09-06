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
as if you would use lightly. The token will be used to authenticate your account. 
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

Download the Lightly Worker
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The Lightly worker comes as a Docker image. Ask your account manager from Lightly for the credentials
to download the image.


In short, installing the Docker image consists of the following steps:

#. Copy the *container-credentials.json* to the instance you want to use for filtering 
#. Authenticate Docker to download the Lightly image
#. Pull the Docker image
#. Check whether the container works

**First**, we need to access the private container registry of Lightly. 
You received a *container-credentials.json* file from your account manager.

**Second**, to be able to download the Docker image you need to log in with these credentials. 
The following command will authenticate your installed Docker account. 
We assume *container-credentials.json* is in your current directory.

.. code-block:: console

    cat container-credentials.json | docker login -u _json_key --password-stdin https://eu.gcr.io

If the above command does not work, try the following. And please make sure the 
json format is correct (no sudden newlines etc.):

.. code-block:: console

    cat container-credentials.json | docker login -u json_key --password-stdin https://eu.gcr.io


.. note:: When docker is freshly installed only the root user
    can run Docker commands. There are two ways to work in this case. 


#. give your user permission to run - recommended
   docker (see https://docs.docker.com/engine/install/linux-postinstall/) 
#. run Docker commands as root (always replace `docker` with `sudo docker`) - functional but less secure

For example, to authenticate  as non-root user you would run 

.. code-block:: console

    cat container-credentials.json | sudo docker login -u _json_key --password-stdin https://eu.gcr.io


**Third**, after authentication you should be able to pull our latest image. 
Using the following command you pull the latest image from our European cloud server:

.. code-block:: console

    docker pull eu.gcr.io/boris-250909/lightly/worker:latest

In case you experience any issues pulling the docker image after successful
authentication :ref:`check out our FAQ section<rst-docker-known-issues-faq-pulling-docker>`.

.. warning::

    Until version 2.1.8 the latest image was named `eu.gcr.io/boris-250909/lightly/sampling:latest` 
    from version 2.2 onwards the image is now called `eu.gcr.io/boris-250909/lightly/worker:latest`.
    Please make sure to update any old Docker run commands to use the new image name.


The downloaded image has a long name. We can reduce it by making use of *docker tag*. 
The following experiments are using the following image name 
*lightly/worker:latest*. 
Create a new Docker tag using the following command:

.. code-block:: console

    docker tag eu.gcr.io/boris-250909/lightly/worker:latest lightly/worker:latest


.. note:: If you don't want to tag the image name you can replace lightly/worker:latest
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
        -v {OUTPUT_DIR}:/home/output_dir \
        lightly/worker:latest \
        token=MY_AWESOME_TOKEN \
        worker.worker_id=MY_WORKER_ID


.. note:: All registered workers and their ids can be found under https://app.lightly.ai/compute/workers.

All outputs generated by jobs will be stored in `{OUTPUT_DIR}`. The output directory will be explained in more detail in the :ref:`docker-first-steps`.


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
