.. _ref-docker-setup:

Setup
===================================


Analytics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The docker container currently reports usage metrics to our analytics software 
(we use mixpanel) which uses https encrypted GET and POST requests to https://api.mixpanel.com. 
The transmitted data includes information about crashes and the number of samples 
that have been filtered. However, **the data does not include input / output samples**, 
filenames, or any other information which can be sensitive to our customers.



Licensing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The licensing and account management is done through the :ref:`ref-authentication-token` 
as if you would use lightly. The token will be used to authenticate your account. 
The authentication happens at every run of the container. Make sure the docker 
container has a working internet connection and has access to 
https://api.lightly.ai.


.. _ref-docker-download-and-install:

Download the Docker Image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ask your account manager from Lightly for the credentials
to download the docker container. 


In short, installing the Docker container consists of the following steps:

#. Copy the *container-credentials.json* to the instance you want to use for filtering 
#. Authenticate Docker to download the Lightly image
#. Pull the Docker image
#. Check whether the container works

**First**, we need to access the private container registry of Lightly. 
You received a *container-credentials.json* file from your account manager.

**Second**, to be able to download the docker image you need to log in with these credentials. 
The following command will authenticate your installed docker account. 
We assume *container-credentials.json* is in your current directory.

.. code-block:: console

    cat container-credentials.json | docker login -u _json_key --password-stdin https://eu.gcr.io

If the above command does not work, try the following:

.. code-block:: console

    cat container-credentials.json | docker login -u json_key --password-stdin https://eu.gcr.io

**Third**, after authentication you should be able to pull our latest image. 
Using the following command you pull the latest image from our European cloud server:

.. code-block:: console

    docker pull eu.gcr.io/boris-250909/lightly/worker:latest

.. warning::

    Until version 2.1.8 the latest image was named `eu.gcr.io/boris-250909/lightly/sampling:latest` 
    from version 2.2 onwards the image is now called `eu.gcr.io/boris-250909/lightly/worker:latest`.
    Please make sure to update any old docker run commands to use the new image name.


The downloaded image has a long name. We can reduce it by making use of *docker tag*. 
The following experiments are using the following image name 
*lightly/worker:latest*. 
Create a new docker tag using the following command:

.. code-block:: console

    docker tag eu.gcr.io/boris-250909/lightly/worker:latest lightly/worker:latest


.. note:: If you don't want to tag the image name you can replace lightly/worker:latest
          by eu.gcr.io/boris-250909/lightly/worker:latest for all commands in this documentation.

.. _ref-docker-setup-sanity-check:

Sanity Check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Finally**, verify the correctness of the docker container by running the following command:

.. code-block:: console

    docker run --rm -it lightly/worker:latest sanity_check=True

You should see an output similar to this one:

.. code-block:: console
    
    [2022-05-02 20:37:27] Lightly Docker Solution v2.2.0
    [2022-05-02 20:37:27] Congratulations! It looks like the Lightly container is running!

Head on to :ref:`rst-docker-first-steps`  to see how to sample your dataset!


Update Lightly Docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To update the docker we simply need to pull the latest docker image.

.. code-block:: console

    docker pull eu.gcr.io/boris-250909/lightly/worker:latest

Don't forget to tag the image again after pulling it.

.. code-block:: console

    docker tag eu.gcr.io/boris-250909/lightly/worker:latest lightly/worker:latest
