Setup
===================================


Analytics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The docker container currently reports usage metrics to our analytics software 
(we use mixpanel) which uses https encrypted GET and POST requests to https://api.mixpanel.com. 
The transmitted data includes information about crashes and the number of samples 
which have been filtered. However, **the data does not include input / output samples**, 
filenames or any other information which can be sensitive to our customers.



Licensing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The licensing and account management is done through the :ref:`my-reference-label` as if 
you would use lightly. The token will be used to authenticate your account. The authentication
happens at every run of the container. Make sure the docker container has a working internet connection
and has access to https://api.lightly.ai.


Download image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ask your account manager from Lightly for the credentials
to download the docker container. 


In short, installing the Docker container consists of the following steps:

#. Copy the container-credentials.json to the instance you want use for filtering 
#. Authenticate Docker to download the Lightly image
#. Pull the Docker image
#. Check whether the container works

**First**, we need to access the private container registry of Lightly. 
You received a *container-credentials.json* file from your account manager.

**Second**, to be able to download the docker image you need to login with these credentials. 
The following command will authenticate your installed docker account. 
We assume *container-credentials.json* is in your current directory.

.. code-block:: console

    cat container-credentials.json | docker login -u json_key --password-stdin https://eu.gcr.io

**Third**, after authentication you should be able to pull our latest image. 
Using the following command you pull the latest image from our European cloud server:

.. code-block:: console

    docker pull eu.gcr.io/boris-250909/lightly/sampling:latest


The downloaded image has a long name. We can reduce it by making use of *docker tag*. 
The following experiments are using the following image name 
*lightly/sampling:latest*. 
Create a new docker tag using the following command:

.. code-block:: console

    docker tag eu.gcr.io/boris-250909/lightly/sampling:latest lightly/sampling:latest


.. note:: If you don't want to tag the image name you can replace lightly/sampling:latest
          by eu.gcr.io/boris-250909/lightly/sampling:latest for all commands in this documentation.

**Finally**, verify the correctness of the docker container by running the following command:

.. code-block:: console

    docker run --rm -it lightly/sampling:latest sanity_check=True

You should see an output similar to this one:

.. code-block:: console

    [2020-11-12 12:49:38] Lightly Docker Solution
    [2020-11-12 12:49:38] You are using docker build: Thu Nov 12 08:46:04 UTC 2020.
    [2020-11-12 12:49:38] Congratulations! It looks like the Lightly container is running!

Head on to the next page to see how to sample your dataset!
