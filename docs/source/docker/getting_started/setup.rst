Setup
===================================

Ask you account manager from Lightly for the credentials
to download the docker container. 


Data Privacy and Analytics
-----------------------------------

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
happens at every run of the container. Make sure the docker container has a working internet conneciton
and has access to https://api.lightly.ai.


Download image
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
In short, installing the Docker container consists of the following steps:

#. Copy the container-credentials.json to the instance you want use for filtering 
#. Authenticate Docker to download the WhatToLabel image
#. Pull the Docker image
#. Check whether the container works

**First**, we need to access the private container registry of WhatToLabel. 
You received a *container-credentials.json* file from your account manager.

**Second**, to be able to download the docker image you need to login with these credentials. 
The following command will authenticate your installed docker account. 
We assume *container-credentials.json* is in your current directory.

.. code-block:: console

    cat container-credentials.json | docker login -u _json_key --password-stdin https://eu.gcr.io

**Third**, after authentication you should be able to pull our latest image. 
Using the following command you pull the latest image from our European cloud server:

.. code-block:: console

    docker pull eu.gcr.io/boris-250909/whattolabel/data-filtering:latest


The downloaded image has a long name. We can reduce it by making use of *docker tag*. 
The following experiments are using the following image name 
*whattolabel/data-filtering:latest*. 
Create a new docker tag using the following command:

.. code-block:: console

    docker tag eu.gcr.io/boris-250909/whattolabel/data-filtering:latest \
        whattolabel/data-filtering:latest


.. note:: If you don't want to tag the image name you can replace whattolabel/data-filtering:latest 
          by eu.gcr.io/boris-250909/whattolabel/data-filtering:latest for the next commands.

**Finally**, verify the correctness of the docker container by running the following command:

.. code-block:: console

    docker run --rm -it whattolabel/data-filtering:latest --sanity_check

You should see an output similar to this one:



Docker Configuration
-----------------------------------

Speed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

--ipc="host" --network="host"
num_workers
Compute efficiency
TODO
