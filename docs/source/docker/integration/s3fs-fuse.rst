
.. _ref-docker-integration-s3fs-fuse:

Load data directly from S3 buckets using s3fs-fuse
===================================================

Learn how to use Lightly Docker directly with an AWS S3 bucket as input.

Very often ML teams don't have all the data locally on a single machine. Instead,
the data is stored in a cloud storage such as AWS S3 or Google Cloud Storage.
Wouldn't it be great to directly connect to these storage buckets instead of
having to download the data everytime to your local machine?

In this example we will do the following:

1. Setup `s3fs-fuse <https://github.com/s3fs-fuse/s3fs-fuse>`_ to mount an S3 bucket to your local machine storage
2. Use the Lightly docker and set the input directory to the mounted S3 bucket

What is s3fs-fuse?
--------------------

`s3fs-fuse <https://github.com/s3fs-fuse/s3fs-fuse>`_ is an open-source project
that allows to mount an S3 bucket to a local file storage. Only a limited set of
file operations are supported (e.g. appending data to a file is very inefficient and slow 
as there is no direct support). However, for just reading files to train a ML model
this approach is more than sufficient.

Here are some of the limitations pointed out in the GitHub readme:

.. note:: Generally S3 cannot offer the same performance or semantics as a local file system. More specifically:
  
  - random writes or appends to files require rewriting the entire object, optimized with multi-part upload copy
  - metadata operations such as listing directories have poor performance due to network latency
  - non-AWS providers may have eventual consistency so reads can temporarily yield stale data (AWS offers read-after-write consistency since Dec 2020)
  - no atomic renames of files or directories
  - no coordination between multiple clients mounting the same bucket
  - no hard links
  - inotify detects only local modifications, not external ones by other clients or tools

Get an AWS Bucket and Credentials
-----------------------------------

From the AWS dashboard go the **S3** service (https://s3.console.aws.amazon.com/s3/home)
and create an S3 bucket if you don't have one yet. 

If you don't have credentials yet you need to go to the **IAM** service 
(https://console.aws.amazon.com/iam/home) on AWS and create
a new user. Make sure you add the **AmazonS3FullAccess** permission. Then create
and download the credentials (.csv file). In the credentials file you should find
the **Access key ID** and **Secret access key** we will use later.

Install s3fs-fuse
-------------------

In order to install `s3fs <https://github.com/s3fs-fuse/s3fs-fuse>`_ we can follow the instructions from the GitHub readme.
On `Debian 9 or newer` or `Ubuntu 16.04 or newer` we can use the following terminal instructions to install it:

.. code-block:: console

    sudo apt install s3fs


Below we show the output for installing s3fs on a Google Cloud Compute instance.

.. code-block:: console
    :caption: Output of the install command

    $ sudo apt install s3fs

    Reading package lists... Done
    Building dependency tree       
    Reading state information... Done
    The following NEW packages will be installed:
      s3fs
    0 upgraded, 1 newly installed, 0 to remove and 81 not upgraded.
    Need to get 214 kB of archives.
    After this operation, 597 kB of additional disk space will be used.
    Get:1 http://deb.debian.org/debian buster/main amd64 s3fs amd64 1.84-1 [214 kB]
    Fetched 214 kB in 0s (8823 kB/s)
    perl: warning: Setting locale failed.
    perl: warning: Please check that your locale settings:
            LANGUAGE = (unset),
            LC_ALL = (unset),
            LC_CTYPE = "UTF-8",
            LANG = "C.UTF-8"
        are supported and installed on your system.
    perl: warning: Falling back to a fallback locale ("C.UTF-8").
    Selecting previously unselected package s3fs.
    (Reading database ... 109361 files and directories currently installed.)
    Preparing to unpack .../archives/s3fs_1.84-1_amd64.deb ...
    Unpacking s3fs (1.84-1) ...
    Setting up s3fs (1.84-1) ...
    Processing triggers for man-db (2.8.5-2) ...


Configure S3 Credentials
--------------------------

Our freshly installed `s3fs-fuse <https://github.com/s3fs-fuse/s3fs-fuse>`_ 
requires access to our S3 bucket. Luckily we can
directly use a AWS credentials file. This file should look like this:

.. code-block:: yaml
    :caption: Example ./aws/credentials

    [default]
    aws_access_key_id=AKIAIOSFODNN7EXAMPLE
    aws_secret_access_key=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY

Let's mount a bucket. We need to create a local folder where we want the S3
content to be mounted on. 

.. code-block:: console
    :caption: Create a local folder **/s3-mount** using the terminal

    mkdir /s3-mount


Now let's use s3fs to mount the bucket to our new folder. Run the following command
in your terminal.

.. code-block:: console
  :caption: Using the default aws credentials from **~/.aws/credentials**

  s3fs simple-test-bucket-igor /s3-mount

.. note:: If you don't specify the location of the `.passwd_file` s3fs uses the default
          location of your aws credentials **~/.aws/credentials**.


If everything went well you should now be able to see the content of your bucket
in your **/s3-mount** folder. If you add a new file to the folder it will 
automatically be uploaded to the bucket.

Optional: use a custom .passwd file for s3fs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you don't want to use the default aws credentials you can also create a separate
passwd file for s3fs:

.. code-block:: console

    echo ACCESS_KEY_ID:SECRET_ACCESS_KEY > ${HOME}/.passwd-s3fs
    chmod 600 ${HOME}/.passwd-s3fs


Now we can mount the S3 bucket using the following command in the terminal.

.. code-block:: console
  :caption: Using the credentials in the **.passwd-s3fs** file

  s3fs simple-test-bucket-igor /s3-mount -o passwd_file=${HOME}/.passwd-s3fs

Use S3 Storage with Lightly Docker
---------------------------------------

Now we can use the docker run command and use the `/s3-mount` directory as the
input dir.

.. code-block:: console

    docker run --gpus all --rm -it \
        -v /s3-mount:/home/input_dir:ro \
        -v /docker/output:/home/output_dir \
        lightly/sampling:latest \
        token=MYAWESOMETOKEN

You can do the same for the docker output directory (in this example I used 
`/docker/output`). We can either use the same bucket and work on subfolders
or use another bucket and repeat the procedure.

Using a mounted S3 bucket for the docker output can be very handy.
Using this approach the pdf report as well as all output files
will directly be uploaded to the S3 storage and can be shared with your team.

Use Caching
--------------

If we use the s3fs setup to train a ML model we would iterate multiple times over
all the images in the bucket. That would not be very efficient as we have lots
of latency overhead as the data is streamed from the bucket. Also the costs could
get high as we create many S3 transactions.

You can specify a folder for the caching by adding `-o use_cache=/tmp` to the command:

.. code-block:: console
  :caption: Using the default aws credentials from **~/.aws/credentials**

  s3fs simple-test-bucket-igor /s3-mount -o use_cache=/tmp


For more information about caching checkout the 
`FAQ wiki of s3fs <https://github.com/s3fs-fuse/s3fs-fuse/wiki/FAQ>`_.


Common Issues
----------------

You need to make sure the AWS S3 region is set accordingly to your bucket location.
In your AWS S3 dashboard you find a list of S3 buckets as well as their region.

.. figure:: images/aws-s3-bucket-list.png
    :align: center
    :alt: Screenshot from aws dashboard showing the S3 buckets
    :figclass: align-center


You can manually specify the AWS region by using the `url=...` flag as shown below:

.. code-block:: console

    s3fs simple-test-bucket-igor /s3-mount -o passwd_file=${HOME}/.passwd-s3fs -o url="https://s3-eu-central-1.amazonaws.com"

