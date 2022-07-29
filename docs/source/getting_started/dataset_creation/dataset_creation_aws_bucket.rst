.. _dataset-creation-aws-bucket:

Create a dataset from an AWS S3 bucket
=======================================


Lightly allows you to configure a remote datasource like Amazon S3 (Amazon Simple Storage Service).
In this guide, we will show you how to setup your S3 bucket, configure your dataset to use said bucket, and only upload metadata to Lightly.

Lightly needs at minimum to be able to read and list permissions on your bucket. It needs them to provide the Lightly Compute Worker access to your dataset,
so that it can process it. Furthermore, the Lightly platform needs access to show you your images in the webapp.

If you want Lightly to create thumbnails for you while uploading the metadata of your images or to write, write permissions are also needed.

There are two ways to set up these Permissions:

1. User Access

This method will create a user with permissions to access your bucket. An Access ID and secret key allow to authenticate as this user.
We recommend this method as it is easy to set up and provides optimal performance.

2. Delegated access

To access your data in your S3 bucket on AWS, Lightly `can assume a role <https://docs.aws.amazon.com/IAM/latest/UserGuide/tutorial_cross-account-with-roles.html>`_ in your account which has the necessary permissions to access your data.
Use this method if internal or external policies of your organization require it or disallow the other method.
It comes with a small overhead for each access to a file in your bucket by Lightly.
The overhead is negligible for larger files (e.g. videos or large images), but may become significant for many small files.

To set up one of the access methods:

.. tabs::

    .. tab:: User Access

        .. include:: dataset_creation_aws_bucket_user_access.rst

    .. tab:: Delegated Access

        .. include:: dataset_creation_aws_bucket_delegated_access.rst


Create and configure a dataset
------------------------------

Create and configure a dataset

1. `Create a new dataset <https://app.lightly.ai/dataset/create>`_ in Lightly.
   Make sure that you choose the input type `Images` or `Videos` correctly,
   depending on the type of files in your S3 bucket.
2. Edit your dataset and select S3 as your datasource

    .. figure:: ../resources/resources_datasource_configure/LightlyEditAWS.jpg
        :align: center
        :alt: Lightly S3 connection config
        :width: 60%

        Lightly S3 connection config


3. As the resource path, enter the full S3 URI to your resource eg. `s3://datalake/projects/farm-animals/`
4. Enter the `access key` and the `secret access key` we obtained from creating a new user in the previous step and select the AWS region in which you created your bucket.

    .. note:: If you are using a delegated access role, toggle the switch `Use IAM role based delegated access` and pass the `external ID` and the `role ARN` from the previous step instead of the secret access key.

5. Toggle the **"Generate thumbnail"** switch if you want Lightly to generate thumbnails for you.
6. If you want to store outputs from Lightly (like thumbnails or extracted frames) in a different directory, you can toggle **"Use a different output datasource"** and enter a different path in your bucket. This allows you to keep your input directory clean as nothing gets ever written there.

    .. note:: Lightly requires list, read, and write access to the `output datasource`. Make sure you have configured it accordingly in the steps before.
7. Press save and ensure that at least the lights for List and Read turn green. If you added permissions for writing, this light should also turn green.

Next steps
----------

Use the Lightly Worker. (see :ref:`ref-docker-setup`).
If you have already set up the Worker, create a dataset with your S3 bucket as datasource. (see :ref:`ref-docker-with-datasource`)
