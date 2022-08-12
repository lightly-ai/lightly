.. _dataset-creation-aws-bucket-server-side-encryption:

S3 Server Side Encryption with KMS
==================================

Its possible to enable server side encryption with a KMS key as outlined by `the official documentation of AWS <https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingServerSideEncryption.html>`_.

Create the KMS key
^^^^^^^^^^^^^^^^^^

1. Go to the `Key Management Service KMS page <https://eu-central-1.console.aws.amazon.com/kms/home>`_ and create a new KMS key for the bucket.
2. Choose a unique name of your choice and select **"Symmetric"** and **"Encrypt and decrypt"**. Click next.
3. On the `define key usage permissions` step 4, ensure that the IAM user or role which is configured to be used with the datasource in the Lightly Worker is selected. Click next and create the key.
4. After creation, you can click on the key and then copy the KMS key arn.

.. note:: The IAM user or role which is configured to be used with the datasource in the Lightly Worker will additionally need the following `AWS KMS permissions <https://docs.aws.amazon.com/kms/latest/developerguide/kms-api-permissions-reference.html>`_:
          `kms:Encrypt`, `kms:Decrypt` and `kms:GenerateDataKey`.



Using the KMS key
^^^^^^^^^^^^^^^^^

When setting up a S3 datasource in Lightly (see :ref:`dataset-creation-aws-bucket`), you can set the KMS key arn.
In that case, the `LIGHTLY_S3_SSE_KMS_KEY` environment variable will be set which will add the following
headers `x-amz-server-side-encryption` and `x-amz-server-side-encryption-aws-kms-key-id` to all requests (`PutObject`)
of the artifacts Lightly creates (like crops, frames, thumbnails) as outlined by `the official documentation of AWS <https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingKMSEncryption.html>`_.