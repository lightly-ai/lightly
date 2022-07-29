.. _dataset-creation-aws-bucket-delegated-access:

**S3 IAM Delegated Access**

1. Go to the `AWS IAM Console <https://console.aws.amazon.com/iam/home?#/roles>`_

2. Click `Create role`

3. Select `AWS Account` as the trusted entity type

    a. Select `Another AWS account` and specify the AWS Account ID of Lightly: `916419735646`

    b. Check `Require external ID`, and choose an external ID. The external ID should be treated like a passphrase

    c. Do not check `Require MFA`.

    d. Click next

4. Select a policy which grants access to your S3 bucket. If no policy has previously been created, here is an example of how the policy should look like:
Please substitute `YOUR_BUCKET` with the name of your bucket.


    .. code-block:: json

        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "lightlyS3Access",
                    "Action": [
                        "s3:GetObject",
                        "s3:DeleteObject",
                        "s3:PutObject",
                        "s3:ListBucket"
                    ],
                    "Effect": "Allow",
                    "Resource": [
                        "arn:aws:s3:::{YOUR_BUCKET}/*",
                        "arn:aws:s3:::{YOUR_BUCKET}"
                    ]
                }
            ]
        }

5. Name the role `Lightly-S3-Integration` and create the role.
6. Edit your new `Lightly-S3-Integration` role: set the `Maximum session duration` to 12 hours.

    .. warning:: If you don't set the maximum duration to 12 hours, Lightly will not be able to access your data. Please make sure to se the `Maximum session duration` to 12 hours.


7. Remember the external ID and the ARN of the newly created role (`arn:aws:iam::123456789012:role/Lightly-S3-Integration`)



.. note:: We recommend setting up separate input and output datasources (see :ref:`rst-docker-first-steps`). For this either use two different roles with narrow scope or one role with broader access.

