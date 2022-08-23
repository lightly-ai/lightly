.. _dataset-creation-aws-bucket-minimum-policy:

Minimum AWS Policy requirements
===============================

It is possibly to make your access policy very restrictive and to even deny anyone with the correct IAM user credentials or role from outside of e.g your VPC or of a certain IP range from reading your data.

The only hard requirement Lightly requires to properly work is `S3:ListBucket`.
With this permission Lightly will only be able to list the filenames within your bucket but can't actually access the contents of your data. Only you will be able to access your data's content.

.. warning::
    The Lightly Worker will need to be running within the permissioned zone (e.g within your VPC or IP range) and will need the configuration flag `datasource.bypass_verify` set to `True` in the worker configuration.

    **Important:** When restricting `S3:GetObject`, it will no longer be possible to use the relevant filenames feature.

.. note:: If you later want to use the Lightly Platform to visualize your data (e.g see the images in the embedding view) you will also need to whitelist the IPs from where you are planning to access it from (e.g the IP of your ISP at your office or the IP of your VPN).



Restrict IP-Range
^^^^^^^^^^^^^^^^^

The following example restricts access to your bucket `datalake` so that only services from the IP range `21.21.21.x` are allowed to access your data (see `"Sid": "RestrictIP"`).
For Lightly to properly work we allow `s3:ListBucket` (see `"Sid": "AllowLightly"`).


.. code-block:: json

    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "RestrictIP",
                "Action": "s3:*",
                "Effect": "Allow",
                "Resource": [
                    "arn:aws:s3:::datalake",
                    "arn:aws:s3:::datalake/projects/farm-animals/*"
                ],
                "Condition": {
                    "IpAddress": {
                        "aws:SourceIp": [
                            "21.21.21.0/24"
                        ]
                    }
                }
            },
            {
                "Sid": "AllowLightly",
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": [
                    "arn:aws:s3:::datalake",
                    "arn:aws:s3:::datalake/projects/farm-animals/*"
                ]
            }
        ]
    }


Restrict VPC
^^^^^^^^^^^^
It is possible to `restrict access to a specific VPC <https://docs.aws.amazon.com/AmazonS3/latest/userguide/example-bucket-policies-vpc-endpoint.html#example-bucket-policies-restrict-access-vpc>`_ by specifying a string condition.


.. code-block:: json
    
    {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "RestrictVPC",
                "Action": "s3:*",
                "Effect": "Allow",
                "Resource": [
                    "arn:aws:s3:::datalake",
                    "arn:aws:s3:::datalake/projects/farm-animals/*"
                ],
                "Condition": {
                    "StringEquals": {
                        "aws:SourceVpc": "vpc-111bbb22"
                    }
                }
            },
            {
                "Sid": "AllowLightly",
                "Effect": "Allow",
                "Action": "s3:ListBucket",
                "Resource": [
                    "arn:aws:s3:::datalake",
                    "arn:aws:s3:::datalake/projects/farm-animals/*"
                ]
            }
        ]
    }



Further Restrictions
^^^^^^^^^^^^^^^^^^^^

There are different ways of expressing the logic of restricting access to your resources.
You can `DENY <https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_effect.html>`_ access to certain permissions or inverting the permission with `NotAction <https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_notaction.html>`_.
There are also further `conditional operators <https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_elements_condition_operators.html#Conditions_IPAddress>`_ and `string conditions <https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_condition-keys.html>`_ to be more explicit.