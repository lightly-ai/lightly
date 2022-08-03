.. _dataset-creation-aws-bucket-user-access:

**S3 IAM User Access**

1. Go to the `Identity and Access Management IAM page <https://console.aws.amazon.com/iamv2/home?#/users>`_ and create a new user for Lightly.
2. Choose a unique name of your choice and select **"Programmatic access"** as **"Access type"**. Click next

    .. figure:: ../resources/AWSCreateUser2.png
        :align: center
        :alt: Create AWS User

        Create AWS User

3. We will want to create very restrictive permissions for this new user so that it can't access other resources of your company. Click on **"Attach existing policies directly"** and then on **"Create policy"**. This will bring you to a new page

    .. figure:: ../resources/AWSCreateUser3.png
        :align: center
        :alt: Setting user permission in AWS

        Setting user permission in AWS

4. As our policy is very simple, we will use the JSON option and enter the following.
Please substitute `datalake` with the name of your bucket and `projects/farm-animals/` with the folder you want to share.

    .. code-block:: json

        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "VisualEditor0",
                    "Effect": "Allow",
                    "Action": "s3:ListBucket",
                    "Resource": [
                        "arn:aws:s3:::datalake",
                        "arn:aws:s3:::datalake/projects/farm-animals/*"
                    ]
                },
                {
                    "Sid": "VisualEditor1",
                    "Effect": "Allow",
                    "Action": "s3:*",
                    "Resource": [
                        "arn:aws:s3:::datalake/projects/farm-animals/*"
                    ]
                }
            ]
        }
    .. figure:: ../resources/AWSCreateUser4.png
        :align: center
        :alt: Permission policy in AWS

        Permission policy in AWS
5. Go to the next page and create tags as you see fit (e.g `external` or `lightly`) and give a name to your new policy before creating it.

    .. figure:: ../resources/AWSCreateUser5.png
        :align: center
        :alt: Review and name permission policy in AWS

        Review and name permission policy in AWS
6. Return to the previous page as shown in the screenshot below and reload. Now when filtering policies, your newly created policy will show up. Select it and continue setting up your new user.

    .. figure:: ../resources/AWSCreateUser6.png
        :align: center
        :alt: Attach permission policy to user in AWS

        Attach permission policy to user in AWS
7. Write down the `Access key ID` and the `Secret access key` in a secure location (such as a password manager) as you will not be able to access this information again (you can generate new keys and revoke old keys under `Security credentials` of a users detail page)

    .. figure:: ../resources/AWSCreateUser7.png
        :align: center
        :alt: Get security credentials (access key id, secret access key) from AWS

        Get security credentials (access key id, secret access key) from AWS

