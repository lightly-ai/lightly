.. _dataset-creation-aws-bucket:

Create a dataset from an AWS S3 bucket
=======================================


Lightly allows you to configure a remote datasource like Amazon S3 (Amazon Simple Storage Service).
In this guide, we will show you how to setup your S3 bucket, configure your dataset to use said bucket, and only upload metadata to Lightly.


Setting up Amazon S3
----------------------

For Lightly to be able to create so-called `presigned URLs/read URLs <https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html>`_ to be used for displaying your data in your browser, Lightly needs at minimum to be able to read and list permissions on your bucket. If you want Lightly to create optimal thumbnails for you while uploading the metadata of your images, write permissions are also needed.

Let us assume your bucket is called `datalake`. And let us assume the folder you want to use with Lightly is located at projects/farm-animals/

**Setting up IAM**

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

4. As our policy is very simple, we will use the JSON option and enter the following while substituting `datalake` with your bucket and `projects/farm-animals/` with the folder you want to share.
    
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




**S3 IAM Delegated Access**

To access your data in your S3 bucket on AWS, Lightly `can assume a role <https://docs.aws.amazon.com/IAM/latest/UserGuide/tutorial_cross-account-with-roles.html>`_ in your account which has the necessary permissions to access your data.
This is `considered best practice <https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html#delegate-using-roles>`_ by AWS.

To set up IAM Delegated Access

1. Go to the `AWS IAM Console <https://console.aws.amazon.com/iam/home?#/roles>`_

2. Click `Create role`
   
3. Select `AWS Account` as the trusted entity type

    a. Select `Another AWS account` and specify the AWS Account ID of Lightly: `311530292373`

    b. Check `Require external ID`, and choose an external ID. The external ID should be treated like a passphrase

    c. Do not check `Require MFA`.
    
    d. Click next

4. Select a policy which grants access to your S3 bucket. If no policy has previously been created, here is an example of how the policy should look like:



    .. code-block:: json
            
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "lightlyS3Access",
                    "Action": [
                        "s3:*Object",
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


7. Remember the external ID and the ARN of the newly created role



Preparing your data
^^^^^^^^^^^^^^^^^^^^^

For the :ref:`lightly-command-line-tool` to be able to create embeddings and extract metadata from your data, `lightly-magic` needs to be able to access your data. You can either download/sync your data from S3 or you can mount S3 as a drive. We recommend downloading your data from S3 as it makes the overall process faster.

Prepare data by downloading from S3 (recommended)
""""""""""""""""""""""""""""""""""""""""""""""""""

1. Install AWS cli by following the `guide of Amazon <https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html>`_
2. Run `aws configure` and set the credentials
3. Download/synchronize the folder located on S3 to your current directory

    .. code-block:: console

        aws s3 sync s3://datalake/projects/farm-animals ./farm


Prepare data by mounting S3 as a drive
"""""""""""""""""""""""""""""""""""""""

For Linux and MacOS we recommend using `s3fs-fuse <https://github.com/s3fs-fuse/s3fs-fuse>`_ to mount S3 buckets to a local file storage. 
You can have a look at our step-by-step guide: :ref:`ref-docker-integration-s3fs-fuse`. 


Uploading your data
---------------------

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

Use `lightly-magic` and `lightly-upload` just as you always would with the following considerations;

- Use `input_dir=datalake/farm-animals`
- If you chose the option to generate thumbnails in your bucket, use the argument `upload=thumbnails`
- Otherwise, use `upload=metadata` instead.
