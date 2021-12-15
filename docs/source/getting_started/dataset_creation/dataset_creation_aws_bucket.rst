How to use S3 with Lightly
------------------------------


Lightly allows you to configure a remote datasource like Amazon S3 (Amazon Simple Storage Service).
In this guide, we will show you how to setup your S3 bucket, configure your dataset to use said bucket, and only upload metadata to Lightly.


Setting up Amazon S3
^^^^^^^^^^^^^^^^^^^^^^
For Lightly to be able to create so-called `presigned URLs/read URLs <https://docs.aws.amazon.com/AmazonS3/latest/userguide/ShareObjectPreSignedURL.html>`_ to be used for displaying your data in your browser, Lightly needs at minimum to be able to read and list permissions on your bucket. If you want Lightly to create optimal thumbnails for you while uploading the metadata of your images, write permissions are also needed.

Let us assume your bucket is called `datalake`. And let us assume the folder you want to use with Lightly is located at projects/farm-animals/

**Setting up IAM**

1. Go to the `Identity and Access Management IAM page <https://console.aws.amazon.com/iamv2/home?#/users>`_ and create a new user for Lightly.
2. Choose a unique name of your choice and select "Programmatic access" as "Access type". Click next
    
    .. figure:: resources/AWSCreateUser2.png
        :align: center
        :alt: Create AWS User

        Create AWS User

3. We will want to create very restrictive permissions for this new user so that it can't access other resources of your company. Click on "Attach existing policies directly" and then on "Create policy". This will bring you to a new page
    
    .. figure:: resources/AWSCreateUser3.png
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
    .. figure:: resources/AWSCreateUser4.png
        :align: center
        :alt: Permission policy in AWS

        Permission policy in AWS
5. Go to the next page and create tags as you see fit (e.g `external` or `lightly`) and give a name to your new policy before creating it.

    .. figure:: resources/AWSCreateUser5.png
        :align: center
        :alt: Review and name permission policy in AWS

        Review and name permission policy in AWS
6. Return to the previous page as shown in the screenshot below and reload. Now when filtering policies your newly created policy will show up. Select it and continue setting up your new user.
    
    .. figure:: resources/AWSCreateUser6.png
        :align: center
        :alt: Attach permission policy to user in AWS

        Attach permission policy to user in AWS
7. Write down the `Access key ID` and the `Secret access key` in a secure location (such as a password manager) as you will not be able to access this information again (you can generate new keys and revoke old keys under `Security credentials` of a users detail page)
    
    .. figure:: resources/AWSCreateUser7.png
        :align: center
        :alt: Get security credentials (access key id, secret access key) from AWS

        Get security credentials (access key id, secret access key) from AWS

**Preparing your data**


For Lightly to be able to create embeddings and extract metadata from your data, `lightly-magic` needs to be able to access your data. You can either download/sync your data from S3 or you can mount S3 as a drive. We recommend downloading your data from S3 as it makes the overall process faster.

**Downloading from S3 (recommended)**

1. Install AWS cli by following the `guide of Amazon <https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2.html>`_
2. Run `aws configure` and set the credentials
3. Download/synchronize the folder located on S3 to your current directory `aws s3 sync s3://datalake/projects/farm-animals ./farm`

**Mount S3 as a drive**

For Linux and MacOS we recommend using `s3fs-fuse <https://github.com/s3fs-fuse/s3fs-fuse>`_ to mount S3 buckets to a local file storage. 
You can have a look at our step-by-step guide: :ref:`ref-docker-integration-s3fs-fuse`. 


Uploading your data
^^^^^^^^^^^^^^^^^^^^^^

Create and configure a dataset

1. `Create a new dataset <https://app.lightly.ai/dataset/create>`_ in Lightly
2. Edit your dataset and select S3 as your datasource

    .. figure:: resources/LightlyEdit1.png
        :align: center
        :alt: Get security credentials (access key id, secret access key) from AWS

        Get security credentials (access key id, secret access key) from AWS

3. As the resource path, enter the full S3 URI to your resource eg. `s3://datalake/projects/farm-animals/`
4. Enter the `access key` and the `secret access key` we obtained from creating a new user in the previous step and select the AWS region in which you created your bucket in
5. The thumbnail suffix allows you to configure
   
    - where your thumbnails are stored when you already have generated thumbnails in your S3 bucket
    - where your thumbnails will be stored when you want Lightly to create thumbnails for you. For this to work, the user policy you have created must possess write permissions.
    - when the thumbnail suffix is not defined/empty, we will load the full image even when requesting the thumbnail.
    
    .. figure:: resources/LightlyEdit2.png
        :align: center
        :alt: Lightly S3 connection config
        :width: 60%

        Lightly S3 connection config

6. Press save and ensure that at least the lights for List and Read turn green.


Use Lightly
Use `lightly-magic` and `lightly-upload` just as you always would with the following considerations;

- If you already have generated thumbnails, don't want to see thumbnails or just want to use the full image for a thumbnail (by setting the thumbnail suffix to empty), add `upload=metadata` to the `lightly-magic` command.
- If you want Lightly to create thumbnails for you, you can add `upload=thumbnails` to the `lightly-magic` command.