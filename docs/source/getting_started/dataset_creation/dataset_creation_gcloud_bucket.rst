.. _dataset-creation-gcloud-bucket:

Create a dataset from Google Cloud Storage
------------------------------------------

Lightly allows you to configure a remote datasource like
`Google Cloud Storage <https://cloud.google.com/storage>`_ .
In this guide, we will show you how to setup your Google Cloud Storage,
configure your dataset to use said bucket, and only upload metadata to Lightly
while keeping your data private.


Setting up Google Cloud Storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Lightly needs to be able to create so-called
`presigned URLs/read URLs <https://cloud.google.com/storage/docs/access-control/signed-urls>`_
for displaying your data in your browser.
Thus it needs at minimum to be able to read and list permissions on your bucket.
If you want Lightly to create optimal thumbnails for you
while uploading the metadata of your images, write permissions are also needed.

Let us assume the bucket is called `lightly-datalake`.
And let us assume the folder you want to use with Lightly is located at `projects/wild-animals/`

**Setting Up a Service Account in IAM**

1. Write down your project ID.
You find it in the `gcloud console <console.cloud.google.com/home/dashboard>`_ under `Project Info`.

2. Navigate to your bucket in the `google cloud storage browser <https://console.cloud.google.com/storage/browser>`_
and from there to `projects/wild-animals/`. Copy the path, in this case
`lightly-datalake/projects/wild-animals`.

.. figure:: ./images_gcloud_bucket/screenshot_gcloud_bucket_project.png
    :align: center
    :alt: Browsing a google cloud storage bucket.
    :width: 60%

3. Navigate to the tab `Permissions`.
Make sure that your access control is uniform.
If it is not, change it to uniform.

.. figure:: ./images_gcloud_bucket/screenshot_gcloud_uniform_access.png
    :align: center
    :alt: Ensuring a google cloud bucket has uniform access.
    :width: 60%

4. Navigate to `IAM & Admin -> Roles <https://console.cloud.google.com/iam-admin/roles>`_.

- There create a new role, with the title and ID `STORAGE_READ_WRITE_LIST`.
- Click on `Add Permissions`, search for `storage.objects`
- Add the permissions `storage.objects.get`, `storage.objects.list`, and `storage.objects.create`.
  The create permissions are needed so that Lightly can create thumbnails in your bucket.
- After adding the permissions, create the role.

.. figure:: ./images_gcloud_bucket/screenshot_gcloud_storage_role.png
    :align: center
    :alt: Creating a role for accessing google cloud storage.
    :width: 60%

5. Navigate to `APIs -> Credentials <https://console.cloud.google.com/apis/credentials>`_.

- Click on `Create Credentials`, choose `Service Account` and insert the name
  `LIGHTLY_USER_WILD_ANIMALS`.
- The description can be `service account for the Lightly API to access the wild animals dataset`.
- Click on `Create and Continue`.
- Choose the Role you just created, i.e. `STORAGE_READ_WRITE_LIST`.
- Add a condition with the title `BUCKET_PROJECTS_WILD_ANIMALS`
  and insert the condition below in the Condition editor. Remember to change the bucket name
  and path to the folder. However, you must keep the "objects" inbetween.
  For more information, head to the `IAM conditions
  <https://cloud.google.com/storage/docs/access-control/iam#conditions>`_.
  The first part of the condition adds listing rights to the whole bucket,
  as they can only be handled on the bucket level. The second part adds object-level
  access rights (i.e. read and create) for all objects in the bucket `lightly-datalake`
  whose name starts with `projects/wild-animals`.

.. code::

    (
        resource.type == 'storage.googleapis.com/Bucket' &&
        resource.name.startsWith("projects/_/buckets/lightly-datalake")
    ) || (
        resource.type == 'storage.googleapis.com/Object' &&
        resource.name.startsWith("projects/_/buckets/lightly-datalake/objects/projects/wild-animals")
    )
- Click on `Done` to create the service account.
- You can change the roles of the service account later in the
  `IAM <https://console.cloud.google.com/iam-admin/iam>`_.

6. Navigate to `APIs -> Credentials <https://console.cloud.google.com/apis/credentials>`_
   again if you are not already there.

- Find the just created user in the list of all service accounts.
- Click on the user and navigate to the `keys` tab.
- Click on `Add key` and create a new private key in JSON Format.
  It will download the corresponding key file.

.. figure:: images_gcloud_bucket/screenshot_gcloud_service_account_key_creation.png
    :align: center
    :alt: Google Cloud Service Account Key Creation
    :width: 60%




Configuring a Lightly dataset to access the Google Cloud Storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create and configure a dataset

1. `Create a new dataset <https://app.lightly.ai/dataset/create>`_ in Lightly
2. Edit your dataset and select `Google Cloud Storage` as your datasource

.. figure:: images_gcloud_bucket/screenshot_gcloud_create_dataset.png
    :align: center
    :alt: Configure google cloud bucket datasource in Lightly Webapp
    :width: 60%


3. As the resource path, enter the full URI to your resource eg. `gs://lightly-datalake/projects/wild-animals`
4. Enter the Google Project ID you wrote down in the first step.
5. Click on `Select Credentials File` to add the key file you downloaded in the previous step.
6. The thumbnail suffix allows you to configure

- where your thumbnails are stored when you already have generated thumbnails in your S3 bucket
- where your thumbnails will be stored when you want Lightly to create thumbnails for you.
  For this to work, the user policy you created must be granted write permissions.
- when the thumbnail suffix is not defined/empty,
  the Lightly Webapp will load the full image even when requesting the thumbnail.


6. Press save and ensure that at least the lights for List and Read turn green.
If you added permissions for writing, this lights should also turn green.

7. Now you should be on the dataset creation page again.

Create the dataset and upload embeddings and metadata.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For creating the dataset and uploading embeddings and metadta to it, you need
the :ref:`lightly-command-line-tool`. Furthermore, you need to have your data locally.
This can be done easiest by using the `gsutil tool <https://cloud.google.com/storage/docs/gsutil>`_
and its `rsync command <https://cloud.google.com/storage/docs/gsutil/commands/rsync>`_:

.. code::

    gsutil -m rsync -r /local/projects/wild-animals gs://datalake-lightly/projects/wild-animals


Use `lightly-magic` and `lightly-upload` with the following considerations:

- use `input_dir=/local/projects/wild-animals`
- If you want to use thumbnails for a more responsive Lightly Webapp,
  add `upload=thumbnails` to the `lightly-magic` command. They will be written
  to your Google Cloud Bucket if you have given write access to it.
- If you have already generated thumbnails in your bucket or don't want to use
  thumbnails, use `upload=metadata` instead.