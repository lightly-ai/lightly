.. _dataset-creation-azure-storage:


Create a dataset from Azure blob storage
=========================================

Lightly allows you to configure a remote datasource like Microsoft's Azure blob storage.
In this guide, we will show you how to setup your Azure blob storage container and configure your dataset to use said container.

One decision you need to make first is whether you want to use thumbnails.
Using thumbnails makes the Lightly Platform more responsive, as not always
the full images will be loaded.
However, the thumbnails will be stored in your container and thus need storage.

.. note::
  
    If you want to use thumbnails you need to give
    Lightly write access to your container to create the thumbnails there for you.
    The write access can be configured not to allow overwriting and
    deleting, thus existing data cannot get lost.

Setting up Azure
------------------

For the purpose of this guide we assume you have a storage account called `lightlydatalake`.
We further assume the container you want to use with lightly is called `farm-animals` and already contains images.
If you don't have a storage account or container yet follow the instructions `here <https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal>`_.

Go to **"Security + networking"** > **"Shared access signature"**. There, configure the access rights to your liking.
Lightly requires at least read and list access to the container. Write access is required for thumbnails or when working with videos and object level.
Make sure to *also* tick the boxes for the required resource types (Container and Object). Set an expiry date and then click on **"Generate SAS and connection string"**. Copy the generated `SAS token` *including the "?"*
and store it in a secure location. Head to the next section to see how you can configure the Lightly dataset.


Preparing your data
^^^^^^^^^^^^^^^^^^^^^

For the :ref:`lightly-command-line-tool` to be able to create embeddings and extract metadata from your data, `lightly-magic` needs to be able to access your data. You can download/sync your data from Azure blob storage.
 
1. Install AzCopy cli by following the `guide of Azure <https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10>`_
2. To sync your data from the container to your local machine (or vice versa), you need a shared access token. On your `storage account` page of the Azure portal, on the left, go to **"Security + networking"** > **"Shared access signature"**. Generate a shared access signature (SAS) which allows access to the container and objects.
3. Copy the `SAS token` and use the following command to sync the Azure blob storage with your local folder:

    .. code-block::

        azcopy sync '/local/lightlydatalake/farm-animals' 'https://lightlydatalake.blob.core.windows.net/farm-animals/{YOUR_SAS_TOKEN}' --recursive



Uploading your data
--------------------

1. `Create a new dataset <https://app.lightly.ai/dataset/create>`_ in Lightly
2. Edit your dataset and select `Azure Blob Storage` as your datasource

    .. figure:: ../resources/resources_datasource_configure/LightlyEditAzure.jpg
        :align: center
        :alt: Lightly Azure Blob Storage config.
        :width: 60%

        Lightly Azure Blob Storage config.

3. As your container name enter `farm-animals`.
4. Enter the storage account name and SAS token from the previous step.
5. Toggle the **"Generate thumbnail"** switch if you want Lightly to generate thumbnails for you.
6. If you want to store outputs from Lightly (like thumbnails or extracted frames) in a different directory, you can toggle **"Use a different output datasource"** and enter a different path in your bucket. This allows you to keep your input directory clean as nothing gets ever written there.
  .. note:: 

    Lightly requires list, read, and write access to the `output datasource`. Make sure you have configured it accordingly in the steps before.
7. Press save and ensure that at least the lights for List and Read turn green. If you added permissions for writing, this light should also turn green.

8. Press save and ensure that all lights turn green.


To add the images to the dataset use `lightly-magic` or `lightly-upload` with the following parameters:

- Use `input_dir=/local/lightlydatalake/farm-animals`
- If you chose the option to generate thumbnails in your bucket,
  use the argument `upload=thumbnails`
- Otherwise, use `upload=metadata` instead.
