.. _dataset-creation-azure-storage:


Create a dataset from Azure blob storage
=========================================

Lightly allows you to configure a remote datasource like Microsoft's Azure blob storage.
In this guide, we will show you how to setup your Azure blob storage container and configure your dataset to use said container.

One decision you need to make first is whether you want to use thumbnails.
Using thumbnails makes the Lightly Platform more responsive, as it's enough to
load the thumbnails instead of the full images in many cases.
As a drawback, the thumbnails will be stored in your bucket and thus need storage.
Depending on this decision, the following steps will differ slightly.


Setting up Azure
------------------

For the purpose of this guide we assume you have a storage account called `lightlydatalake`.
We further assume the container you want to use with lightly is called `farm-animals` and already contains images.
If you don't have a storage account or container yet follow the instructions `here <https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal>`_.

Go to "Security + networking > Access keys". Copy the Key and store it in a secure location. Head to the next section to see how you can configure the Lightly dataset.


Preparing your data
^^^^^^^^^^^^^^^^^^^^^

For Lightly to be able to create embeddings and extract metadata from your data, `lightly-magic` needs to be able to access your data. You can either download/sync your data from S3 or you can mount S3 as a drive. We recommend downloading your data from S3 as it makes the overall process faster.
 
1. Install AzCopy cli by following the `guide of Azure <https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10>`_
2. To copy your data from the container to your local machine, go to "Security + networking > Shared access signature" on the storage account page in the Azure portal. Generate a shared access signature (SAS) which allows access to the container and objects.
3. Copy the SAS token and use the following command:

    .. code-block::

        azcopy copy 'https://lightlydatalake.blob.core.windows.net/{YOUR_SAS_TOKEN}' '/local/lightlydatalake/farm-animals' --recursive






Uploading your data
--------------------

1. `Create a new dataset <https://app.lightly.ai/dataset/create>`_ in Lightly
2. Edit your dataset and select `Azure Blob Storage` as your datasource

    .. figure:: ../resources/LightlyEdit1.png
        :align: center
        :alt: Edit your dataset.

        To edit your dataset click the `Edit` button on the top.

3. As your container name enter `farm-animals`.
4. Enter the storage account name and storage account key from the previous step.
5. The thumbnail suffix depends on the option you chose in the first step
   
    - You want Lightly to create the thumbnail for you.
      Then choose the naming scheme to your liking.
    - You have already generated thumbnails in your bucket.
      Then choose the thumbnail suffix such that it reflects your naming scheme.
    - You don't want to use thumbnails.
      Then leave the thumbnail suffix undefined/empty.

    .. figure:: ../resources/LightlyEditAzure.jpg
        :align: center
        :alt: Lightly Azure Blob Storage config.
        :width: 60%

        Lightly Azure Blob Storage config.

6. Press save and ensure that all lights turn green.


To add the images to the dataset use `lightly-magic` or `lightly-upload` with the following parameters:

- Use `input_dir=/local/lightlydatalake/farm-animals`
- If you chose the option to generate thumbnails in your bucket,
  use the argument `upload=thumbnails`
- Otherwise, use `upload=metadata` instead.
