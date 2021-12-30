.. _dataset-creation-azure-storage:


Create a dataset from Azure blob storage
-----------------------------------------

Lightly allows you to configure a remote datasource like Microsoft's Azure blob storage.
In this guide, we will show you how to setup your Azure blob storage container and configure your dataset to use said container.

One decision you need to make first is whether you want to use thumbnails.
Using thumbnails makes the Lightly Platform more responsive, as it's enough to
load the thumbnails instead of the full images in many cases.
As a drawback, the thumbnails will be stored in your bucket and thus need storage.
You have the following three options:


- You want to use thumbnails, but don't have them yet. Then you need to give
  Lightly write access to your bucket to create the thumbnails there for you.
- You already have thumbnails in your bucket with a consistent name scheme, e.g.
  an image called `img.jpg` has a corresponding thumbnail called `img_thumb.jpg`.
  In this case, a read access to your bucket is sufficient.
- You don't want to use thumbnails. Then a read access to your bucket
  is sufficient. The Lightly Platform will load the full image
  even when requesting the thumbnail.

Depending on this decision, the following steps will differ slightly.

Setting up Azure
^^^^^^^^^^^^^^^^^

For the purpose of this guide we assume you have a storage account called `lightlydatalake`.
We further assume the container you want to use with lightly is called `farm-animals` and already contains images.
If you don't have a storage account or container yet follow the instructions `here <https://docs.microsoft.com/en-us/azure/storage/common/storage-account-create?tabs=azure-portal>`_.
Head to the next section to see how you can configure the Lightly dataset.


Configuring a Lightly dataset to access the Azure storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
      Then choose the thumbnail suffix such that it reflects you naming scheme.
    - You don't want to use thumbnails.
      Then leave the thumbnail suffix undefined/empty.

    .. figure:: ../resources/LightlyEditAzure.jpg
        :align: center
        :alt: Lightly Azure Blob Storage config.
        :width: 60%

        Lightly Azure Blob Storage config.

6. Press save and ensure that at least the lights for List and Read turn green.


Uploading your data
^^^^^^^^^^^^^^^^^^^^


For creating the dataset and uploading embeddings and metadata to it you need
the Lightly :ref:`lightly-command-line-tool`.

Furthermore, you need to have your data locally on your machine.
This can easily be done by using `AzCopy <https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10>`_.
To copy your data from the container to your local machine, go to "Security + networking > Shared access signature" on the storage account
page in the Azure portal. Generate a shared access signature (SAS) which allows access to the container and objects. Copy the SAS token and use
the following command:


.. code-block::

    azcopy copy 'https://lightlydatalake.blob.core.windows.net/{YOUR_SAS_TOKEN}' '/local/lightlydatalake/farm-animals' --recursive


To add the images to the dataset use `lightly-magic` or `lightly-upload` with the following parameters:

- Use `input_dir=/local/lightlydatalake/farm-animals`
- If you chose the option to generate thumbnails in your bucket,
  use the argument `upload=thumbnails`
- Otherwise, use `upload=metadata` instead.
