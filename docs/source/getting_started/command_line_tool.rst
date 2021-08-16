.. _lightly-command-line-tool:

Command-line tool
=================

The Lightly framework provides you with a command-line interface (CLI) to train 
self-supervised models and create embeddings without having to write a single 
line of code.

You can also have a look at this video to get an overview of how to work with 
the CLI.


.. raw:: html

    <div style="position: relative; height: 0; 
        overflow: hidden; max-width: 100%; padding-bottom: 20px; height: auto;">
        <iframe width="560" height="315" 
            src="https://www.youtube.com/embed/66a4O5G2Ajo" 
            frameborder="0" allow="accelerometer; autoplay; 
            clipboard-write; encrypted-media; gyroscope; picture-in-picture" 
            allowfullscreen>
        </iframe>
    </div>


Check the installation of lightly
-----------------------------------
To see if the lightly command-line tool was installed correctly, you can run the
following command which will print the installed lightly version:

.. code-block:: bash

    lightly-version

If lightly was installed correctly, you should see something like this:

.. code-block:: bash

    lightly version 1.1.4

Crop Images using Labels or Predictions
---------------------------------------------------
For some tasks, self-supervised learning on an image level has disadvantages. For 
example, when training an object detection model we care about local features
describing the objects rather than features describing the full image.

One simple trick to overcome this limitation, is to use labels or to use a pre-trained model
to get bounding boxes around the objects and then cropping the objects out of the
image.

We can do this using the **lightly-crop** CLI command. The CLI command crops 
objects out of the input images based on labels and copies them into an output folder.
The new folder consists now of the cropped images.

.. code-block:: bash

    # Crop images and set the crop to be 20% around the bounding box
    lightly-crop input_dir=images label_dir=labels output_dir=cropped_images crop_padding=0.2

    # Crop images and use the class names in the filename
    lightly-crop input_dir=images label_dir=labels output_dir=cropped_images \
                 label_names_file=data.yaml

The labels should be in the yolo format. For each image you should have a
corresponding .txt file. Each row in the .txt file has the following format:

* class x_center y_center width height

.. code-block:: text

    0 0.23 0.14 0.05 0.04
    1 0.43 0.13 0.12 0.08

An example for the label names .yaml file:

.. code-block:: yaml

    names: [cat, dog]

You can use the output of the lightly-crop command as the *input_dir* for your
lightly-train command.

Training, Embedding, and Uploading in a go - Magic
---------------------------------------------------
Lightly-magic is a singular command for training, embedding, and uploading to the Lightly Platform. 
    
* To start with, we need to input the directory of the dataset, pass it to input_dir 
* It requires information on the number of epochs to perform, set trainer.max_epochs,
* To use a pre-trained model, simply set trainer.max_epochs=0.
* The embedding model is used to embed all images in the input directory and saves the embeddings in a CSV file. A new dataset with the specified name is created on the Lightly platform. The embeddings file is uploaded to it, and the images themselves are uploaded with loader.num_workers workers in parallel, this utilizes CPU cores for faster processing.
* To set a custom batch size just set the value to loader.batch_size for the same.
* To parse in a new dataset set the value of  new_dataset_name as the name of the dataset. For example, the dataset to be loaded is myNewDataset, pass the value as new_dataset_name=myNewDataset
    
    
    
.. code-block:: bash
    
    # embedding and uploading the images and embeddings to web app
    # passing in input directory of dataset(Data_dir),using pretrained model
    # with assigned token as yourToken and datasedID as yourDatasedID
    # loader.num_workers species the number of cpu cores used for speeding the process.
    lightly-magic input_dir=data_dir trainer.max_epochs=0 loader.num_workers=8 \
    token=yourToken dataset_id=yourDatasetId
    
    # passing a custom dataset in the above example instead of datasetId
    lightly-magic input_dir=data_dir trainer.max_epochs=0 loader.num_workers=8 \
    token=yourToken new_dataset_name=myNewDataset
    
    # To use a custom batch size, pass the batch size to loader.batch_size parameter
    # updating the previous example by passing value for loader.batch_size
    lightly-magic input_dir=data_dir trainer.max_epochs=0 loader.batch_size=128 \
    loader.num_workers=8 token=yourToken new_dataset_name=myNewDataset


Train a model using the CLI
---------------------------------------
Training a model using default parameters can be done with just one command. Let's
assume you have a folder of cat images named `cat` and want to train a model on it.
You can use the following command to train a model and save the checkpoint:

.. code-block:: bash

    # train a model using default parameters
    lightly-train input_dir=cat

    # train a model for 5 epochs
    lightly-train input_dir=cat trainer.max_epochs=5

    # continue training from a checkpoint for another 10 epochs
    lightly-train input_dir=cat trainer.max_epochs=10 checkpoint=mycheckpoint.ckpt

For a full list of supported arguments run

.. code-block:: bash
    
    lightly-train --help


You can get an overview of the various CLI parameters you can set in 
:ref:`ref-cli-config-default`. 


.. _ref-cli-embeddings-lightly:

Create embeddings using the CLI
-----------------------------------------
Once you have a trained model checkpoint, you can create an embedding of a dataset.

.. code-block:: bash

    # use pre-trained models provided by Lighly
    lightly-embed input_dir=cat

    # use custom checkpoint
    lightly-embed input_dir=cat checkpoint=mycheckpoint.ckpt


The embeddings.csv file should look like the following:

.. csv-table:: embeddings_example.csv
   :header: "filenames","embedding_0","embedding_1","embedding_2","embedding_3","labels"
   :widths: 20, 20, 20, 20, 20, 20
    
    101053-1.jpg,-51.535,-2.325,-21.750,78.265,0
    101101-1.jpg,-67.958,-2.800,-28.861,103.812,0
    101146-1.jpg,-59.831,-2.719,-25.413,90.945,0


.. _ref-upload-data-lightly:

Upload data using the CLI
--------------------------------------------------------

In this example we will upload a dataset to the Lightly Platform.
First, make sure you have an account on `Lightly <https://www.lightly.ai>`_. 
A free account is sufficient. Log in to the app and create a new dataset. 
You will get a *token* and *dataset_id* which can be used to upload your dataset.
Alternatively, you can create a new dataset directly with the *token*
by providing the *new_dataset_name* instead of the *dataset_id*.

.. code-block:: bash

    # upload only the dataset
    lightly-upload input_dir=cat token=your_token dataset_id=your_dataset_id

    # you can upload the dataset together with the embeddings
    lightly-upload input_dir=cat embeddings=your_embedding.csv \
                   token=your_token dataset_id=your_dataset_id

    # create a new dataset and upload to it
    lightly-upload input_dir=cat token=your_token new_dataset_name=your_dataset_name

.. note:: To obtain your *token* and *dataset_id* check: 
          :ref:`ref-authentication-token` and :ref:`ref-webapp-dataset-id`.


.. _ref-upload-embedding-lightly:

Upload embeddings using the CLI 
----------------------------------

You can upload embeddings directly to the Lightly Platform using the CLI.
Again, you can use the *dataset_id* and *new_dataset_name* interchangeably.

.. code-block:: bash

    # upload only the embeddings
    lightly-upload embeddings=your_embedding.csv token=your_token \
                   dataset_id=your_dataset_id

    # you can upload the dataset together with the embeddings
    lightly-upload input_dir=cat embeddings=your_embedding.csv \
                   token=your_token new_dataset_name=your_dataset_name


.. _ref-upload-custom-metadata-lightly:

Upload custom metadata using the CLI
---------------------------------------
    
You can upload custom metadata along with your images. Custom metadata can be used
to gain additional insights in the web-app (see :ref:`platform-custom-metadata`).

To upload custom metadata, simply pass it to the `lightly-magic` command:

.. code-block:: bash

    lightly-magic input_dir=data_dir trainer.max_epochs=0 loader.num_workers=8 \
    token=yourToken dataset_id=yourDatasetId custom_metadata=yourCustomMetadata.json

Alternatively, you can upload custom metadata to an already existing dataset like this:

.. code-block:: bash

    lightly-upload token=yourToken dataset_id=yourDatasetId custom_metadata=yourCustomMetadata.json

.. note::

    You can learn more about the required format of the custom metadata at :ref:`platform-custom-metadata`.


Download data using the CLI
-----------------------------------------------
You can download a dataset with a given tag from the Lightly Platform using the 
following CLI command. The CLI provides you with three options:

* Download the list of filenames for a given tag in the dataset.
  
* Download the images for a given tag in the dataset.
  
* Copy the images for a given tag from an input directory to a target directory.

The last option allows you to very quickly extract only the images in a given tag
without the need to download them explicitly.

.. code-block:: bash

    # download a list of files
    lightly-download tag_name=my_tag_name dataset_id=your_dataset_id token=your_token

    # download the images and store them in an output directory
    lightly-download tag_name=my_tag_name dataset_id=your_dataset_id token=your_token \
                     output_dir=path/to/output/dir

    # copy images from an input directory to an output directory
    lightly-download tag_name=my_tag_name dataset_id=your_dataset_id token=your_token \
                     input_dir=path/to/input/dir output_dir=path/to/output/dir




    




