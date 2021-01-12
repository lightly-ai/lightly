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

For a full list of supported arguments run

.. code-block:: bash
    
    lightly-train --help


You can get an overview of the various parameters in 
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
You will get a *token* and *dataset_id* which can be used to upload your dataset

.. code-block:: bash

    # upload only the dataset
    lightly-upload input_dir=cat token=your_token dataset_id=your_dataset_id

    # you can upload the dataset together with the embedding
    lightly-upload input_dir=cat embedding=your_embedding.csv \
                   token=your_token dataset_id=your_dataset_id

.. note:: To obtain your *token* and *dataset_id* check: 
          :ref:`ref-authentication-token` and :ref:`ref-webapp-dataset-id`.


.. _ref-upload-embedding-lightly:

Upload embeddings using the CLI 
----------------------------------

You can upload embeddings directly to the Lightly Platform using the CLI.

.. code-block:: bash

    # upload only the embedding
    lightly-upload embedding=your_embedding.csv token=your_token \
                   dataset_id=your_dataset_id

    # you can upload the dataset together with the embedding
    lightly-upload input_dir=cat embedding=your_embedding.csv \
                   token=your_token dataset_id=your_dataset_id


Download data using the CLI
-----------------------------------------------
You can download a dataset with a given tag from the Lightly Platform using the 
following CLI command. The CLI provides you with two options. Either you 
download just a list or copy the files from the original dataset into a new 
folder. The second option is very handy for quick prototyping.

.. code-block:: bash

    # download a list of files
    lightly-download tag_name=my_tag_name dataset_id=your_dataset_id token=your_token

    # copy files in a tag to a new folder
    lightly-download tag_name=my_tag_name dataset_id=your_dataset_id token=your_token \
                     input_dir=cat output_dir=cat_curated

