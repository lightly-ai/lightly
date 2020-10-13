Command-line tool
=================

The Lighly framework provides you with a command-line interface to train self-supervised models
and create embeddings without having to write a single line of code.

Train a self-supervised model
-----------------------------
Training a model using default parameters can be done with just one command. Let's
assume you have a folder of cat images named `cats` and want to train a model on it.
You can use the following command to train a model and save the checkpoint:

.. code-block:: bash

    # train a model using default parameters
    lightly-train input_dir=cats

    # train a model for 5 epochs
    lightly-train input_dir=cats trainer.max_epochs=5

For a full list of supported arguments run

.. code-block:: bash
    
    lightly-train --help

Create an embedding using a trained model
-----------------------------------------
Once you have a trained model checkpoint you can create an embedding of a dataset.

.. code-block:: bash

    # use pre-trained models provided by Lighly
    lightly-embed input_dir=cats

    # use custom checkpoint
    lightly-embed input_dir=cats checkpoint=mycheckpoint.ckpt


Upload the dataset and embedding to the Lightly platform
--------------------------------------------------------
You need to be registered on `Lightly <https://www.lightly.ai>`_. A free account is sufficient.
Log in to the app and create a new dataset. You will get a token and dataset_id which can 
be used to upload your dataset

.. code-block:: bash

    # upload only the dataset
    lightly-upload input_dir=cats token=your_token dataset_id=your_dataset_id

    # you can upload the dataset together with the embedding
    lightly-upload input_dir=cats embedding=your_embedding.csv \
                   token=your_token dataset_id=your_dataset_id

Download a dataset after curating on Lightly.ai
-----------------------------------------------
You can download a dataset with a given tag from the Lighly platform using the following CLI command.
The CLI provides you with two options. Either you download just a list or copy the files from the dataset 
into a new folder. The second option is very handy for quick prototyping.

.. code-block:: bash

    # download a list of files
    lightly-download tag_name=my_tag_name dataset_id=your_dataset_id token=your_token

    # copy files in a tag to a new folder
    lightly-download tag_name=my_tag_name dataset_id=your_dataset_id token=your_token \
                     input_dir=cats output_dir=cats_curated

