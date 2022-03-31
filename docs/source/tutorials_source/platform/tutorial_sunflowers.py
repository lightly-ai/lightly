"""

.. _lightly-tutorial-sunflowers:

Tutorial 2: Diversify the Sunflowers Dataset
=============================================

This tutorial highlights the basic functionality of selecting in the web-app.
You can use the CORESET selection strategy to choose a diverse subset of your dataset.
This can be useful many purposes, e.g. for having a good subset of data to label
or for creating a validation or test dataset that covers the complete sample space.
Removing duplicate images can also help you in reducing bias and imbalances in your dataset.

What you will learn
--------------------

* Upload images and embeddings to the web-app via the Python package
* Sample a diverse subset of your original dataset in the web-app
* Download the filenames of the subset and use it to create a new local dataset folder.

Requirements
-------------
You can use your own dataset or the one we provide for this tutorial. The dataset
we provide consists of 734 images of sunflowers. You can
download it here :download:`Sunflowers.zip <../../../_data/Sunflowers.zip>`.

To use the Lightly platform, we need to upload the dataset with embeddings to it.
The first step for this is to train a self-supervised embedding model.
Then, embed your dataset and lastly, upload the dataset and embeddings to the Lightly platform.
These three steps can be done using a single terminal command from the lightly pip package: lightly-magic
But first, we need to install lightly from the Python package index.

.. code-block:: bash

    # Install lightly as a pip package
    pip install lightly

.. code-block:: bash

    # The lightly-magic command first needs the input directory of your dataset.
    # Then it needs the information for how many epochs to train an embedding model on it.
    # If you want to use our pretrained model instead, set trainer.max_epochs=0.
    # Next, the embedding model is used to embed all images in the input directory
    # and saves the embeddings in a csv file. Last, a new dataset with the specified name
    # is created on the Lightly platform.

    lightly-magic input_dir="./Sunflowers" trainer.max_epochs=0 token=YOUR_TOKEN
    new_dataset_name="sunflowers_dataset"

.. note::

   The lightly-magic command with prefilled parameters is displayed in the web-app when you
   create a new dataset. `Head over there and try it! <https://app.lightly.ai>`_
   For more information on the CLI commands refer to :ref:`lightly-command-line-tool` and :ref:`lightly-at-a-glance`.

Create a Selection
------------------

Now, you have everything you need to create a selection of your dataset. For this,
head to the *Embedding* page of your dataset. You should see a two-dimensional
scatter plot of your embeddings. If you hover over the images, their thumbnails
will appear. Can you find clusters of similar images?

.. figure:: ../../tutorials_source/platform/images/sunflowers_scatter_before_selection.jpg
    :align: center
    :alt: Alt text
    :figclass: align-center

    You should see a two-dimensional scatter plot of your dataset as shown above.
    Hover over an image to view a thumbnail of it.
    There are also features like selecting and browsing some images and creating
    a tag from it.

.. note::

   We reduce the dimensionality of the embeddings to 2 dimensions before plotting them.
   You can switch between the PCA, tSNE and UMAP dimensionality reduction methods.

Right above the scatter plot you should see a button "Create Sampling". Click on it to
create a selection. You will need to configure the following settings:

* **Embedding:** Choose the embedding to use for the selection.
* **Sampling Strategy:** Choose the selection strategy to use. This will be one of:

   * CORESET: Selects samples which are diverse.
   * CORAL: Combines CORESET with uncertainty scores to do active learning.
   * RANDOM: Selects samples uniformly at random.
* **Stopping Condition:** Indicate how many samples you want to keep.
* **Name:** Give your selection a name. A new tag will be created under this name.

.. figure:: ../../tutorials_source/platform/images/selection_create_request.png
    :align: center
    :alt: Alt text
    :figclass: align-center
    :figwidth: 400px

    Example of a filled out selection request in the web-app.

After confirming your settings, a worker will start processing your request. Once
it's done, the page switches to the new tag. You can see how the scatter plot now shows
selected images and discarded images in a different color. Play around with the different selection strategies
to see differences between the results.

.. figure:: ../../tutorials_source/platform/images/sunflowers_scatter_after_selection.jpg
    :align: center
    :alt: Alt text
    :figclass: align-center

    After the selection you can see which samples were selected and which ones were discarded.
    Here, the green dots are part of the new tag while the gray ones are left away. Notice
    how the CORESET selection strategy selects an evenly spaced subset of images.

.. note::

   The CORESET selection strategy chooses the samples evenly spaced out in the 32-dimensional space.
   This does not necessarily translate into being evenly spaced out after the dimensionality
   reduction to 2 dimensions.


Download a selection
--------------------

Now you can use this diverse subset for your machine learning project.
Just head over to the *Download* tag to see the different download options.
Apart from downloading the filenames or the images directly, you can also
use the lightly-download command to copy the files in the subset from your existing
to a new directory. The CLI command with prefilled arguments is already provided.


"""
