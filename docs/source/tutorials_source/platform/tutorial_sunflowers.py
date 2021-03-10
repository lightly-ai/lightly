"""

Tutorial 2: Sample Sunflowers
=============================================

This tutorial highlights the basic functionality of sampling in the web-app.

What you will learn
--------------------

* Upload images to the web-app via frontend or Python package
* Upload embeddings to the web-app via frontend or the Python package
* Sample a diverse subset of your original dataset in the web-app

Requirements
-------------
You can use your own dataset or the one we provide for this tutorial. The dataset
we provide consists of 734 images of sunflowers and an embedding file. You can
download it here :download:`Sunflowers.zip <../../../_data/Sunflowers.zip>`.

You can also use our pre-trained models to create your own embeddings:

.. code::

   lightly-embed input_dir=/path/to/your/dataset

For more information refer to :ref:`lightly-command-line-tool` and :ref:`lightly-at-a-glance`.

Upload the Data
----------------

To upload images, you need to create a dataset on the `Lightly Platform <https://app.lightly.ai>`_.
You can then upload the images via drag and drop from your machine or use the Python package
with the displayed command:

.. code::

   lightly-upload input_dir=/path/to/your/dataset dataset_id=YOUR_DATASET_ID token=YOUR_TOKEN


Similarly, to upload the embeddings, you can use the drag and drop option on the  `Platform <https://app.lightly.ai>`_
under *Embedding* or use the Python package:

.. code::

   lightly-upload embeddings=path/to/embeddings.csv dataset_id=YOUR_DATASET_ID token=YOUR_TOKEN


.. note::

   The commands for uploading images and embeddings are displayed in the web-app when you
   create a new dataset. `Head over there and try it! <https://app.lightly.ai>`_


Create a Sampling
------------------

Now, you have everything you need to create a sampling of your dataset. For this,
head to the *Embedding* page of your dataset. You should see a two-dimensional
scatter plot of your embeddings. If you hover over the images, their thumbnails
will appear. Can you find clusters of similar images?

.. note::

   We use principal component analysis (PCA) to reduce the dimensionality of the
   embeddings before plotting them. This transformation approximately preserves
   relative distances and angles between points.

Right above the scatter plot you should see a button "Create". Click on it to
create a sampling. You will need to configure the following settings:

* **Embedding:** Choose the embedding to use for the sampling.
* **Sampling Strategy:** Choose the sampling strategy to use. This will be one of:

   * Coreset: Greedily selects samples which are diverse.
   * Coral: Combines Coreset with uncertainty scores to do active learning.
   * Random: Selects samples uniformly at random.
* **Stopping Condition:** Indicate how many samples you want to keep.
* **Name:** Give your sampling a name. A new tag will be created under this name.

After confirming your settings, a worker will start processing your request. Once
it's done, you can find the new tag with the selected images under your tags. If
you select it, you can see how the scatter plot now shows selected images and 
discarded images in a different color. Play around with the different samplers to see
differences between the results.

"""
