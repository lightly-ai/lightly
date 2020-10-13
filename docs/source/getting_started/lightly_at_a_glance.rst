Lightly at a glance
===================

Lightly is a computer vision framework for training deep learning models using self-supervised learning.
The framework can be used for a wide range of useful applications such as finding the nearest 
neighbors, similarity search, transfer learning, or data analytics.

Additionally, you can use the Lightly framework to directly interact with the `lightly platform <https://www.lightly.ai>`_.

Walk-through of an example using Lightly
----------------------------------------
In this short example, we will train a model using self-supervision and use it to 
create embeddings.

.. code-block:: python

    from lightly import train_embedding_model
    from lightly import embed_images

    # first we train our model for 1 epoch using a folder of cat images 'cats'
    checkpoint = train_embedding_model(input_dir='cats', trainer={'max_epochs': 1})

    # let's embed our 'cats' dataset using our trained model
    embeddings, labels, filenames = embed_images(input_dir='cats', checkpoint=checkpoint)

    # now, let's inspect the shape of our embeddings
    print(embeddings.shape)

Congrats, you just trained your first model using self-supervised learning!

.. note::
    Self-supervised learning does not require labels for a model to be trained on. Lightly,
    however, supports the use of additional labels. For example, if you train a model
    on a folder 'cats' with subfolders 'Maine Coon', 'Bengal' and 'British Shorthair'
    Lightly automatically returns the enumerated labels as a list.

What's next?
------------
Get started by :ref:`rst-installing` and follow through the tutorial to learn how to get the most out of using Lightly
