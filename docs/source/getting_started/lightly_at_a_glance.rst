.. _lightly-at-a-glance:

Lightly at a Glance
===================

Lightly is a computer vision framework for training deep learning models using self-supervised learning.
The framework can be used for a wide range of useful applications such as finding the nearest 
neighbors, similarity search, transfer learning, or data analytics.

Additionally, you can use the Lightly framework to directly interact with the `lightly platform <https://www.lightly.ai>`_.

You can install lightly using pip.

.. code-block:: bash

    pip install lightly


How Lightly Works
-----------------
The flexible design of Lightly makes it easy to integrate in your Python code. Lightly is built completely around PyTorch
frameworks and the different pieces can be put together to fit *your* requirements.

Data and Transformations
^^^^^^^^^^^^^^^^^^^^^^^^
The basic building block of self-supervised methods
such as `SimCLR <https://arxiv.org/abs/2002.05709>`_ are image transformations. Each image is transformed into
two new images by randomly applied augmentations. The task of the self-supervised model is then to identify the
images which come from the same original among a set of negative examples.

Lightly implements these transformations
as torchvision transforms in the collate function of the dataloader. For example, the collate
function below will apply two different, randomized transforms to each image: A randomized resized crop and a
random color jitter.

.. code-block:: python

    import lightly.data as data

    # the collate function applies random transforms to the input images
    collate_fn = data.ImageCollateFunction(input_size=32, cj_prob=0.5)

Let's now load an image dataset and create a PyTorch dataloader with the collate function from above.

.. code-block:: python

    import torch

    # create a dataset from your image folder
    dataset = data.LightlyDataset(input_dir='./my/cute/cats/dataset/')

    # build a PyTorch dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,                # pass the dataset to the dataloader
        batch_size=128,         # a large batch size helps with the learning
        shuffle=True,           # shuffling is important!
        collate_fn=collate_fn)  # apply transformations to the input images

Head to the next section to see how you can train a ResNet on the data you just prepared.

Training
^^^^^^^^

Now, we need an embedding model, an optimizer and a loss function. We use a ResNet together
with the normalized temperature-scaled cross entropy loss and simple stochastic gradient descent.

.. code-block:: python

    import torchvision

    import lightly.models as models
    import lightly.loss as loss

    # use a resnet backbone
    resnet = torchvision.models.resnet.resnet18()
    resnet = nn.Sequential(*list(resnet.children())[:-1])

    # build the simclr model
    model = models.SimCLR(resnet, num_ftrs=512)

    # use a criterion for self-supervised learning
    # (normalized temperature-scaled cross entropy loss)
    criterion = loss.NTXentLoss(temperature=0.5)

    # get a PyTorch optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-0, weight_decay=1e-5)

Put everything together in an embedding model and train it for 10 epochs on a single GPU.

.. code-block:: python

    import lightly.embedding as embedding

    # put all the pieces together in a single pytorch_lightning trainable!
    embedding_model = embedding.SelfSupervisedEmbedding(
        model,
        criterion,
        optimizer,
        dataloader)

    # do self-supervised learning for 10 epochs
    embedding_model.train_embedding(gpus=1, max_epochs=10)

Congrats, you just trained your first model using self-supervised learning!

Embeddings
^^^^^^^^^^
You can use the trained model to embed your images or even access the embedding
model directly.

.. code-block:: python 

    # make a new dataloader without the transformations
    dataloader = torch.utils.data.DataLoader(
        dataset,        # use the same dataset as before
        batch_size=1,   # we can use batch size 1 for inference
        shuffle=False,  # don't shuffle your data during inference
    )

    # embed your image dataset
    embeddings, labels, filenames = embedding_model.embed(dataloader)

    # access the ResNet backbone
    resnet = embedding_model.model.backbone

Done! You can continue to use the embeddings to find nearest neighbors or do similarity search.
Furthermore, the ResNet backbone can be used for transfer and few-shot learning.

.. note::
    Self-supervised learning does not require labels for a model to be trained on. Lightly,
    however, supports the use of additional labels. For example, if you train a model
    on a folder 'cats' with subfolders 'Maine Coon', 'Bengal' and 'British Shorthair'
    Lightly automatically returns the enumerated labels as a list.

Lightly in Three Lines
----------------------------------------

Lightly also offers an easy-to-use interface. The following lines show how the package can 
be used to train a model with self-supervision and create embeddings with only three lines
of code.

.. code-block:: python

    from lightly import train_embedding_model, embed_images

    # first we train our model for 10 epochs
    checkpoint = train_embedding_model(input_dir='./my/cute/cats/dataset/', trainer={'max_epochs': 10})

    # let's embed our 'cats' dataset using our trained model
    embeddings, labels, filenames = embed_images(input_dir='./my/cute/cats/dataset/', checkpoint=checkpoint)

    # now, let's inspect the shape of our embeddings
    print(embeddings.shape)


What's next?
------------
Get started by :ref:`rst-installing` and follow through the tutorials to 
learn how to get the most out of using Lightly:

Tutorials:

- :ref:`input-structure-label`
- :ref:`lightly-moco-tutorial-2`
- :ref:`lightly-simclr-tutorial-3`  
- :ref:`lightly-simsiam-tutorial-4`  
- :ref:`lightly-custom-augmentation-5` 
