.. _lightly-at-a-glance:

Self-supervised learning
========================

Lightly is a computer vision framework for training deep learning models using self-supervised learning.
The framework can be used for a wide range of useful applications such as finding the nearest 
neighbors, similarity search, transfer learning, or data analytics.

Additionally, you can use the Lightly framework to directly interact with the `lightly platform <https://www.lightly.ai>`_.
Check out our section on :ref:`lightly-platform` for more information.


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

.. note:: You can also use a custom PyTorch `Dataset` instead of the 
          `LightlyDataset`. Just make sure your `Dataset` implementation returns
          a tuple of **(sample, target, filename)** to support the basic functions
          for training models. See :py:class:`lightly.data.dataset`
          for more information.


Head to the next section to see how you can train a ResNet on the data you just prepared.

Model, Loss and Training
^^^^^^^^^^^^^^^^^^^^^^^^

Now, we need an embedding model, an optimizer and a loss function. We use a ResNet together
with the normalized temperature-scaled cross entropy loss and simple stochastic gradient descent.

.. code-block:: python

    import torchvision

    from lightly.loss import NTXentLoss
    from lightly.models.modules.heads import SimCLRProjectionHead

    # use a resnet backbone
    resnet = torchvision.models.resnet18()
    resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

    # build a SimCLR model
    class SimCLR(torch.nn.Module):
        def __init__(self, backbone, hidden_dim, out_dim):
            super().__init__()
            self.backbone = backbone
            self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim)

        def forward(self, x):
            h = self.backbone(x).flatten(start_dim=1)
            z = self.projection_head(h)
            return z

    model = SimCLR(resnet, hidden_dim=512, out_dim=128)

    # use a criterion for self-supervised learning
    # (normalized temperature-scaled cross entropy loss)
    criterion = NTXentLoss(temperature=0.5)

    # get a PyTorch optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-0, weight_decay=1e-5)


.. note:: You can also use custom backbones and use lightly to train them using
          self-supervised learning. Learn more about how to use custom backbones
          in our 
          `colab playground <https://colab.research.google.com/drive/1ubepXnpANiWOSmq80e-mqAxjLx53m-zu?usp=sharing>`_.


Train the model for 10 epochs.

.. code-block:: python

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_epochs = 10
    for epoch in range(max_epochs):
        for (x0, x1), _, _ in dataloader:

            x0 = x0.to(device)
            x1 = x1.to(device)

            z0 = model(x0)
            z1 = model(x1)

            loss = criterion(z0, z1)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()


Congrats, you just trained your first model using self-supervised learning!

You can of course also use `PyTorch Lightning <https://www.pytorchlightning.ai/>`_ to implement and train your model.

.. code-block:: python

    import pytorch_lightning as pl

    class SimCLR(pl.LightningModule):
        def __init__(self, backbone, hidden_dim, out_dim):
            super().__init__()
            self.backbone = backbone
            self.projection_head = SimCLRProjectionHead(hidden_dim, hidden_dim, out_dim)
            self.criterion = NTXentLoss(temperature=0.5)

        def forward(self, x):
            h = self.backbone(x).flatten(start_dim=1)
            z = self.projection_head(h)
            return z

        def training_step(self, batch, batch_idx):
            (x0, x1), _, _ = batch
            z0 = self.forward(x0)
            z1 = self.forward(x1)
            loss = self.criterion(z0, z1)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.SGD(self.parameters(), lr=1e-0)
            return optimizer

    model = SimCLR(resnet, hidden_dim=512, out_dim=128)
    gpus = 1 if torch.cuda.is_available() else None
    trainer = pl.Trainer(max_epochs=max_epochs, gpus=gpus)
    trainer.fit(
        model,
        dataloader
    )

To train on a machine with multiple GPUs we recommend using the 
`distributed data parallel` backend.

.. code-block:: python

    # if we have a machine with 4 GPUs we set gpus=4
    trainer = pl.Trainer(
        max_epochs=max_epochs, 
        gpus=4, 
        distributed_backend='ddp'
    )
    trainer.fit(
        model,
        dataloader
    )

Embeddings
^^^^^^^^^^
You can use the trained model to embed your images or even access the embedding
model directly.

.. code-block:: python 

    # make a new dataloader without the transformations
    # The only transformation needed is to make a torch tensor out of the PIL image
    dataset.transform = torchvision.transforms.ToTensor()
    dataloader = torch.utils.data.DataLoader(
        dataset,        # use the same dataset as before
        batch_size=1,   # we can use batch size 1 for inference
        shuffle=False,  # don't shuffle your data during inference
    )

    # embed your image dataset
    embeddings = []
    model.eval()
    with torch.no_grad():
        for img, label, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)

        embeddings = torch.cat(embeddings, 0)

Done! You can continue to use the embeddings to find nearest neighbors or do similarity search.
Furthermore, the ResNet backbone can be used for transfer and few-shot learning.

.. code-block:: python

    # access the ResNet backbone
    resnet = model.backbone

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

    from lightly.core import train_embedding_model, embed_images

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
