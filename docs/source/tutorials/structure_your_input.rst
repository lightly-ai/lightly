.. _input-structure-label:

Tutorial 1: Structure Your Input
================================

The modern-day open-source ecosystem has changed a lot over the years, and there are now
many viable options for data pipelining. For experimentation purposes,
the `torchvision.data <https://pytorch.org/vision/main/datasets.html>`_ submodule provides a robust implementation for most use cases,
and the `HuggingFace Hub <https://hf.co>`_ has emerged as a growing collection of datasets that span a variety of domains and tasks.
When you have your own data in memory, the ability to quickly create datasets and dataloaders is of prime importance.

In this blog post, we will provide a brief overview of `LightlyDataset <https://docs.lightly.ai/self-supervised-learning/lightly.data.html#lightly.data.dataset.LightlyDataset>`_
and go throught examples of using datasets from various open-source libraries such as PyTorch and
Huggingface with the Lightly SSL open-source package. We will also look into how we can create dataloaders
for video tasks while incorporating weak labels.


Native LightlyDataset format
----------------------------

The LightlyDataset class aims to provide a uniform data interface for all models and functions in the Lightly SSL package.
It allows us to create both Image and Video dataset classes with or without labels.

Supported File Types
^^^^^^^^^^^^^^^^^^^^

Since Lightly SSL uses `Pillow <https://github.com/python-pillow/Pillow>`_
for image loading, it also supports all the image formats supported by
`Pillow <https://github.com/python-pillow/Pillow>`_.

- .jpg, .png, .tiff and 
  `many more <https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html>`_

To load videos directly, Lightly SSL uses
`torchvision <https://github.com/pytorch/vision>`_ and
`PyAV <https://github.com/PyAV-Org/PyAV>`_. The following formats are supported.

- .mov, .mp4 and .avi

Unlabelled Image Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^

Creating an Unlabelled Image Dataset is the simplest and the most common use case.

Assuming all images are in a single directory, you can simply pass in the path to the directory containing all the images, viz.

.. code-block:: python

    from lightly.data import LightlyDataset

    dataset = LightlyDataset(input_dir='image_dir/')

.. note::

    Internally each image will get assigned a default label of 0

Labeled Image Datasets
^^^^^^^^^^^^^^^^^^^^^^

If you have (weak) labels for your images and if each label has its directory,
then you can simply pass in the path to the parent directory,
and the dataset class will assign the labels automatically, viz.

.. code-block:: bash

    # directory with subdirectories containing all images
    data/
    +-- weak-label-1/
        +-- img-1.jpg
        +-- img-2.jpg
        ...
        +-- img-N1.jpg
    +-- weak-label-2/
        +-- img-1.jpg
        +-- img-2.jpg
        ...
        +-- img-N2.jpg
    ...
    ...
    ...
    +-- weak-label-10/
        +-- img-1.jpg
        +-- img-2.jpg
        ...
        +-- img-N10.jpg

.. code-block:: python

   from lightly.data import LightlyDataset

   labeled_dataset = LightlyDataset(input_dir='labeled_images_dir/')

Video Datasets
^^^^^^^^^^^^^^

The Lightly SSL package also has native support for videos (`.mov`, `.mp4`, and `.avi` file extensions are supported),
without having to exctract the frames first. This can save a lot of disk space as video files are
typically strongly compressed. Like the aforementioned cases if your videos are in a flat directory or with labels in subdirectories,
you can simply pass the path into the LightlyDataset constructor.

An example for an input directory with videos could look like this:

.. code-block:: bash

    data/
    +-- my_video_1.mov
    +-- my_video_2.mp4
    +-- subdir/
        +-- my_video_3.avi
        +-- my_video_4.avi

.. code-block:: python

   from lightly.data import LightlyDataset

   video_dataset = LightlyDataset(input_dir='video_dir/')

We assign a weak label to each video.

.. note::

   To use video-specific features of Lightly SSL download the necessary extra dependencies `pip install lightly"[video]"`. Also,
   Randomly accessing video frames is slower compared to accessing the extracted frames on disk. However,
   by working directly on video files, one can save a lot of disk space because the frames do not have to
   be extracted beforehand.

PyTorch Datasets
----------------

The Lightly SSL package has helper functions that allow us to use any dataset from PyTorch or torchvision directly within the ecosystem. For example,
to use the CIFAR10 dataset from torchvision we can simply use the
`from_torch_dataset() <https://docs.lightly.ai/self-supervised-learning/lightly.data.html#lightly.data.dataset.LightlyDataset.from_torch_dataset>`_
helper method of the `LightlyDataset <https://docs.lightly.ai/self-supervised-learning/lightly.data.html#lightly.data.dataset.LightlyDataset>`_ class.

.. code-block:: python

   import torchvision
   from lightly.data import LightlyDataset

   base = torchvision.datasets.CIFAR10(root = "./data", download = True)
   torch_dataset = LightlyDataset.from_torch_dataset(base)

HuggingFace Datasets
--------------------

To use a dataset from the huggingface hub 🤗, we can simply apply the desired transformations using the 
`set_transform <https://huggingface.co/docs/datasets/v2.20.0/en/package_reference/main_classes#datasets.Dataset.set_transform>`_
helper method and then create a native PyTorch dataloader.


.. code-block:: python

    import torch
    from datasets import load_dataset
    from lightly.transforms import SimCLRTransform

    base = load_dataset("uoft-cs/cifar10", trust_remote_code=True)

    ## Use pre-defined set of transformations from Lightly SSL
    transform = SimCLRTransform()

    def apply_transform(example_batch, transform=transform):
        """
        Apply the given transform across a batch. To be used in a 'map' like manner

        Args:
          example_batch (Dict): a batch of data, should contain the key 'image'
          tranform (Callable): image transformations to be performed

        Returns:
          updated batch with transformations applied to the image
        """

        assert "image" in example_batch.keys(), "batch should be of type Dict[str, Any] with a key 'image'"

        example_batch["image"] = [
            transform(image.convert("RGB")) for image in example_batch["image"]
        ]
        return example_batch

    base.set_transform(apply_transform)

    hf_dataloader = torch.utils.data.DataLoader(base["train"])

Image Augmentations
-------------------

Many SSL methods leverage image augmentations to better learn invariances in the training process. For example,
by using different crops of a given images and positive view, the SSL model will be trained to produce a representation
that is invariant to these different crops. When using a grayscale operation, or a colorjitter one as positive views,
the representation will have to be invariant to the color information [1]_.

We can use off the shelf augmentations from libraries like `torchvision transforms <https://pytorch.org/vision/stable/transforms.html>`_
and `albumentations <https://albumentations.ai/docs/>`_ or the ones offered by Lightly SSL's
`transforms <https://docs.lightly.ai/self-supervised-learning/lightly.transforms.html>`_ submodule while creating our datasets, viz.

.. code-block:: python

    import albumentations as A
    import torchvision.transforms as T
    from albumentations.pytorch import ToTensorV2

    ## Torchvision Transforms
    torchvision_transforms = T.Compose(
        [
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ]
    )

    ## Albumentation Transforms
    albumentation_transforms = A.Compose(
        [
            A.CenterCrop(height=128, width=128),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    ## Lightly Transforms
    lightly_transform = SimCLRTransform()

.. note::

   You can also create your own augmentations, for more details please refer to :ref:`lightly-custom-augmentation-5` 

Using Transforms for LightlyDataset
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can use any of these transforms directly while creating a `LightlyDataset` as follows:

.. code-block:: python

    ## Applying Augmentations to a Unlabelled Images Dataset
    torchvision_aug_image_dataset = LightlyDataset(input_dir='image_dir/', transform=torchvision_transforms)
    albumentations_aug_image_dataset = LightlyDataset(input_dir='image_dir/', transform=albumentation_transforms)
    lightly_aug_image_dataset = LightlyDataset(input_dir='image_dir/', transform=lightly_transforms)

    ## Similarly for other data formats (Labeled Image Datasets and Video Datasets)

Using Transforms for PyTorch Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Similar to native `LightlyDataset`'s we can also pass the transforms while constructing a `LightlyDataset` from a PyTorch Dataset, viz.

.. code-block:: python

    import torchvision

    base = torchvision.datasets.CIFAR10(root="data/torchvision/", download=True)

    torchvision_aug_dataset = LightlyDataset.from_torch_dataset(
        base, transform=torchvision_transforms
    )
    albumentation_aug_dataset = LightlyDataset.from_torch_dataset(
        base, transform=albumentation_transforms
    )
    lightly_aug_dataset = LightlyDataset.from_torch_dataset(
        base, transform=lightly_transform
    )

Conclusion
----------

In this blogpost, we went through examples of using various open-source packages to create datasets and dataloaders with Lightly SSL,
and how they can be used in a training pipeline. We saw how the Lightly SSL package is flexible enough to work with all major data sources,
and how we can write training pipelines that work with any format.

Now that we are are familiar with creating datasets and dataloaders, lets'
jump right into training a model:

- :ref:`lightly-moco-tutorial-2`
- :ref:`lightly-simclr-tutorial-3`
- :ref:`lightly-simsiam-tutorial-4`
- :ref:`lightly-custom-augmentation-5`
- :ref:`lightly-detectron-tutorial-6`

If you are looking for a use case that's not covered by the above tutorials please
let us know by `creating an issue <https://github.com/lightly-ai/lightly/issues/new>`_
for it.

.. [1] Section 3.1, Role of Data Augmentation. A Cookbook of Self-Supervised Learning (arXiv:2304.12210)

