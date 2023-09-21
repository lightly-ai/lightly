.. _input-structure-label:

Tutorial 1: Structure Your Input
================================

If you are familiar with torch-like image dataset, you can skip this tutorial and
jump right into training a model:

- :ref:`lightly-moco-tutorial-2`
- :ref:`lightly-simclr-tutorial-3`  
- :ref:`lightly-simsiam-tutorial-4`
- :ref:`lightly-custom-augmentation-5`
- :ref:`lightly-detectron-tutorial-6`

If you are looking for a use case that's not covered by the above tutorials please
let us know by `creating an issue <https://github.com/lightly-ai/lightly/issues/new>`_
for it.


Supported File Types
--------------------

By default, the `Lightly SSL Python package <https://pypi.org/project/lightly/>`_ 
can process images or videos for self-supervised learning or for generating embeddings.
You can always write your own torch-like dataset to use other file types.

Images
^^^^^^^^^^^^^^^^^^^^^

Since Lightly SSL uses `Pillow <https://github.com/python-pillow/Pillow>`_ 
for image loading, it also supports all the image formats supported by 
`Pillow <https://github.com/python-pillow/Pillow>`_.

- .jpg, .png, .tiff and 
  `many more <https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html>`_

Videos
^^^^^^^^^^^^^^^^^^^^^

To load videos directly, Lightly SSL uses 
`torchvision <https://github.com/pytorch/vision>`_ and 
`PyAV <https://github.com/PyAV-Org/PyAV>`_. The following formats are supported.

- .mov, .mp4 and .avi


Image Folder Datasets
---------------------

Image folder datasets contain raw images and are typically specified with the `input_dir` key-word.


Flat Directory Containing Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can store all images of interest in a single folder without additional hierarchy. For example below,
Lightly SSL will load all filenames and images in the directory `data/`. Additionally, it will assign all images
a placeholder label.

.. code-block:: bash

    # a single directory containing all images
    data/
    +--- img-1.jpg
    +--- img-2.jpg
    ...
    +--- img-N.jpg

For the structure above, Lightly SSL will understand the input as follows:

.. code-block:: python

    filenames = [
        'img-1.jpg',
        'img-2.jpg',
        ...
        'img-N.jpg',
    ]

    labels = [
        0,
        0,
        ...
        0,
    ]

Directory with Subdirectories Containing Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can give structure to your input directory by collecting the input images in subdirectories. In this case,
the filenames loaded by Lightly SSL are with respect to the "root directory" `data/`. Furthermore, Lightly SSL assigns
each image a so-called "weak-label" indicating to which subdirectory it belongs.

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

For the structure above, Lightly SSL will understand the input as follows:

.. code-block:: python

    filenames = [
        'weak-label-1/img-1.jpg',
        'weak-label-1/img-2.jpg',
        ...
        'weak-label-1/img-N1.jpg',
        'weak-label-2/img-1.jpg',
        ...
        'weak-label-2/img-N2.jpg',
        ...
        'weak-label-10/img-N10.jpg',
    ]

    labels = [
        0,
        0,
        ...
        0,
        1,
        ...
        1,
        ...
        9,
    ]

Video Folder Datasets
---------------------
The Lightly SSL Python package allows you to work `directly` on video data, without having
to exctract the frames first. This can save a lot of disk space as video files are
typically strongly compressed. Using Lightly SSL on video data is as simple as pointing 
the software at an input directory where one or more videos are stored. The package will
automatically detect all video files and index them so that each frame can be accessed.

An example for an input directory with videos could look like this:

.. code-block:: bash

    data/
    +-- my_video_1.mov
    +-- my_video_2.mp4
    +-- subdir/
        +-- my_video_3.avi
        +-- my_video_4.avi

We assign a weak label to each video.


.. note::

    Randomly accessing video frames is slower compared to accessing the extracted frames on disk. However,
    by working directly on video files, one can save a lot of disk space because the frames do not have to
    be extracted beforehand.
