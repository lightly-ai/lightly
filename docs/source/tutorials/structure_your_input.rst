.. _input-structure-label:

Tutorial 1: Structure Your Input
================================

The `lightly Python package <https://pypi.org/project/lightly/>`_ can process image datasets to generate embeddings 
or to upload data to the `Lightly platform <https://app.lightly.ai>`_. In this tutorial you will learn how to structure
your image dataset such that it is understood by our framework.

You can also skip this tutorial and jump right into training a model:

- :ref:`lightly-moco-tutorial-2`
- :ref:`lightly-simclr-tutorial-3`  

Supported File Types
--------------------

Images
^^^^^^^^^^^^^^^^^^^^^

Since lightly uses `Pillow <https://github.com/python-pillow/Pillow>`_ 
for image loading it also supports all the image formats supported by 
`Pillow <https://github.com/python-pillow/Pillow>`_.

- .jpg, .png, .tiff and 
  `many more <https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html>`_

Videos
^^^^^^^^^^^^^^^^^^^^^

To load videos directly lightly uses 
`torchvision <https://github.com/pytorch/vision>`_ and 
`PyAV <https://github.com/PyAV-Org/PyAV>`_. The following formats are supported.

- .mov, .mp4 and .avi



Image Folder Datasets
---------------------

Image folder datasets contain raw images and are typically specified with the `input_dir` key-word.


Flat Directory Containing Images
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can store all images of interest in a single folder without additional hierarchy. For example below,
lightly will load all filenames and images in the directory `data/`. Additionally, it will assign all images
a placeholder label.

.. code-block:: bash

    # a single directory containing all images
    data/
    +--- img-1.jpg
    +--- img-2.jpg
    ...
    +--- img-N.jpg

For the structure above, lightly will understand the input as follows:

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
the filenames loaded by lightly are with respect to the "root directory" `data/`. Furthermore, lightly assigns
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

For the structure above, lightly will understand the input as follows:

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
        10,
    ]

Video Folder Datasets
---------------------
The lightly Python package allows you to work `directly` on video data, without having
to exctract the frames first. This can save a lot of disc space as video files are
typically strongly compressed. Using lightly on video data is as simple as pointing 
the software at an input directory where one or more videos are stored. The package will
automatically detect all video files and index them so that each frame can be accessed.

An example for an input directory with videos could look like this:

.. code-block:: bash

    data/
    +-- my_video_1.mov
    +-- my_video_2.mp4
    +-- my_video_3.avi

We assign a weak label to each video.
To upload the three videos from above to the platform, you can use

.. code-block:: bash

    lightly-upload token='123' dataset_id='XYZ' input_dir='data/'

All other operations (like training a self-supervised model and embedding the frames individually)
also work on video data. Give it a try! 

.. note::

    Randomly accessing video frames is slower compared to accessing the extracted frames on disc. However,
    by working directly on video files, one can save a lot of disc space because the frames do not have to 
    be exctracted beforehand.

Torchvision Datasets
--------------------

Lightly also supports a direct interface to some of the `torchvision datasets <https://pytorch.org/docs/stable/torchvision/datasets.html>`_.
From the command-line interface, they can easily be specified with the `data` and `root` keyowords. The following torchvision
datasets are currently supported by the lightly Python package:

* cifar10
* cifar100
* cityscapes
* stl10
* voc07-det
* voc12-det
* voc07-seg
* voc12-seg

For example, the following command downloads the cifar10 datasets and generates embeddings for all images:

.. code-block:: bash

    lightly-embed data='cifar10' root='./'


Embedding Files
---------------

Embeddings generated by the lightly Python package are typically stored in a `.csv` file and can then be uploaded to the 
Lightly platform from the command line. If the embeddings were generated with the lightly command-line tool, they have  
the correct format already.

You can also save your own embeddings in a `.csv` file to upload them. In that case, make sure the file meets the format 
requirements: Use the `save_embeddings` function from `lightly.utils.io` to convert your embeddings, weak-labels, and 
filenames to the right shape.

.. code-block:: python

    import lightly.utils.io as io

    # embeddings:
    # embeddings are stored as an n_samples x dim numpy array
    embeddings = np.array([[0.1, 0.5],
                           [0.2, 0.2],
                           [0.1, 0.9],
                           [0.3, 0.2]])
    
    # weak-labels
    # a list of integers carrying meta-information about the images
    labels = [0, 1, 1, 0]

    # filenames
    # list of strings containing the filenames of the images w.r.t the input directory
    filenames = [
        'weak-label-0/img-1.jpg',
        'weak-label-1/img-1.jpg',
        'weak-label-1/img-2.jpg',
        'weak-label-0/img-2.jpg',
    ]

    io.save_embeddings('my_embeddings_file.csv', embeddings, labels, filenames)

The code shown above will produce the following `.csv` file:

.. list-table:: my_embeddings_file.csv
   :widths: 50 50 50 50
   :header-rows: 1

   * - filenames
     - embedding_0
     - embedding_1
     - labels
   * - weak-label-0/img-1.jpg
     - 0.1
     - 0.5
     - 0
   * - weak-label-1/img-1.jpg
     - 0.2
     - 0.2
     - 1
   * - weak-label-1/img-2.jpg
     - 0.1
     - 0.9
     - 1
   * - weak-label-0/img-2.jpg
     - 0.3
     - 0.2
     - 0

.. note:: Note that lightly automatically creates "weak" labels for datasets
          with subfolders. Each subfolder corresponds to one weak label.
          The labels are called "weak" since they might not be used for a task
          you want to solve with ML directly but still can be relevant to group
          the data into buckets.


Advanced usage of Embeddings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In some cases you want to enrich the embeddings with additional information.
The lightly csv scheme is very simple and can be easily extended.
For example, you can add your own embeddings to the existing embeddings. This 
could be useful if you have additional meta information about each sample.

.. _lightly-custom-labels:

Add Custom Embeddings
""""""""""""""""""""""""""""""

To add custom embeddings you need to add mre embedding columns to the .csv file.
Make sure you keep the enumeration of the embeddings in correct order.


Here you see an embedding from lightly with a 2-dimensional embedding vector.

.. list-table:: lightly_embeddings.csv
   :widths: 50 50 50 50
   :header-rows: 1

   * - filenames
     - embedding_0
     - embedding_1
     - labels
   * - img-1.jpg
     - 0.1
     - 0.5
     - 0
   * - img-2.jpg
     - 0.2
     - 0.2
     - 0
   * - img-3.jpg
     - 0.1
     - 0.9
     - 1

We can now append our embedding vector to the .csv file.

.. list-table:: lightly_with_custom_embeddings.csv
   :widths: 50 50 50 50 50 50
   :header-rows: 1

   * - filenames
     - embedding_0
     - embedding_1
     - embedding_2
     - embedding_3
     - labels
   * - img-1.jpg
     - 0.1
     - 0.5
     - 0.2
     - 0.7
     - 0
   * - img-2.jpg
     - 0.2
     - -0.2
     - 1.1
     - -0.4
     - 0
   * - img-3.jpg
     - 0.1
     - 0.9
     - -0.2
     - 0.5
     - 1

.. note:: The embedding columns must be grouped together. You can not have
          another column between two embedding columns.


Next Steps
-----------------

Now that you understand the various data formats lightly supports you can 
start training a model:

- :ref:`lightly-moco-tutorial-2`
- :ref:`lightly-simclr-tutorial-3`  