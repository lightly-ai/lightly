.. _lightly-command-line-tool:

Command-line tool
=================

The Lightly framework provides you with a command-line interface (CLI) to train 
self-supervised models and create embeddings without having to write a single 
line of code.

Check the installation of lightly
-----------------------------------
To see if the lightly command-line tool was installed correctly, you can run the
following command which will print the installed lightly version:

.. code-block:: bash

    lightly-version

If lightly was installed correctly, you should see something like this:

.. code-block:: bash

    lightly version 1.1.4

Crop Images using Labels or Predictions
---------------------------------------------------
For some tasks, self-supervised learning on an image level has disadvantages. For 
example, when training an object detection model we care about local features
describing the objects rather than features describing the full image.

One simple trick to overcome this limitation, is to use labels or to use a pre-trained model
to get bounding boxes around the objects and then cropping the objects out of the
image.

We can do this using the **lightly-crop** CLI command. The CLI command crops 
objects out of the input images based on labels and copies them into an output folder.
The new folder consists now of the cropped images.

.. code-block:: bash

    # Crop images and set the crop to be 20% around the bounding box
    lightly-crop input_dir=images label_dir=labels output_dir=cropped_images crop_padding=0.2

    # Crop images and use the class names in the filename
    lightly-crop input_dir=images label_dir=labels output_dir=cropped_images \
                 label_names_file=data.yaml

The labels should be in the yolo format. For each image you should have a
corresponding .txt file. Each row in the .txt file has the following format:

* class x_center y_center width height

.. code-block:: text

    0 0.23 0.14 0.05 0.04
    1 0.43 0.13 0.12 0.08

An example for the label names .yaml file:

.. code-block:: yaml

    names: [cat, dog]

You can use the output of the lightly-crop command as the *input_dir* for your
lightly-train command.

Download data using the CLI
-----------------------------------------------
You can download a dataset with a given tag from the Lightly Platform using the 
following CLI command. The CLI provides you with three options:

* Download the list of filenames for a given tag in the dataset.
  
* Download the images for a given tag in the dataset.
  
* Copy the images for a given tag from an input directory to a target directory.

The last option allows you to very quickly extract only the images in a given tag
without the need to download them explicitly.

.. code-block:: bash

    # download a list of files
    lightly-download tag_name=my_tag_name dataset_id=your_dataset_id token=your_token

    # download the images and store them in an output directory
    lightly-download tag_name=my_tag_name dataset_id=your_dataset_id token=your_token \
                     output_dir=path/to/output/dir

    # copy images from an input directory to an output directory
    lightly-download tag_name=my_tag_name dataset_id=your_dataset_id token=your_token \
                     input_dir=path/to/input/dir output_dir=path/to/output/dir

