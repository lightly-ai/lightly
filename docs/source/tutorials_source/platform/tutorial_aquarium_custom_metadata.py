"""

.. note:: 
    In many real world applications of machine learning datasets are heavily biased.
    This is a direct consequence of the fact that the data is sampled from a distribution
    which itself is biased as well. In this tutorial, we show how such dataset biases
    can be mitigated through clever rebalancing.


Tutorial 5: Custom Metadata and Rebalancing
=============================================

The addition of custom metadata to your dataset can lead to valuable insights which
in turn can drastically improve your machine learning models. Here, you will learn
how to upload custom metadata to your dataset, gain insights about the data, and 
how to curate the data for better model training.


What you will learn
--------------------

* How to upload custom metadata to your datasets.
* How to detect and mitigate biases in your dataset through rebalancing.


Requirements
-------------

You can use your own dataset or the one we provide for this tutorial. The dataset
we will use is the training set of `Roboflow's Aquarium Dataset <https://public.roboflow.com/object-detection/aquarium>`_.
It consists of 448 images from two aquariums in the United States and was made for object detection.
You can download it here :download:`Aquarium.zip <../../../_data/Aquarium.zip>`.

For this tutorial the `lightly` pip package needs to be installed:

.. code-block:: bash

    # Install lightly as a pip package
    pip install lightly


Custom Metadata
-----------------
-> talk about how metadata is often available (with examples)

-> in this tutorial we will extract the metadata from the annotations

-> look at statistics from the Roboflow website: strong imbalance (x23 from least to most common)

-> we want to counteract this imbalance

-> show how to extract and save in correct structure (code below)


.. code-block:: python

    import json
    from lightly.utils import save_custom_metadata

    PATH_TO_COCO_ANNOTATIONS = 'PATH/TO/AQUARIUM/_annotations.cooo.json'
    OUTPUT_FILE = 'my_custom_metadata.json'


    # read coco annotations
    with open(PATH_TO_COCO_ANNOTATIONS, 'r')  as f:
        coco_annotations = json.load(f)
        annotations = coco_annotations['annotations']
        categories = coco_annotations['categories']
        images = coco_annotations['images']


    # create a mapping from category id to category name
    category_id_to_category_name = {}
    for category in categories:
        category_id_to_category_name[category['id']] = category['name']


    # create a list of pairs of (filename, metadata)
    custom_metadata = []
    for image in images:

        # we want to count the number of instances of every class on the image
        metadata = {'number_of_instances': {}}
        for category in categories:
            metadata['number_of_instances'][category['name']] = 0

        # count all annotations for that image
        for annotation in annotations:
            if annotation['image_id'] == image['id']:
                metadata['number_of_instances'][category_id_to_category_name[annotation['category_id']]] += 1

        # append (filename, metadata) to the list
        custom_metadata.append((image['file_name'], metadata))


    # save custom metadata in the correct json format
    save_custom_metadata(OUTPUT_FILE, custom_metadata)


Upload the custom dataset along with your dataset:

.. code-block:: bash

    lightly-magic input_dir="./aquarium" trainer.max_epochs=0 token=YOUR_TOKEN new_dataset_name="Aquarium" custom_metadata=YOUR_CUSTOM_METADATA_FILE.json


Note that if you already have a dataset on the Lightly platform, you can add custom metadata with

.. code-block::bash

    lightly-upload token=YOUR_TOKEN dataset_id=YOUR_DATASET_ID custom_metadata=YOUR_CUSTOM_METADATA_FILE.json


Configuration
---------------
-> configure a custom metadata which shows the number of fish on an image

-> explain how to confirm configuration is correct (sort-by & plot)


Rebalancing
----------------
-> subsample the "many fish cluster" to reduce the number of fish in the dataset

-> result should be a rebalanced dataset

"""
