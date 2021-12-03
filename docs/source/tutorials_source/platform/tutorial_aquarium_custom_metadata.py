"""
.. _lightly-tutorial-aquarium-custom-metadata:

.. note:: 
    In many real world applications of machine learning datasets are heavily biased.
    This is a direct consequence of the fact that the data is sampled from a distribution
    which itself is biased as well. In this tutorial, we show how such dataset biases
    can be mitigated through clever rebalancing.


Tutorial 5: Custom Metadata and Rebalancing
=============================================

The addition of custom metadata to your dataset can lead to valuable insights which
in turn can drastically improve your machine learning models. Lightly supports the upload
of categorical (e.g. weather scenario, road type, or copyright license) and numerical metadata (e.g.
instance counts, outside temperature, or vehicle velocity). Here, you will learn how to upload custom metadata 
to your dataset, gain insights about the data, and how to curate the data for better model training.


What you will learn
--------------------

* How to upload custom metadata to your datasets.
* How to detect and mitigate biases in your dataset through rebalancing.


Requirements
-------------

You can use your own dataset or the one we provide for this tutorial. The dataset
we will use is the training set of `Roboflow's Aquarium Dataset <https://public.roboflow.com/object-detection/aquarium>`_.
It consists of 448 images from two aquariums in the United States and was made for object detection.
You can download it here :download:`Aquarium.zip <../../../_data/aquarium.zip>`.

For this tutorial the `lightly` pip package needs to be installed:

.. code-block:: bash

    # Install lightly as a pip package
    pip install lightly


Custom Metadata
-----------------

In many datasets, images come with additional metainformation. For example, in autonomous driving,
it's normal to record not only the image but also the current weather situation, the road type, and
further situation specific information like the presence of vulnerable street users. In our case,
the metadata can be extracted from the annotations.

.. note:: 

    Balancing a dataset by different available metadata is crucial for machine learning.

Let's take a look at the dataset statistics from the `Roboflow website <https://public.roboflow.com/object-detection/aquarium>`_.
The table below contains the category names and instance counts of the dataset

+---------------+----------------+
| Category Name | Instance Count |
+===============+================+
| fish          | 2669           |
+---------------+----------------+
| jellyfish     | 694            |
+---------------+----------------+
| penguin       | 516            |
+---------------+----------------+
| shark         | 354            |
+---------------+----------------+
| puffin        | 284            |
+---------------+----------------+
| stingray      | 184            |
+---------------+----------------+
| starfish      | 116            |
+---------------+----------------+

We can see that the dataset at hand is heavily imbalanced - the category distribution is long tail. 
The most common category is `fish` and objects of that category appear 23 times more often than members 
of the weakest represented category `starfish`. In order to counteract this imbalance, we want to 
**remove redundant images which show a lot of fish** from the dataset.


Let's start by figuring out how many objects of each category are on each image. The following code extracts metadata from COCO annotations and saves them in a 
lightly-friendly format. If you want to skip this step, you can use the provided `_annotations.coco.metadata.json`.

.. note::

    Check out :py:class:`lightly.api.api_workflow_client.ApiWorkflowClient.upload_custom_metadata`
    to learn more about the expected format of the custom metadata json file.


.. note:: You can save your own custom metadata with :py:class:`lightly.utils.io.save_custom_metadata`.

.. code-block:: python

    import json
    from lightly.utils import save_custom_metadata

    PATH_TO_COCO_ANNOTATIONS = './aquarium/_annotations.coco.json'
    OUTPUT_FILE = '_annotations.coco.metadata.json'

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

        # we want to count the number of instances of every category on the image
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


Now that we have extracted and saved the custom metadata in the correct format,
we can upload the dataset to the `Lightly web-app <https://app.lightly.ai>`_. 
Don't forget to replace the variables `YOUR_TOKEN`, and `YOUR_CUSTOM_METADATA_FILE.json` in the command below.

.. code-block:: bash

    lightly-magic input_dir="./aquarium" trainer.max_epochs=0 token=YOUR_TOKEN new_dataset_name="Aquarium" custom_metadata=YOUR_CUSTOM_METADATA_FILE.json


Note that if you already have a dataset on the Lightly platform, you can add custom metadata with:

.. code-block:: bash

    lightly-upload token=YOUR_TOKEN dataset_id=YOUR_DATASET_ID custom_metadata=YOUR_CUSTOM_METADATA_FILE.json


Configuration
---------------
In order to use custom metadata in the web-app, it needs to be configured first. For this, head to your dataset and 
select the `Settings > Custom Metadata` menu. We name the configuration `Number of Fish` since we're interested in the
number of fish on each image. Next, we fill in the `Path`: To do so, we click on it and select `number_of_instances.fish`.
Finally, we pick `Integer` as a data type. The configured custom metadata should look like this:

.. figure:: ../../tutorials_source/platform/images/custom_metadata_configuration_aquarium.png
   :align: center
   :alt: Custom metadata configuration for number of fish

   Completed configuration for the custom metadata "Number of Fish".


To verify the configuration works correctly, we can head to the `Explore` screen and sort our dataset by the `Number of Fish`.
If we sort in descending order we see that images with many fish are shown first. This verifies that the metadata is configured properly.

Next, we head to the `Embedding` page. On the top right, we select `Color by property` and set it to `Number of Fish`. This will highlight
clusters of similar images with many fish on them:

.. figure:: ../../tutorials_source/platform/images/custom_metadata_scatter_plot.png
   :align: center
   :alt: Custom metadata scatter plot

   Scatter plot highlighting clusters of similar images with many fish on them (light green).


Rebalancing
----------------
As we have seen earlier in this tutorial, the aquarium dataset is heavily imbalanced. The scatter plot also tells us that the bulk of the 
fish instances in the dataset comes from a few, similar images. We want to rebalance the dataset by only keeping a diverse subset of these
images. For this, we start by creating a new tag consisting only of images with a lot of fish in them. We do so, by shift-clicking in the
scatter plot and drawing a circle around the clusters of interest. We call the tag `Fish`.

.. figure:: ../../tutorials_source/platform/images/custom_metadata_scatter_plot_with_tag.png
    :align: center
    :alt: Custom metadata scatter plot (fish tag)

    Scatter plot of the tag we named "Fish".

Now, we can use CORESET sampling to diversify this tag and reduce the number of images in it (see `Tutorial 2: Diversify the Sunflowers Dataset` 
if you're not familiar with subsampling). We use CORESET to create the tag `FewerFish` with only 10 remaining images.

Lastly, all we need to do in order to get the balanced dataset is merge the `FewerFish` tag with the remainder of the dataset. For this we can use
`tag arithmetics`:

1. We open the tag tree by clicking on the three dots in the navigation bar and unravel it.

2. We create a new tag which is the inverse of the `Fish` tag. We start by left-clicking on `initial-tag` and shift-clicking on the `Fish` tag. The tag arithmetics menu should pop up. We select `difference`, put in the name `NoFish` and hit enter.

3. We create the final dataset by taking the union of the `NoFish` and the `FewerFish` tag.

In the end, the tag tree should look like this:

.. figure:: ../../tutorials_source/platform/images/final_tag_tree.png
    :align: center
    :alt: Final tag tree.

    The final tag tree with the tags `Fish`, `FewerFish`, `NoFish`, and `final-tag`.

Now we can use the balanced dataset to train our machine learning model. We can easily download the data from the `Download` page.

"""
