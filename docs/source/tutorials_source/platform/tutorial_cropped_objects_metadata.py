"""

.. _lightly-tutorial-cropped-objects-metadata

Tutorial 6: Find false negatives of object detection
=============================================

    In object detection applications, it can happen that the detector does not detect an object
    because it did not see any examples of this or similar object yet.
    This especially a problem in warehouse and retail applications,
    as new products get added to the shelves every day. These new products with new appearence
    often have a low so-called objectness score. Currently, finding these objects false detected
    as non-objects needs a lot of manual labour, especially when having 1000s of new images coming
    in every day. Lightly can make this work easier in a two-step approach:

    1. A human finds one false negative and adds it to the missing examples.
    2. Lightly finds all similar images which are also false negatives.
    Thus it can propose to directly also add them as missing examples.

    If there are e.g. 9 similar images for each missing example found by a human,
    Lightly can speed up the process by a factor of 10.

What you will learn
--------------------

* How to use an object detection model to crop objects out of a full image and save them as images.
* How to save the object detection scores as metadata in a Lightly format.
* How to upload the cropped images with their corresponding metadata to the webapp.
* How to use the webapp to find false negatives and similar examples easily.

Requirements
-------------

You can use your own dataset or the one we provide for this tutorial. The dataset
we will use is the `SKU110k dataset <https://github.com/eg4000/SKU110K_CVPR19>`_ showing store shelves.
The tutorial is computationally expensive if you run it on the full 110k images of products,
thus we recommend running it on a subset of the dataset. E.g. copy 100 images from one folder to
a new folder and use the path to the latter folder as input. Alternatively, run it on your own dataset.

For this tutorial the `lightly` pip package needs to be installed:

.. code-block:: bash

    # Install lightly as a pip package
    pip install lightly.

Steps
-------------

The steps of this tutorial are quite straightforward:
    1. Define the dataset and torch dataloader for your dataset. You need to provide the path to the dataset images.
    2. Define a pretrained object detection model. We use the retina net trained on COCO 2017. As it was not pretrained on a retail dataset, its performance is not state-of-the art. Nonetheless, it is sufficient for this tutorial and very easy to use.
    3. Predict with the model on the dataset.
    4. Use the bounding boxes of the object predictions to crop the objects out of the full images and save them in the output directory.
    5. Extract the objectness scores and save them as custom metadata in a .json file.
    6. Use the lightly-magic command to upload the cropped images, their embeddings and the objectness scores.
    7. In the Lightly Webapp: Configure the objectness score as custom metadata.
    8. In the Lightly Webapp: Sort the images in the explore view by increasing objectness score. This allows to easily find missing examples / false positives and similar images to them.

Computational Expense
-------------
* Step 3 needs to run a model on every image in your dataset. In step 6, when embedding all cropped images, an embedding model needs to be run on every of them. These two are computationally expensive. They run much faster if you have CUDA support.
* In step 4, every detected object needs to be cropped out and saved on the disk.
* In step 6, uploading all cropped images takes a while, approximately 30 images/s can be achieved.

.. code-block:: python

    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from lightly.active_learning.utils import BoundingBox
    from lightly.data import LightlyDataset
    from lightly.utils import save_custom_metadata
    from lightly.utils.cropping.crop_image_by_bounding_boxes import crop_dataset_by_bounding_boxes_and_save

    BASE_PATH = "path/to/dataset/"
    DATASET_PATH = BASE_PATH+"images"  # the path were the full images are found
    OUTPUT_DIR = BASE_PATH+"cropped_images" # the path where the cropped images will be saved
    # the file where the objectness scores will be saved
    METADATA_OUTPUT_FILE = BASE_PATH+"cropped_images_objectness_scores.json"

    ''' 1. Define the dataset and dataloader'''
    x_size = 2048
    y_size = 2048
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((x_size, y_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        ])
    dataset = LightlyDataset(DATASET_PATH, transform=transform)

    ''' 2. Define the pretrained object detection model'''
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)

    ''' 3. Predict with the model on the dataset '''
    model.eval()
    dataloader = DataLoader(dataset, batch_size=2)
    predictions = []
    with torch.no_grad():
        for x, _, _ in tqdm(dataloader):
            pred = model(x)
            predictions.append(pred)

    predictions = [i for sublist in predictions for i in sublist]

    ''' 4. Save the cropped objects '''
    class_indices_list_list = [list(prediction["labels"]) for prediction in predictions]
    bounding_boxes_list_list = []
    for prediction in predictions:
        bounding_boxes_list = []
        for box in prediction["boxes"]:
            x0 = box[0] / x_size
            y0 = box[1] / y_size
            x1 = box[2] / x_size
            y1 = box[3] / y_size
            bounding_boxes_list.append(BoundingBox(x0, y0, x1, y1))
        bounding_boxes_list_list.append(bounding_boxes_list)

    cropped_images_list_list = crop_dataset_by_bounding_boxes_and_save(dataset, OUTPUT_DIR,
        bounding_boxes_list_list, class_indices_list_list)

    '''  5. Save the objectness scores as metadata '''
    objectness_scores_list_list = [list(prediction["scores"]) for prediction in predictions]
    metadata_list = []
    for cropped_images_list, objectness_scores_list in zip(cropped_images_list_list, objectness_scores_list_list):
        for cropped_images_filename, objectness_score in zip(cropped_images_list, objectness_scores_list):
            metadata = {"objectness_score": float(objectness_score)}
            metadata_list.append((cropped_images_filename, metadata))
    save_custom_metadata(METADATA_OUTPUT_FILE, metadata_list)

    ''' 6. Tell the lightly CLI command '''
    cli_command = f"lightly-magic input_dir={OUTPUT_DIR} new_dataset_name=SKU_110k_val_cropped trainer.max_epochs=0 "
    cli_command += f"custom_metadata={METADATA_OUTPUT_FILE} token=MY_TOKEN"
    print(f"Upload the images and custom metadata with the following CLI command:")
    print(cli_command)

6. Adapt this command to include your token and run it in a terminal. It will embed the images
with a pretrained model, create a new dataset in the Lightly webapp and upload the images,
embeddings and metadata to it. You can also change some arguments,
e.g. to train a better embedding model instead of relying on a pretrained one.
For more information, head to :ref:`lightly-command-line-tool`.

.. figure:: ../../tutorials_source/platform/images/tutorial_cropped_objects_metadata/sku110k_lightly_magic.jpg
    :align: center
    :alt: Terminal output of lightly-magic command.

7. Once the cropped images, embeddings and metadata are uploaded, we can use the Lightly Webapp
to configure the Objectness Score as metadata. This is done in the Configuration view.

.. figure:: ../../tutorials_source/platform/images/tutorial_cropped_objects_metadata/sku110k_config_metadata.jpg
    :align: center
    :alt: Configuration of Objectness Score as metadata

8. Now we can switch to the explore view and select to sort by the Objectness Score in ascending order.
We directly see that many images show images despite having a low objectness score,
thus they are false negatives / missing examples.

.. figure:: ../../tutorials_source/platform/images/tutorial_cropped_objects_metadata/sku110k_explore_sort_objectness.jpg
    :align: center
    :alt: Explore view of images sorted by ascending objectness score

When clicking on one them, we wee that they it has a low objectness score of only 0.05, despite showing an object,
thus it is false negative. Similar images which are also false negatives are shown as well.
Thus all of them can be added directly to the list of missing examples,
instead of finding and adding all of them by hand.

.. figure:: ../../tutorials_source/platform/images/tutorial_cropped_objects_metadata/sku110k_find_similar.jpg
    :align: center
    :alt: Detail view of a missing examples together with similar samples


"""