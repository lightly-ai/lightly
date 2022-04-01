.. _ref-docker-datasource-predictions:

Add Predictions to a datasource
===============================

Lightly can not only use images you provided in a datasource, but also predictions of a ML model on your images.
They are used for active learning for selecting images based on the objects in them.
Furthermore, object detection predictions can be used running Lightly on object level.
By providing the predictions in the datasource,
you have full control over them and they scale well to millions of samples.
Furthermore, if you add new samples to your datasource, you can simultaneously
add their predictions to the datasource.
If you already have labels instead of predictions, you can treat them
just as predictions and upload them the same way.


Predictions Folder Structure
----------------------------

In the following, we will outline the format of the predictions required by the
Lightly docker. Everything regarding predictions will take place in a subdirectory
of your configured datasource called `.lightly/predictions`. The general structure
of this directory will look like this:


.. code-block:: bash

    datasource/my_dataset
        + .lightly/predictions/
            + tasks.json
            + task_1/
                 + schema.json
                 + image_1.json
                 ...
                 + image_N.json
            + task_2/
                 + schema.json
                 + image_1.json
                 ...
                 + image_N.json
    image_1.png
    image_2.png
    ...
    image_N.png


Where each subdirectory corresponds to one prediction task (e.g. a classification task
and an object detection task). All of the files are explained in the next sections.


Prediction Tasks
----------------
To let Lightly know what kind of prediction tasks you want to work with, Lightly
needs to know their names. It's very easy to let Lightly know which tasks exist:
simply add a `tasks.json`` in your storage bucket stored at the subdirectory `.lightly/predictions/`.

The `tasks.json` file must include a list of your task names which must match name
of the subdirectory where your prediction schemas will be located.

.. note::

    Only the task names listed within `tasks.json`` will be considered.
    Please ensure that the task name corresponds with the location of your prediction schema.
    This allows you to specify which subfolder are considered by the Lightly docker.

For example, let's say we are working with the following folder structure:

.. code-block:: bash

    .lightly/predictions/
        + tasks.json
        + classification_weather/
             + schema.json
             ...
        + classification_scenery/
             + schema.json
             ...
        + object_detection_people/
            + schema.json
            ...
        + some_directory_containing_irrelevant_things/


we can specify which subfolders contain relevant predictions in the `tasks.json`:

.. code-block:: javascript
    :caption: .lightly/predictions/tasks.json

    [
        "classification_weather",
        "classification_scenery",
        "object_detection_people"
    ]

.. note::

    If you list a subfolder which doesn't contain a valid `schema.json` file,
    the Lightly docker will fail! See below how to create a good `schema.json` file.


Prediction Schema
-----------------
For Lightly it's required to store a prediction schema. The schema helps the Lightly
Platform to correctly identify and display classes. It also helps to prevent errors
as all predictions which are loaded are validated against this schema.


For classification and object detection the prediction schema must include all the categories and ids.
For other tasks such as keypoint detection it can be useful to store additional information
like edges between keypoints.

You can provide all this information to Lightly by adding a `schema.json` to the directory of the respective task.

The schema.json file must have a key categories with a corresponding list of categories following the COCO annotation format.
For example, let's say we are working with a classification model predicting the weather on an image.
The three classes are sunny, clouded, and rainy.


.. code-block:: javascript
    :caption: .lightly/predictions/classification_weather/schema.json

    {
        "categories": [
            {
                "id": 0,
                "name": "sunny",
            },
            {
                "id": 1,
                "name": "clouded",
            },
            {
                "id": 2,
                "name": "rainy",
            }
        ]
    }



Prediction Files
----------------
Lightly requires a **single prediction file per image**. The file should be a .json
following the format defined under :ref:`Prediction Format` and stored in the subdirectory
`.lightly/predictions/${TASK_NAME}` in the storage bucket the dataset was configured with.
In order to make sure Lightly can match the predictions to the correct source image,
it's necessary to follow the naming convention:

.. code-block:: bash

    # filename of the prediction for image FILENAME.EXT
    .lightly/predictions/${TASK_NAME}/${FILENAME}.json

    # example: my_image.png, classification
    .lightly/predictions/my_classification_task/my_image.json

    # example: my_subdir/my_image.png, classification
    .lightly/predictions/my_classification_task/my_subdir/my_image.json


Prediction Files for Videos
---------------------------
When working with videos, Lightly requires a prediction file per frame. Lightly
uses a naming convention to identify frames: The filename of a frame consists of
the video filename, the video format, and the frame number (padded with 8 zeros)
separated by hyphens:

.. code-block:: bash

    # filename of the predictions of the Xth frame of video FILENAME.EXT
    .lightly/predictions/${TASK_NAME}/${FILENAME}-${X:08d}-${EXT}.json

    # example: my_video.mp4, frame 99
    .lightly/predictions/my_classification_task/my_video-00000099-mp4.json

    # example: my_subdir/my_video.mp4, frame 99
    .lightly/predictions/my_classification_task/my_subdir/my_video-00000099-mp4.json


Prediction Format
-----------------
Predictions for an image must have a `file_name` and `predictions`.
Here, `file_name` serves as a unique identifier to retrieve the image for which
the predictions are made and predictions is a list of `Prediction Singletons` for the corresponding task.

Example classification:

.. code-block:: javascript
    :caption: .lightly/predictions/classification_weather/my_image.json

    {
        "file_name": "my_image.png"
        "predictions": [ // classes: [sunny, clouded, rainy]
            {
                "category_id": 0,
                "probabilities": [0.8, 0.1, 0.1]
            }
        ]
    }

Example object detection:

.. code-block:: javascript
    :caption: .lightly/predictions/object_detection/my_image.json

    {
        "file_name": "my_image.png",
        "predictions": [ // classes: [person, car]
            {
                "category_id": 0,
                "bbox": [...],
                "score": 0.8
            },
            {
                "category_id": 1,
                "bbox": [...],
                "score": 0.9
            },
            {
                "category_id": 0,
                "bbox": [...],
                "score": 0.5
            }
        ]
    }

Note: The filename should always be the full path from the root directory.


Prediction Singletons
---------------------
The prediction singletons closely follow the `COCO results <https://cocodataset.org/#format-results>`_ format while dropping
the `image_id`. Note the the `category_id` must be the same as the one defined
in the schema and that the probabilities (if provided) must follow the order of the category ids.

**Classification:**

For classification, please use the following format:

.. code-block:: javascript

    [{
        "category_id"       : int,
        "probabilities"     : [p0, p1, ..., pN],
    }]

**Object Detection:**

For detection with bounding boxes, please use the following format:

.. code-block:: javascript

    [{
        "category_id"       : int,
        "bbox"              : [x, y, width, height],
        "score"             : float,
        "probabilities"     : [p0, p1, ..., pN],         // optional
    }]

The bounding box format follows the `COCO results <https://cocodataset.org/#format-results>`_ documentation.

.. note::

    Box coordinates are floats measured from the top left image corner (and are 0-indexed).
    We recommend rounding coordinates to the nearest tenth of a pixel to reduce resulting JSON file size.


Creating the predictions folder
-------------------------------

For creating the predictions folder, we recommend writing a script that takes your predictions and
saves them in the format just outlined. You can either save the predictions first on your local machine
and then upload them for your datasource or upload them directly to your datasource.

As an example, the following script takes an object detection annotations file in the coco format.
It needs the path to the annotations file and the output directory
where the .lightly folder should be created as input.
Don't forget to change these 2 parameters at the top of the script.

.. code-block:: python

    ### CHANGE THESE PARAMETERS
    output_filepath = "/path/to/create/.lightly/dir"
    annotation_filepath = "/path/to/_annotations.coco.json"

    import json
    import os
    from pathlib import Path

    # create prediction directory
    path_predictions = os.path.join(output_filepath, '.lightly/predictions')
    Path(path_predictions).mkdir(exist_ok=True, parents=True)

    # Create task.json
    path_task_json = os.path.join(path_predictions, 'tasks.json')
    tasks = ["my_object_detection_task"]
    with open(path_task_json, 'w') as f:
        json.dump(tasks, f)

    # read coco annotations
    with open(annotation_filepath, 'r') as f:
        coco_dict = json.load(f)

    # Create schema.json for task
    path_predictions_task = os.path.join(path_predictions, tasks[0])
    Path(path_predictions_task).mkdir(exist_ok=True)
    schema = {
        "task_description": "object-detection",
        "categories": coco_dict['categories']
    }
    path_schema_json = os.path.join(path_predictions_task, 'schema.json')
    with open(path_schema_json, 'w') as f:
        json.dump(schema, f)

    # Create predictions themselves
    image_id_to_prediction = dict()
    for image in coco_dict['images']:
        prediction = {
            'file_name': image['file_name'],
            'predictions': [],
        }
        image_id_to_prediction[image['id']] = prediction
    for ann in coco_dict['annotations']:
        pred = {
            'category_id': ann['category_id'],
            'bbox': ann['bbox'],
            'score': 0
        }
        image_id_to_prediction[ann['image_id']]['predictions'].append(pred)

    for prediction in image_id_to_prediction.values():
        filename_prediction = os.path.splitext(prediction['file_name'])[0] + '.json'
        path_to_prediction = os.path.join(path_predictions_task, filename_prediction)
        with open(path_to_prediction, 'w') as f:
            json.dump(prediction, f)