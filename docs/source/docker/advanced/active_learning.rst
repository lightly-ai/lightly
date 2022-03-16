.. _ref-docker-active-learning:

Active Learning
===============

Lightly makes use of active learning scores to select the samples which will yield
the biggest improvements of your machine learning model. The scores are calculated
on-the-fly based on model predictions and provide the selection algorithm with feedback
about the uncertainty of the model for the given sample. 

Learn more about the concept of active learning scores:
:ref:`lightly-active-learning-scorers`.

Prerequisites
--------------
In order to do active learning with Lightly, you will need the following things:

- The installed Lightly docker (see :ref:`ref-docker-setup`)
- A dataset with a configured datasource (see :ref:`ref-docker-with-datasource-datapool`)

.. note::

    The dataset does not need to be empty! For example, an initial selection without
    active learning can be used to train a model. The predictions from this model
    can then be used to improve your dataset by adding new images to it through active learning.


Next, you need to prepare the model predictions in the correct format.

Predictions
-----------

In the following, we will outline the format of the predictions required by the 
Lightly docker. Everything regarding predictions will take place in a subdirectory
of your configured datasource called `.lightly/predictions`. The general structure
of this directory will look like this:


.. code-block:: bash

    .lightly/predictions/
        + tasks.json
        + task_1/
             + schema.json
             + image_1.json
             ...
             + image_X.json
        + task_2/
             + schema.json
             + image_1.json
             ...
             + image_Y.json


Where each subdirectory corresponds to one prediction task (e.g. a classification task
and an object detection task). All of the files are explained in the next sections.


Prediction Tasks
^^^^^^^^^^^^^^^^
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
        + some_directory_containing_irrelevant_things/


we can specify which subfolders contain relevant predictions in the `tasks.json`:

.. code-block:: javascript
    :caption: .lightly/predictions/tasks.json

    [
        "classification_weather",
        "classification_scenery"
    ]

.. note::

    If you list a subfolder which doesn't contain a valid `schema.json` file,
    the Lightly docker will fail! See below how to create a good `schema.json` file.


Prediction Schema
^^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
^^^^^^^^^^^^^^^^^
Predictions for an image must have a `file_name` and `predictions`.
Here, `file_name` serves as a unique identifier to retrieve the image for which
the predictions are made and predictions is a list of `Prediction Singletons` for the corresponding task.

Example classification:

.. code-block:: javascript
    :caption: .lightly/predictions/classification_weather/my_image.json

    {
        "file_name": "my_image.png"
        "predictions": [ // classes: [sunny, clouded, rain]
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
^^^^^^^^^^^^^^^^^^^^^
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

Selection
-------------------------
Once you have everything set up as described above, you can do an active learning
iteration by specifying the following three things in your Lightly docker config:

- `method`
- `active_learning.task_name`
- `active_learning.score_name`

Here's an example of how your config could look like if you want to add `100` more
images to your dataset by doing active learning:


.. tabs::

    .. tab:: Run Command

        .. code-block:: bash

            docker run --gpus all --rm -it \
                -v {OUTPUT_DIR}:/home/output_dir \
                lightly/sampling:latest \
                token=MYAWESOMETOKEN \
                enable_corruptness_check=True \
                remove_exact_duplicates=True \
                pretagging=False \
                pretagging_debug=False \
                method='coral' \
                stopping_condition.n_samples=100 \
                stopping_condition.min_distance=-1 \
                scorer_config.frequency_penalty=0.25 \
                scorer_config.min_score=0.9 \
                active_learning.task_name='my-classification-task' \
                active_learning.score_name='uncertainty_margin' \
                datasource.dataset_id={DATASET_ID} \
                datasource.process_all=True


    .. tab:: JSON

        .. code-block:: javascript  

            {
                enable_corruptness_check: true,
                remove_exact_duplicates: true,
                enable_training: false,
                pretagging: false,
                pretagging_debug: false,
                method: 'coral',
                stopping_condition: {
                    n_samples: 100,
                    min_distance: -1
                },
                scorer: 'object-frequency',
                scorer_config: {
                    frequency_penalty: 0.25,
                    min_score: 0.9
                },
                active_learning: {
                    task_name: 'my-classification-task',
                    score_name: 'uncertainty_margin'
                },
                datasource: {
                    process_all: true
                }
            }

Here, we set the `method` to `coral` which simultaneously considers the diversity
and the active learning scores of the samples. The stopping condition was set the
`n_samples: 100` and under `active_learning.task_name` we entered the task name of
our predictions (see :ref:`TODO`). For this
iteration, we picked the `uncertainty_margin` score. Learn more about the different scores in the next section.


Active Learning with Custom Scores
----------------------------------

.. note::
    This is not recommended anymore as of March 2022 and will be deprecated in the future!


For running an active learning step with the Lightly docker, we need to perform
3 steps:

1. Create an `embeddings.csv` file. You can use your own models or the Lightly docker for this.
2. Add your active learning scores as an additional column to the embeddings file.
3. Use the Lightly docker to perform a active learning sampling on the scores.

Learn more about the concept of active learning 
:ref:`lightly-active-learning-scorers`.


Create Embeddings
^^^^^^^^^^^^^^^^^

You can create embeddings using your own model. Just make sure the resulting
`embeddings.csv` file matches the required format:
:ref:`ref-cli-embeddings-lightly`. 

Alternatively, you can run the docker as usual and as described in the 
:ref:`rst-docker-first-steps` section.
The only difference is that you set the number of samples to be sampled to `1.0`,
as this prevents sampling and just creates the embedding of the full dataset.

E.g. create and run a bash script with the following content:

.. code::

    # Have this in a step_1_run_docker_create_embeddings.sh
    INPUT_DIR=/path/to/your/dataset
    SHARED_DIR=/path/to/shared
    OUTPUT_DIR=/path/to/output

    TOKEN= # put your token here
    N_SAMPLES=1.0

    docker run --gpus all --rm -it \
      -v ${INPUT_DIR}:/home/input_dir:ro  \
      -v ${SHARED_DIR}:/home/shared_dir:ro \
      -v ${OUTPUT_DIR}:/home/output_dir \
      lightly/sampling:latest \
      token=${TOKEN} \
      lightly.loader.num_workers=4     \
      stopping_condition.n_samples=${N_SAMPLES}\
      method=coreset \
      enable_training=True     \
      lightly.trainer.max_epochs=20

Running it will create a terminal output similar to the following:

.. code-block::

    [2021-09-29 13:32:11] Loading initial dataset...
    [2021-09-29 13:32:11] Found 372 input images in input_dir.
    [2021-09-29 13:32:11] Lightly On-Premise License is valid
    [2021-09-29 13:32:11] Checking for corrupt images (disable with enable_corruptness_check=False).
    Corrupt images found: 0: 100%|██████████████████| 372/372 [00:01<00:00, 310.35it/s]
    [2021-09-29 13:32:14] Training self-supervised model.
    GPU available: True, used: True
    [2021-09-29 13:32:57,696][lightning][INFO] - GPU available: True, used: True
    TPU available: None, using: 0 TPU cores
    [2021-09-29 13:32:57,697][lightning][INFO] - TPU available: None, using: 0 TPU cores
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    [2021-09-29 13:32:57,697][lightning][INFO] - LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

      | Name      | Type       | Params
    -----------------------------------------
    0 | model     | SimCLR     | 11.2 M
    1 | criterion | NTXentLoss | 0
    -----------------------------------------
    11.2 M    Trainable params
    0         Non-trainable params
    [2021-09-29 13:34:29,772][lightning][INFO] - Saving latest checkpoint...
    Epoch 19: 100%|████████████████████████████████| 23/23 [00:04<00:00,  5.10it/s, loss=2.52, v_num=0]
    [2021-09-29 13:34:29] Embedding images.
    Compute efficiency: 0.90: 100%|█████████████████████████| 24/24 [00:01<00:00, 21.85it/s]
    [2021-09-29 13:34:31] Saving embeddings to output_dir/2021-09-29/13:32:11/data/embeddings.csv.
    [2021-09-29 13:34:31] Unique embeddings are stored in output_dir/2021-09-29/13:32:11/data/embeddings.csv
    [2021-09-29 13:34:31] Normalizing embeddings to unit length (disable with normalize_embeddings=False).
    [2021-09-29 13:34:31] Normalized embeddings are stored in output_dir/2021-09-29/13:32:11/data/normalized_embeddings.csv
    [2021-09-29 13:34:31] Sampling dataset with stopping condition: n_samples=372
    [2021-09-29 13:34:31] Skipped sampling because the number of remaining images is smaller than the number of requested samples.
    [2021-09-29 13:34:31] Writing report to output_dir/2021-09-29/13:32:11/report.pdf.
    [2021-09-29 13:35:04] Writing csv with information about removed samples to output_dir/2021-09-29/13:32:11/removed_samples.csv
    [2021-09-29 13:35:04] Done!

By running it, this will create an `embeddings.csv` file
in the output directory. Locate it and save the path to it.
E.g. It may be found under
`/path/to/output/2021-09-28/15:47:34/data/embeddings.csv`

It should look similar to this:

+----------------+--------------+--------------+--------------+--------------+---------+
| filenames      | embedding_0  | embedding_1  | embedding_2  | embedding_3  | labels  |
+================+==============+==============+==============+==============+=========+
| cats/0001.jpg  | 0.29625183   | 0.50055015   | 0.36491454   | 0.8156051    | 0       |
+----------------+--------------+--------------+--------------+--------------+---------+
| dogs/0005.jpg  | 0.36491454   | 0.29625183   | 0.38491454   | 0.36491454   | 1       |
+----------------+--------------+--------------+--------------+--------------+---------+
| cats/0014.jpg  | 0.8156051    | 0.59055015   | 0.29625183   | 0.50055015   | 0       |
+----------------+--------------+--------------+--------------+--------------+---------+


Add Active Learning Scores
^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to use the predictions from your model as active learning scores,
you can use the :ref:`lightly-active-learning-scorers` from the lightly pip package.

.. note:: You can also use your own scorers. Just make sure that you get a value
          between `0.0` and `1.0` for each sample. A number close to `1.0` should
          indicate a very important sample you want to be selected with a higher
          probability.

We provide a simple Python script to append a list of `scores` to the `embeddings.csv` file.

.. code-block:: python

    # Have this in a step_2_add_al_scores.py

    from typing import Iterable
    import csv
    import os

    """
    Run your detection model here
    Use the scorers offered by lightly to generate active learning scores.
    """

    # Let's assume that you have one active learning score for every image.
    # WARNING: The order of the scores MUST match the order of filenames
    # in the embeddings.csv.
    scores: Iterable[float] =  # must be an iterable of floats,
    # e.g. a list of float or a 1d-numpy array

    # define the function to add the scores to the embeddings.csv
    def add_al_scores_to_csv(
            input_file_path: str, output_file_path: str,
            scores: Iterable[float], column_name: str = "al_score"
    ):
        with open(input_file_path, 'r') as read_obj:
            with open(output_file_path, 'w') as write_obj:
                csv_reader = csv.reader(read_obj)
                csv_writer = csv.writer(write_obj)

                # add the column name
                first_row = next(csv_reader)
                first_row.append(column_name)
                csv_writer.writerow(first_row)

                # add the scores
                for row, score in zip(csv_reader, scores):
                    row.append(str(score))
                    csv_writer.writerow(row)

    # use the function
    # adapt the following line to use the correct path to the embeddings.csv
    input_embeddings_csv = '/path/to/output/2021-07-28/12:00:00/data/embeddings.csv'
    output_embeddings_csv = input_embeddings_csv.replace('.csv', '_al.csv')
    add_al_scores_to_csv(input_embeddings_csv, output_embeddings_csv, scores)

    print("Use the following path to the embeddings_al.csv in the next step:")
    print(output_embeddings_csv)

Running it will create a terminal output similar to the following:

.. code-block::

    (base) user@machine:~/GitHub/playground/docker_with_al$ sudo python3 step_2_add_al_scores.py
    Use the following path to the embedding.csv in the next step:
    /path/to/output/2021-07-28/12:00:00/data/embeddings_al.csv

Your embeddings_al.csv should look similar to this:

+----------------+--------------+--------------+--------------+--------------+---------+-----------+
| filenames      | embedding_0  | embedding_1  | embedding_2  | embedding_3  | labels  | al_score  |
+================+==============+==============+==============+==============+=========+===========+
| cats/0001.jpg  | 0.29625183   | 0.50055015   | 0.36491454   | 0.8156051    | 0       | 0.7231    |
+----------------+--------------+--------------+--------------+--------------+---------+-----------+
| dogs/0005.jpg  | 0.36491454   | 0.29625183   | 0.38491454   | 0.36491454   | 1       | 0.91941   |
+----------------+--------------+--------------+--------------+--------------+---------+-----------+
| cats/0014.jpg  | 0.8156051    | 0.59055015   | 0.29625183   | 0.50055015   | 0       | 0.01422   |
+----------------+--------------+--------------+--------------+--------------+---------+-----------+


Run Active Learning using the Docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At this point you should have an `embeddings.csv` file with the active learning 
scores in a column named `al_scores`. 

We can now perform an active learning sampling using the `CORAL` sampler. In order
to do the sampling on the `embeddings.csv` file we need to make this file
accessible to the docker. We can do this by using the `shared_dir` feature of the
docker as described in :ref:`docker-sampling-from-embeddings`. 

E.g. use the following bash script.

.. code-block:: bash

    #!/bin/bash -e

    # Have this in a step_3_run_docker_coral.sh
    
    INPUT_DIR=/path/to/your/dataset/
    SHARED_DIR=/path/to/shared/
    OUTPUT_DIR=/path/to/output/
    
    EMBEDDING_FILE= # insert the path printed in the last step here.
    # e.g. /path/to/output/2021-07-28/12:00:00/data/embeddings_al.csv

    cp INPUT_EMBEDDING_FILE SHARED_DIR # copy the embedding file to the shared directory
    EMBEDDINGS_REL_TO_SHARED=embeddings_al.csv
    

    TOKEN= # put your token here
    N_SAMPLES= # Choose how many samples you want to use here, e.g. 0.1 for 10 percent.

    docker run --gpus all --rm -it \
        -v ${INPUT_DIR}:/home/input_dir:ro  \
        -v ${SHARED_DIR}:/home/shared_dir:ro \
        -v ${OUTPUT_DIR}:/home/output_dir \
        lightly/sampling:latest \
        token=${TOKEN} \
        lightly.loader.num_workers=4     \
        stopping_condition.n_samples=${N_SAMPLES}\
        method=coral \
        enable_training=False     \
        dump_dataset=True \
        upload_dataset=False \
        embeddings=${EMBEDDINGS_REL_TO_SHARED} \
        active_learning_score_column_name="al_score" \
        scorer=""
      
Your terminal output should look similar to this:

.. code-block::

    [2021-09-29 09:36:27] Loading initial embedding file...
    [2021-09-29 09:36:27] Output images will not be resized.
    [2021-09-29 09:36:27] Found 372 input images in shared_dir/embeddings_al.csv.
    [2021-09-29 09:36:27] Lightly On-Premise License is valid
    [2021-09-29 09:36:28] Removing exact duplicates (disable with remove_exact_duplicates=False).
    [2021-09-29 09:36:28] Found 0 exact duplicates.
    [2021-09-29 09:36:28] Unique embeddings are stored in shared_dir/embeddings_al.csv
    [2021-09-29 09:36:28] Normalizing embeddings to unit length (disable with normalize_embeddings=False).
    [2021-09-29 09:36:28] Normalized embeddings are stored in output_dir/2021-09-29/09:36:27/data/normalized_embeddings.csv
    [2021-09-29 09:36:28] Sampling dataset with stopping condition: n_samples=10
    [2021-09-29 09:36:28] Sampled 10 images.
    [2021-09-29 09:36:28] Writing report to output_dir/2021-09-29/09:36:27/report.pdf.
    [2021-09-29 09:36:56] Writing csv with information about removed samples to output_dir/2021-09-29/09:36:27/removed_samples.csv
    [2021-09-29 09:36:56] Done!
