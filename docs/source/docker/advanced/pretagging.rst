.. _ref-docker-pretagging:

Pretagging
======================

Lightly Docker supports the use of pre-trained models to tag the dataset. We 
call this pretagging. For now, we offer a pre-trained model for object detection 
optimized for autonomous-driving.

Using a pretrained model does not resolve the need for high quality human annotations.
However, we can use the model predictions to get some idea of the underlying 
distribution within the dataset.

The model is capable of detecting the following core classes:

 - bicycle
 - bus
 - car
 - motorcycle
 - person
 - train
 - truck


How It Works
---------------

Our pretagging model is based on a FasterRCNN model with a ResNet-50 backbone.
The model has been trained on a dataset consisting of ~100k images.

The results of pretagging are visualized in the report. We report both, the 
object distribution before and after the selection process. 

The following image shows an example of such a histogram for the input data
before filtering.

.. figure:: ../resources/pretagging_histogram_example.png
    :align: center
    :alt: some alt text

    Histogram plot of the pretagging model for the input data (full dataset).
    The plot shows the distribution of the various detected classes. 
    Further it shows the average number of objects per image.

For every docker run with pretagging enabled we also dump all model predictions
into a json file with the following format:

.. code-block:: javascript

    // boxes have format x1, y1, x2, y2
    [
        {
            "filename": "0000000095.png",
            "boxes": [
                [
                    0.869,
                    0.153,
                    0.885,
                    0.197
                ],
                [
                    0.231,
                    0.175,
                    0.291,
                    0.202
                ]
            ],
            "labels": [
                "person",
                "car"
            ],
            "scores": [
                0.9845203757286072,
                0.9323102831840515
            ]
        },
        ...
    ]


Usage
---------------

Pretagging can be activated by passing the following argument to your docker
run command: `pretagging=True`

- `pretagging=True` enables the use of the pretagging model
- `pretagging_debug=True` add a few images to the report for debugging showing showing the image with the bounding box predictions.
- `pretagging_upload=True` enables uploading of the predictions to a configured datasource.


The final docker run command to enable pretagging as well as pretagging_debug
should look like this:

.. code-block:: console

   docker run --gpus all --rm -it \
      -v {INPUT_DIR}:/home/input_dir:ro \
      -v {SHARED_DIR}:/home/shared_dir \
      -v {OUTPUT_DIR}:/home/output_dir \
      lightly/worker:latest \
      token=MYAWESOMETOKEN \
      pretagging=True \
      pretagging_debug=True

The following shows an example of how the debugging images in the report look like:

.. figure:: ../resources/pretagging_debug_example.png
    :align: center
    :alt: some alt text

    The plot shows the detected bounding boxes from the pretagging overlayed
    on the image. Use the debug feature to figure out whether the pretagging 
    mechanism works properly on your dataset.


Pretagging for Selection
^^^^^^^^^^^^^^^^^^^^^^^^

You can also use pretagging to guide the data selection process. This can be
helpful if you for example only care about images where there is at least one
person and more than one car.

To create such a pretagging selection mechanism you need to create a config file.

For the example of selecting only images with >=1 person and >=2 cars we can 
create a `min_requirements.json` file like this:

.. code-block:: json

    {
        "person": 1,
        "car": 2
    }

Move this file to the shared directory (to make it accessible to the docker
container).
Finally, run the docker with `pretagging=True`
and `pretagging_config=min_requirements.json`.
Only images satisfying all declared requirements will be selected.
