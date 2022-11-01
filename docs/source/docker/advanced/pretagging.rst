.. _docker-pretagging:

Pretagging
======================

Lightly Worker supports the use of pre-trained models to tag the dataset. We 
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

For every Lightly Worker run with pretagging enabled we also dump all model predictions
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

Pretagging can be activated by passing the following argument to your
Lightly Worker config: :code:`'pretagging': True`

- :code:`'pretagging': True` enables the use of the pretagging model
- :code:`'pretagging_debug': True` add a few images to the report for debugging showing the image with the bounding box predictions.


A full Python script showing how to create such as job is shown here:

.. literalinclude:: ./code_examples/python_run_pretagging.py
  :linenos:
  :emphasize-lines: 75-76
  :language: python


After running the Python script to create the job we need to make sure we have
a running Lightly Worker to process the job. We can use the following
code to sping up a Lightly Worker

.. code-block:: console

  docker run --shm-size="1024m" --rm --gpus all -it \
    lightly/worker:latest \
    token=YOUR_TOKEN  worker.worker_id=YOUR_WORKER_ID

The following shows an example of how the debugging images in the report look like:

.. figure:: ../resources/pretagging_debug_example.png
    :align: center
    :alt: some alt text

    The plot shows the detected bounding boxes from the pretagging overlayed
    on the image. Use the debug feature to figure out whether the pretagging 
    mechanism works properly on your dataset.
