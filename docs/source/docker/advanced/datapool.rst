.. _datapool:

Datapool
=================

Lightly has been designed in a way that you can incrementally build up a 
dataset for your project. The software automatically keeps track of the 
representations of previously selected samples and uses this information 
to pick new samples in order to maximize the quality of the final dataset. 
It also allows for combining two different datasets into one.

For example, let's imagine we have a dataset of street videos. After running
the Lightly Worker once we added 4 more street videos to the bucket.
The new raw data might include samples which should be added to your dataset
in the Lightly Platform, so you want to add a subset of them to your dataset.

This workflow is supported by the Lightly Platform using a datapool.
It remembers which raw data in your bucket has already been processed
and will ignore it in future Lightly Worker runs.

Thus you can run the Lightly Worker with the same command again. It will find
your new raw data in the bucket, stream, embed and subsample it and then add it to
your existing dataset. The selection strategy will take the existing data in your dataset
into account when selecting new data to be added to your dataset.

.. image:: ./images/webapp-embedding-after-2nd-docker.png

After the Lightly Worker run we can go to the embedding view of the Lightly Platform
to see the newly added samples there in a new tag. We see that the new samples
(in green) fill some gaps left by the images in the first iteration (in grey).
However, there are still some gaps left, which could be filled by adding more videos
to the bucket and running the Lightly Worker again.

This workflow of iteratively growing your dataset with the Lightly Worker
has the following advantages:

- You can learn from your findings after each iteration
  to know which raw data you need to collect next.
- Only your new data is processed, saving you time and compute cost.
- You don't need to configure anything, just run the same command again.
- Only samples which are different to the existing ones are added to the dataset.

If you want to search all data in your bucket for new samples
instead of only newly added data,
then set :code:`'datasource.process_all': True` in your worker config. This has the
same effect as creating a new Lightly dataset and running the Lightly Worker from scratch
on the full dataset. We process all data instead of only the newly added ones.


Example
---------------

In this example we will do the following steps:

#. Schedule a run to process a cloud bucket with 3 videos
#. Add 2 more videos to the same bucket
#. Run the Lightly Worker with the same config again to use the datapool feature


Here we show the content of the bucket before running the Lightly Worker for the
first time.

.. code-block:: console

    videos/
    |-- campus4-c0.avi
    |-- passageway1-c1.avi
    `-- terrace1-c0.avi

Now we can run the following code to select a subset based on the 
:code:`'stopping_condition_minimum_distance': 0.1` stopping condition. In a first,
selection run we only select images with the specific minimum distance between 
each other based on the embeddings. 

.. literalinclude:: ./code_examples/python_run_datapool_example.py
  :linenos:
  :language: python

After running the code we have to make sure we have a running Lightly Worker 
to process the job.
We can start the Lightly Worker using the following command

.. code-block:: console

  docker run --shm-size="1024m" --rm --gpus all -it \
    -v /docker-output:/home/output_dir lightly/worker:latest \
    token=YOUR_TOKEN  worker.worker_id=YOUR_WORKER_ID

After we have processed the initial data and created a dataset, 
we've collected more data and our bucket now looks like this:

.. code-block:: console

    videos/
    |-- campus4-c0.avi
    |-- campus7-c0.avi
    |-- passageway1-c1.avi
    |-- terrace1-c0.avi
    `-- terrace1-c3.avi

We can run the same script again (it won't create a new dataset but use the
existing one based on the dataset name).


How It Works
---------------

The Lightly Datapool keeps track of the selected samples in a csv file called
`datapool_latest.csv`. It contains the filenames of the selected images and their
embeddings. This feature is currently only supported without training of a custom
model. Please make sure :code:`'enable_training': False` is set in your worker config.
