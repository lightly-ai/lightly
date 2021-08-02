Configuration
===================================

As the lightly framework the docker solution can be configured using Hydra.

The example below shows how the `token` parameter can be set when running the docker container.

.. code-block:: console

    docker run --rm -it \
        -v INPUT_DIR:/home/input_dir:ro \
        -v OUTPUT_DIR:/home/shared_dir \
        --ipc="host" --network="host" \
        lightly/sampling:latest \
        token=MYAWESOMETOKEN


.. _rst-docker-parameters:

List of Parameters
-----------------------------------

The following are parameters which can be passed to the container:

.. code-block:: yaml

    # access token (get it from app.lightly.ai)
    token: ''

    # set to true to check whether installation was successful
    sanity_check: False

    # enable check for corrupted images (copies healthy ones if necessary)
    enable_corruptness_check: True

    # remove exact duplicates
    remove_exact_duplicates: True

    # path to the checkpoint relative to the shared directory
    checkpoint: ''

    # path to the embeddings file relative to the shared directory
    embeddings: ''

    # enable training, only possible when no embeddings are passed
    enable_training: False

    # dump the final dataset to the output directory
    dump_dataset: False
    dump_sampled_embeddings: True
    # set the size of the dumped images, use =x or =[x,y] to match the shortest
    # edge to x or to resize the image to (x,y), use =-1 for no resizing (default)
    output_image_size: -1
    output_image_format: 'png'

    # upload?
    upload_dataset: False

    # pretagging
    pretagging: False
    pretagging_debug: False
    pretagging_config: ''

    # append weak labels
    append_weak_labels: False

    # normalize the embeddings to unit length
    normalize_embeddings: True

    # active learning scorer
    scorer: 'object-frequency'
    scorer_config:
      frequency_penalty: 0.25
      min_score: 0.9


    # sampling
    method: 'coreset'
    stopping_condition:
        n_samples: -1
        min_distance: -1.

    # datapool
    datapool:
        name:                       # name of the datapool
        keep_history: True          # if True keeps backup of all previous data pool states

    # report
    n_example_images: 6             # the number of retained/removed image pairs to show in the report
    memory_requirement_in_GB: 2     # maximum size of the distance matrix allowed for statistics in GB
    show_video_sampling_timeline: True

Additionally, you can pass all arguments which can be passed to the lightly CLI tool with the `lightly` prefix.
For example,

.. code-block:: console

    docker run --rm -it \
        -v INPUT_DIR:/home/input_dir:ro \
        -v OUTPUT_DIR:/home/output_dir \
        lightly/sampling:latest \
        token=MYAWESOMETOKEN \
        lightly.loader.batch_size=512

sets the batch size during training and embedding to 512. You find a list of all
lightly CLI parameters here: :ref:`ref-cli-config-default`

Choosing the Right Parameters
-----------------------------------

Below you find some distributions and the resulting histogram of the pairwise
distances. Typically, datasets consist of multiple normal or uniform 
distributions (second row). This makes sense. In autonomous driving, we collect
data in various cities, different weather conditions, or other factors. When 
working with video data from multiple cameras each camera might form a cluster
since images from the same static camera have lots of perceptual similarity.

The more interesting question is what kind of distribution you're aiming for.


**If we want to diversify the dataset** (e.g. create a really hard test set
covering all the special cases) we might want to aim for what looks like a grid.
The log histogram (yes, we plot the histograms in log scale!) for a grid pattern with
equal distance between two neighboring samples looks like a D.


**If you want to remove nearby duplicates** (e.g. reduce overfitting and bias)
we see good results when trying to sample using the *min_distance* stop condition.
E.g. set the *min_distance* to 0.1 to get rid of the small peak (if there is any)
close to 0 pairwise distance. 


.. image:: images/histograms_overview.png



Increase I/O Performance
-----------------------------------
During the embedding process, the I/O bandwidth can often slow down the computation. A progress bar shows you the current compute 
efficiency which is calculated as the time spent on computation compared to overall time per batch. A number close to 1.0 tells you
that your system is well utilized. A number close to 0.0 however, suggests that there is an I/O bottleneck. This can be the case for
datasets consisting of very high-resolution images. Loading them from harddisk and preprocessing can take a lot of time.

To mitigate the effect of low I/O speed one can use background workers to load the data. First, we need to tell Docker to use
the host system for inter-process communication. Then, we can tell the filter to use multiple workers for data preprocessing.
You can use them by adding the following two parts to your docker run command:

* **-\-ipc="host"** sets the host for inter-process communication. 
  This flag needs to be set to use background workers. Since this is an argument 
  to the docker run command we add it before our filter arguments.

* **lightly.loader.num_workers=8** sets the number of background processes 
  to be used for data preprocessing. Usually, the number of physical 
  CPU cores works well.

.. code-block:: console

    docker run --rm -it \
        -v INPUT_DIR:/home/input_dir:ro \
        -v OUTPUT_DIR:/home/output_dir \
        --ipc=host \
        lightly/sampling:latest \
        token=MYAWESOMETOKEN \
        lightly.loader.num_workers=8


