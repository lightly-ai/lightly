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


List of Parameters
-----------------------------------

The following are parameters which can be passed to the container:

.. code-block:: yaml

    # access token
    token: ''

    # set to true to check whether installation was successful
    sanity_check: False 

    # check for corrupted images
    enable_corruptness_check: True

    # remove exact duplicates
    remove_exact_duplicates: True

    # dump the final dataset to the output directory
    dump_dataset: False

    # pass checkpoint
    checkpoint: ''

    # pass embeddings
    embeddings: ''

    # enable training, only possible when no embeddings are passed
    enable_training: False

    # normalize the embeddings to unit length
    normalize_embeddings: True

    # sampling
    method: 'coreset'               # choose from ['coreset', 'random']
    stopping_condition:
        n_samples: -1               # float in [0., 1.] for percentage, int for number of samples, -1 means inactive
        min_distance: -1.           # float, minimum distance between two images in the sampled dataset, -1. means inactive

    # report
    n_example_images: 6             # the number of retained/removed image pairs to show in the report
    memory_requirement_in_GB: 2     # maximum size of the distance matrix required for statistics in GB

Additionally, you can pass all arguments which can be passed to the lightly CLI tool with the `lightly` prefix.
For example,

.. code-block:: console

    docker run --rm -it \
        -v INPUT_DIR:/home/input_dir:ro \
        -v OUTPUT_DIR:/home/shared_dir \
        lightly/sampling:latest \
        token=MYAWESOMETOKEN \
        lightly.loader.batch_size=512

sets the batch size during training and embedding to 512.

Choosing the Right Parameters
-----------------------------------

Below you find some distributions and the resulting histogram of the pairwise
distances. Typically, datasets consist of multiple normal or uniform 
distributions (second row). This makes sense. In autonomous driving, we collect
data in various cities, different weather conditions, or other factors. When 
working with video data from multiple cameras each camera might form a cluster
since camera from the same static camera has lots of perceptual similarity.

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

* -\-ipc="host" sets the host for inter-process communication. This flag needs to be set to use background workers. Since this is an argument to the docker run command we add it before our filter arguments.

* lightly.loader.num_workers=8 sets the number of background processes to be used for data preprocessing. Usually, the number of physical CPU cores works well.

.. code-block:: console

    docker run --rm -it \
        -v INPUT_DIR:/home/input_dir:ro \
        -v OUTPUT_DIR:/home/shared_dir \
        --ipc=host \
        lightly/sampling:latest \
        token=MYAWESOMETOKEN \
        lightly.loader.num_workers=8


