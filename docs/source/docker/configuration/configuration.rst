Configuration
===================================

As the lightly framework the docker solution can be configured using Hydra.

The example below shows how the `token` parameter can be set when running the docker container.

.. code-block:: console

    docker run --rm -it \
        -v ${INPUT_DIR}:/home/input_dir:ro \
        -v ${SHARED_DIR}:/home/output_dir:ro \
        -v ${OUTPUT_DIR}:/home/shared_dir \
        --ipc="host" --network="host" \
        lightly/sampling:latest \
        token=MYAWESOMETOKEN


List of Parameters
-----------------------------------

The following are parameter which can be passed to the container:

.. code-block:: yaml

    # access token
    token: ''

    # set to true to check whether installation was successful
    sanity_check: False 

    # set to false to disable check for corrupted images
    enable_corruptness_check: True

    # remove exact duplicates
    remove_exact_duplicates: True

    # pass checkpoint
    checkpoint: ''

    # pass embeddings
    embeddings: ''

    # train?
    enable_training: False

    # normalize
    normalize_embeddings: True

    # sampling
    method: 'coreset'
    stopping_condition:
        n_samples: -1       # -1 means inactive
        min_distance: -1.   # -1. means inactive

    # report
    n_example_images: 6
    memory_requirement_in_GB: 2

Additionally, you can pass all arguments which can be passed to the lightly CLI tool with the `lightly` prefix.
For example,

.. code-block:: console

    docker run --rm -it \
        -v ${INPUT_DIR}:/home/input_dir:ro \
        -v ${SHARED_DIR}:/home/output_dir:ro \
        -v ${OUTPUT_DIR}:/home/shared_dir \
        lightly/sampling:latest \
        token=MYAWESOMETOKEN \
        lightly.loader.batch_size=512

sets the batch size during training and embedding to 512.


Increase I/O Performance
-----------------------------------
During the embedding process the I/O bandwidth can often slow down computation. A progress bar shows you the current compute 
efficiency which is calculated as the time spent on computation compared to overall time per batch. A number close to 1.0 tells you
that your system is well utilized. A number close to 0.0 however, suggests that there is an I/O bottleneck. This can be the case for
datasets consisting of very high resolution images. Loading them from harddisk and preprocessing can take a lot of time.

To mitigate the effect of low I/O speed one can use background workers to load the data. First, we need to tell docker to use
the host system for inter-process communication. Then, we can tell the filter to use multiple workers for data preprocessing.
You can use them by adding the following two parts to your docker run command:

* -\-ipc="host" sets the host for inter-process communication. This flag needs to be set to use background workers. Since this is an argument to the docker run command we add it before our filter arguments.

* lightly.loader.num_workers=8 sets the number of background processes to be used for data preprocessing. Usually, the number of physical CPU cores works well.

.. code-block:: console

    docker run --rm -it \
        -v ${INPUT_DIR}:/home/input_dir:ro \
        -v ${SHARED_DIR}:/home/output_dir:ro \
        -v ${OUTPUT_DIR}:/home/shared_dir \
        --ipc=host \
        lightly/sampling:latest \
        token=MYAWESOMETOKEN \
        lightly.loader.num_workers=8
