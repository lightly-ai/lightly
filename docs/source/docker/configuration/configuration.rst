Configuration
===================================

As the lightly framework the docker solution can be configured using Hydra.

You can set parameters like this:

.. code-block:: console

    docker




List of Parameters
-----------------------------------

The following are parameter which can be passed to the container:

.. code-block:: yaml

    # access token
    token: ''

    # set to true to check whether installation was successful
    sanity_check: False 

    # set to false to disable check for corrupted images
    enable_corruptness_check: False

    # remove exact duplicates
    remove_exact_duplicates: False

    # pass checkpoint
    checkpoint: ''

    # pass embeddings
    embeddings: ''

    # save?
    save_sampled_dataset: False
    save_sampled_embeddings: False


    # sampling
    method: 'coreset'
    stopping_condition:
    n_samples: 100
    min_distance: 0.
    existing_selection_column_name: ''
    active_learning_score_column_name: ''
    masked_out_column_name: ''

    # report
    n_example_images: 6
    memory_requirement_in_GB: 2

    # i/o (don't change)
    input_dir: 'input_dir'
    output_dir: 'output_dir'
    shared_dir: 'shared_dir'

