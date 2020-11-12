First Steps
===================================

The docker solution can be used like a CLI interface. You run the container, tell it where to find data and where to store the result. That's it.
There are various parameters you can pass to the container. We put a lot of effort to also expose the full lightly framework configuration.
You could basically use the docker solution to train a self-supervised model instead of using the Python framework.

Before jumping into the detail lets have a look at some basics.
The docker container can be used like a linux script. You can control parameters by changing flags.

Use the following command to get an overview of the available parameters:

.. code-block:: console

    docker run --gpus all --rm -it whattolabel/data-filtering:latest --help


Storage Access
-----------------------------------

We use volume mapping provided by the docker run command to process datasets. 
A docker container itself is not considered to be a good place to store your data. 
Volume mapping allows the container to work with the filesystem of the host system.

There are **three** types of volume mappings:

* **Input Data:** The path to the dataset we want to process
* **Cache:** The path to a cache directory (used to speed up repetitive processing of the same dataset)
* **Experiment Output:** The path to the results of processing the dataset (consisting of output file lists, reports etc.)

Data Format
TODO

Embedding and Filtering a Dataset
-----------------------------------

Using the docker solution looks like this:

.. code-block:: console

    docker run --gpus all --rm -it -v /datasets/food-101/train:/home/input_dir:ro \
        -v /datasets/food-101_filtered:/home/output_dir \
        -v /datasets/docker_shared_dir:/home/shared_dir \
        --ipc="host" --network="host" \
        lightly/sampling:latest token=myawesometoken collate.input_size=64 \
        loader.batch_size=256 loader.num_workers=4 trainer.max_epochs=3 \
        stopping_condition.n_samples=30000 remove_exact_duplicates=True \
        enable_corruptness_check=True


Train a Self-Supervised Model
-----------------------------------

Train from scratch...
TODO

Continue training of a model using a checkpoint...
TODO

Use a Pre-Trained Model
-----------------------------------

set max_epochs=0 ...
TODO


Use Weak Labels
-----------------------------------

(how to enable/ disable weak labels)
TODO 

How to pass your own labels
TODO


Report
-----------------------------------

Explain output folder structure
TODO

Explain report content
TODO

Explain how to use report information
TODO