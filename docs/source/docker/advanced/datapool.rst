Docker
=================

The Lightly Datapool is a tool which allows users to incrementally build up a 
dataset for their project. It keeps track of the representations of previously
selected samples and uses this information to pick new samples in order to
maximize the quality of the final dataset. It also allows for combining two 
different datasets into one.

- | If you're interested in how the datapool works, go to
  | --> `How It Works`_

- | To see how you can use the datapool, check out
  | --> `Usage`_


How It Works
---------------

The Lightly Datapool keeps track of the selected samples in a csv file called
`datapool_latest.csv`. It contains the filenames of the selected images, their
embeddings, and their weak labels. Additionally, after training a self-supervised
model, the datapool contains the checkpoint `checkpoint_latest.ckpt` which was 
used to generate the embeddings.

The datapool is located in the `shared` directory. In general, it is a directory
with the following structure:


.. code-block:: bash

    # example of a datapool
    datapool/
    +--- datapool_latest.csv
    +--- checkpoint_latest.ckpt
    +--- history/
  
The files `datapool_latest.csv` and `checkpoint_latest.csv` are updated after every
run of the Lightly Docker. The history folder contains the previous versions of 
the datapool. This feature is meant to prevent accidental overrides and can be 
deactivated from the command-line (see `Usage`_ for more information).

Usage
---------------

.. note:: To use the datapool feature, the Lightly Docker requires write access
          to a shared directory. This directory can be passed with the `-v` flag.


To **initialize** a datapool, simply pass the name of the datapool as an argument
to your docker run command and sample from a dataset as always. The Lightly Docker
will automatically create a datapool directory and populate it with the required
files.

.. code-block:: console

   docker run --gpus all --rm -it \
      -v INPUT_DIR:/home/input_dir:ro \
      -v SHARED_DIR:/home/shared_dir \
      -v OUTPUT_DIR:/home/output_dir \
      lightly/sampling:latest \
      token=MYAWESOMETOKEN \
      append_weak_labels=False \
      stopping_condition.min_distance=0.1 \
      datapool.name=my_datapool


To **append** to your datapool, pass the name of an existing datapool as an argument.
The Lightly Docker will read the embeddings and filenames from the existing pool and
consider them during sampling. Then, it will update the datapool and checkpoint files.

.. note:: You can't change the dimension of the embeddings once the datapool has
          been initialized so choose carefully!

.. code-block:: console

   docker run --gpus all --rm -it \
      -v OTHER_INPUT_DIR:/home/input_dir:ro \
      -v SHARED_DIR:/home/shared_dir \
      -v OUTPUT_DIR:/home/output_dir \
      lightly/sampling:latest \
      token=MYAWESOMETOKEN \
      append_weak_labels=False \
      stopping_condition.min_distance=0.1 \
      datapool.name=my_datapool


To **deactivate automatic archiving** of the past datapool versions, you can pass
set the flag `keep_history` to False.

.. code-block:: console

   docker run --gpus all --rm -it \
      -v INPUT_DIR:/home/input_dir:ro \
      -v SHARED_DIR:/home/shared_dir \
      -v OUTPUT_DIR:/home/output_dir \
      lightly/sampling:latest \
      token=MYAWESOMETOKEN \
      append_weak_labels=False \
      stopping_condition.min_distance=0.1 \
      datapool.name=my_datapool \
      datapool.keep_history=False
