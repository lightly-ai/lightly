Sequence Selection
==================

Sequence selection allows users to select sequences of a video instead of single frames.
The key concept is the parameter `selected_sequence_length`. If its value is one (default),
the docker selects single frames. If it is larger than one, each video is split into 
sequences of that length and the frame representations are aggregated into a sequence
representation. The sampling then happens on these sequence representations.

.. note:: Sequence selection works on videos or on folders of alphabetically sorted
    frames.


- | If you're interested in how the sequence selection works, go to
  | --> `How It Works`_

- | To see how you can use the sequence selection, check out
  | --> `Usage`_


How It Works
-------------
Sequence selection consists of the following steps:

- Each input video is split into sequences of length `selected_sequence_length`.
- Next, the embeddings of all frames in a sequence are aggregated (averaged).
- The Lightly sampling algorithm selects relevant sequences.
- Finally, the indices of the selected sequence frames are reconstructed.
- The report is generated and (if requested) the selected frames are saved.
  

Usage
-----------

To select sequences of length `X` simply add the argument `selected_sequence_length=X`
to your docker run command. For example, let's say we have a folder with two videos
which we randomly downloaded from `Pexels <https://www.pexels.com/>`_: 

.. code-block:: console

    ls /datasets/pexels/
    > Pexels Videos 1409899.mp4  Pexels Videos 2495382.mp4

Now, we want to select sequences of length ten. We use:

.. code-block:: console

    export INPUT_DIR=/datasets/pexels

    docker run --gpus all --rm -it \
        -v $INPUT_DIR:/home/input_dir:ro \
        -v OUTPUT_DIR:/home/output_dir \
        lightly/sampling:latest \
        token=MYAWESOMETOKEN \
        lightly.loader.num_workers=0 \
        stopping_condition.n_samples=200 \
        enable_corruptness_check=False \
        remove_exact_duplicates=False \
        dump_dataset=True \
        selected_sequence_length=10

The above command will select 20 sequences each consisting of ten frames. The selected
frames are then saved in the output directory for further processing.

.. warning:: The stopping condition `n_samples` must be equal to to the number of
    desired sequences times the `selected_sequence_length`. In the example above,
    20 sequences times ten frames is exactly 200.


In our example, a look at a PCA of the embeddings of the selected frames nicely shows
the 20 selected sequences. The following image is taken from the output of the Lightly
docker:

.. figure:: images/sequence_selection_pca.png
    :align: center
    :alt: PCA of embeddings of frames
    :figwidth: 80%

    PCA of the embeddings of the frames in the selected sequences from the two
    input videos (yellow and purple).