Active learning
==============================================

For running an active learning step with the Lightly docker, we need to perform
3 steps:

1. Create an `embeddings.csv` file with the Lightly docker.
2. Add your active learning scores as an additional column.
3. Use the Lightly docker to perform a active learning sampling on the scores.



Step 1: Create an embeddings.csv file with the Lightly docker.
---------------
Run the docker as usual and as described in the getting started section.
The only difference is that you set the number of samples to be sampled to 1.0,
this prevents sampling.
E.g. create and run a bash script with the following content:

.. code::

    # Have this in a step_1_run_docker_create_embeddings.sh
    INPUT_DIR=/path/to/your/dataset
    SHARED_DIR=/path/to/shared
    OUTPUT_DIR=/path/to/output

    TOKEN= # put your token here
    N_SAMPLES=1.0

    docker run --gpus all --rm -it \
      -v ${INPUT_DIR}:/home/input_dir:ro  \
      -v ${SHARED_DIR}:/home/shared_dir:ro \
      -v ${OUTPUT_DIR}:/home/output_dir \
      lightly/sampling:latest \
      token=${TOKEN} \
      lightly.loader.num_workers=4     \
      stopping_condition.n_samples=${N_SAMPLES}\
      method=coreset \
      remove_exact_duplicates=True     \
      enable_corruptness_check=True \
      enable_training=True     \
      lightly.trainer.max_epochs=20 \
      dump_dataset=False \
      upload_dataset=False

Running it will create a terminal output similar to the following:

.. code-block::

    [2021-09-29 13:32:11] Loading initial dataset...
    [2021-09-29 13:32:11] Found 372 input images in input_dir.
    [2021-09-29 13:32:11] Lightly On-Premise License is valid
    [2021-09-29 13:32:11] Checking for corrupt images (disable with enable_corruptness_check=False).
    Corrupt images found: 0: 100%|██████████████████| 372/372 [00:01<00:00, 310.35it/s]
    [2021-09-29 13:32:14] Training self-supervised model.
    GPU available: True, used: True
    [2021-09-29 13:32:57,696][lightning][INFO] - GPU available: True, used: True
    TPU available: None, using: 0 TPU cores
    [2021-09-29 13:32:57,697][lightning][INFO] - TPU available: None, using: 0 TPU cores
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    [2021-09-29 13:32:57,697][lightning][INFO] - LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

      | Name      | Type       | Params
    -----------------------------------------
    0 | model     | SimCLR     | 11.2 M
    1 | criterion | NTXentLoss | 0
    -----------------------------------------
    11.2 M    Trainable params
    0         Non-trainable params
    [2021-09-29 13:34:29,772][lightning][INFO] - Saving latest checkpoint...
    Epoch 19: 100%|████████████████████████████████| 23/23 [00:04<00:00,  5.10it/s, loss=2.52, v_num=0]
    [2021-09-29 13:34:29] Embedding images.
    Compute efficiency: 0.90: 100%|█████████████████████████| 24/24 [00:01<00:00, 21.85it/s]
    [2021-09-29 13:34:31] Saving embeddings to output_dir/2021-09-29/13:32:11/data/embeddings.csv.
    [2021-09-29 13:34:31] Removing exact duplicates (disable with remove_exact_duplicates=False).
    [2021-09-29 13:34:31] Found 0 exact duplicates.
    [2021-09-29 13:34:31] Unique embeddings are stored in output_dir/2021-09-29/13:32:11/data/embeddings.csv
    [2021-09-29 13:34:31] Normalizing embeddings to unit length (disable with normalize_embeddings=False).
    [2021-09-29 13:34:31] Normalized embeddings are stored in output_dir/2021-09-29/13:32:11/data/normalized_embeddings.csv
    [2021-09-29 13:34:31] Sampling dataset with stopping condition: n_samples=372
    [2021-09-29 13:34:31] Skipped sampling because the number of remaining images is smaller than the number of requested samples.
    [2021-09-29 13:34:31] Writing report to output_dir/2021-09-29/13:32:11/report.pdf.
    [2021-09-29 13:35:04] Writing csv with information about removed samples to output_dir/2021-09-29/13:32:11/removed_samples.csv
    [2021-09-29 13:35:04] Done!

By running it, this will create an embeddings.csv file
in the output directory. Locate it and save the path to it.
E.g. It may be found under
`/path/to/output/2021-09-28/15:47:34/data/embeddings.csv`

It should look similar to this:

+----------------+--------------+--------------+--------------+--------------+---------+
| filenames      | embedding_0  | embedding_1  | embedding_2  | embedding_3  | labels  |
+================+==============+==============+==============+==============+=========+
| cats/0001.jpg  | 0.29625183   | 0.50055015   | 0.36491454   | 0.8156051    | 0       |
+----------------+--------------+--------------+--------------+--------------+---------+
| dogs/0005.jpg  | 0.36491454   | 0.29625183   | 0.38491454   | 0.36491454   | 1       |
+----------------+--------------+--------------+--------------+--------------+---------+
| cats/0014.jpg  | 0.8156051    | 0.59055015   | 0.29625183   | 0.50055015   | 0       |
+----------------+--------------+--------------+--------------+--------------+---------+


Step 2. Add your active learning scores as an additonal column.
---------------
If you want to use the predictions from your model as active learning scores,
you can use the `scorers from the lightly pip package <https://docs.lightly.ai/getting_started/active_learning.html#scorers>`_ .


.. code-block:: python

    # Have this in a step_2_add_al_scores.py

    from typing import Iterable
    import csv
    import os

    """
    Run your detection model here
    Use the scorers offered by lightly to generate active learning scores.
    """

    # Let's assume that you have one active learning score for every image.
    # WARNING: The order of the scores MUST match the order of filenames
    # in the embeddings.csv.
    scores: Iterable[float] =  # must be an iterable of floats,
    # e.g. a list of float or a 1d-numpy array

    # define the function to add the scores to the embeddings.csv
    def add_al_scores_to_csv(
            input_file_path: str, output_file_path: str,
            scores: Iterable[float], column_name: str = "al_score"
    ):
        with open(input_file_path, 'r') as read_obj:
            with open(output_file_path, 'w') as write_obj:
                csv_reader = csv.reader(read_obj)
                csv_writer = csv.writer(write_obj)

                # add the column name
                first_row = next(csv_reader)
                first_row.append(column_name)
                csv_writer.writerow(first_row)

                # add the scores
                for row, score in zip(csv_reader, scores):
                    row.append(str(score))
                    csv_writer.writerow(row)

    # use the function
    # adapt the following line to use the correct path to the embeddings.csv
    input_embeddings_csv = '/path/to/output/2021-07-28/12:00:00/data/embeddings.csv'
    output_embeddings_csv = input_embeddings_csv.replace('.csv', '_al.csv')
    add_al_scores_to_csv(input_embeddings_csv, output_embeddings_csv, scores)

    print("Use the following path to the embeddings_al.csv in the next step:")
    print(output_embeddings_csv)

Running it will create a terminal output similar to the following:

.. code-block::

    (base) user@machine:~/GitHub/playground/docker_with_al$ sudo python3 step_2_add_al_scores.py
    Use the following path to the embedding.csv in the next step:
    /path/to/output/2021-07-28/12:00:00/data/embeddings_al.csv

Your embeddings_al.csv should look similar to this:

+----------------+--------------+--------------+--------------+--------------+---------+-----------+
| filenames      | embedding_0  | embedding_1  | embedding_2  | embedding_3  | labels  | al_score  |
+================+==============+==============+==============+==============+=========+===========+
| cats/0001.jpg  | 0.29625183   | 0.50055015   | 0.36491454   | 0.8156051    | 0       | 0.7231    |
+----------------+--------------+--------------+--------------+--------------+---------+-----------+
| dogs/0005.jpg  | 0.36491454   | 0.29625183   | 0.38491454   | 0.36491454   | 1       | 0.91941   |
+----------------+--------------+--------------+--------------+--------------+---------+-----------+
| cats/0014.jpg  | 0.8156051    | 0.59055015   | 0.29625183   | 0.50055015   | 0       | 0.01422   |
+----------------+--------------+--------------+--------------+--------------+---------+-----------+


Step 3. Use the Lightly docker to perform a sampling on the scores.
---------------
Run the docker and use the generated embedding file from the last step.
Then perform an active learning sampling using the `CORAL` sampler.
E.g. use the following bash script.


.. code-block:: bash

    #!/bin/bash -e

    # Have this in a step_3_run_docker_coral.sh
    
    INPUT_DIR=/path/to/your/dataset/
    SHARED_DIR=/path/to/shared/
    OUTPUT_DIR=/path/to/output/
    
    EMBEDDING_FILE= # insert the path printed in the last step here.
    # e.g. /path/to/output/2021-07-28/12:00:00/data/embeddings_al.csv

    cp INPUT_EMBEDDING_FILE SHARED_DIR # copy the embedding file to the shared directory
    EMBEDDINGS_REL_TO_SHARED=embeddings_al.csv
    

    TOKEN= # put your token here
    N_SAMPLES= # Choose how many samples you want to use here, e.g. 0.1 for 10 percent.

    docker run --gpus all --rm -it \
        -v ${INPUT_DIR}:/home/input_dir:ro  \
        -v ${SHARED_DIR}:/home/shared_dir:ro \
        -v ${OUTPUT_DIR}:/home/output_dir \
        lightly/sampling:latest \
        token=${TOKEN} \
        lightly.loader.num_workers=4     \
        stopping_condition.n_samples=${N_SAMPLES}\
        method=coral \
        enable_training=False     \
        dump_dataset=True \
        upload_dataset=False \
        embeddings=${EMBEDDINGS_REL_TO_SHARED} \
        active_learning_score_column_name="al_score" \
        scorer=""
      
Your terminal output should look similar to this:

.. code-block::

    [2021-09-29 09:36:27] Loading initial embedding file...
    [2021-09-29 09:36:27] Output images will not be resized.
    [2021-09-29 09:36:27] Found 372 input images in shared_dir/embeddings_al.csv.
    [2021-09-29 09:36:27] Lightly On-Premise License is valid
    [2021-09-29 09:36:28] Removing exact duplicates (disable with remove_exact_duplicates=False).
    [2021-09-29 09:36:28] Found 0 exact duplicates.
    [2021-09-29 09:36:28] Unique embeddings are stored in shared_dir/embeddings_al.csv
    [2021-09-29 09:36:28] Normalizing embeddings to unit length (disable with normalize_embeddings=False).
    [2021-09-29 09:36:28] Normalized embeddings are stored in output_dir/2021-09-29/09:36:27/data/normalized_embeddings.csv
    [2021-09-29 09:36:28] Sampling dataset with stopping condition: n_samples=10
    [2021-09-29 09:36:28] Sampled 10 images.
    [2021-09-29 09:36:28] Writing report to output_dir/2021-09-29/09:36:27/report.pdf.
    [2021-09-29 09:36:56] Writing csv with information about removed samples to output_dir/2021-09-29/09:36:27/removed_samples.csv
    [2021-09-29 09:36:56] Done!
