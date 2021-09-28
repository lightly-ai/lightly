"""

.. _docker-active_learning:

Documentation: Use the docker for active learning
==============================================

For running an active learning step with the Lightly docker, we need to perform
3 steps:

1. Create an embeddings.csv file with the Lightly docker.
2. Add your active learning scores as an additonal column.
3. Use the Lightly docker to perform a sampling on the scores.



# Step 1:
Run the docker as usual and as described in the getting started section.
The only difference is that you set the number of samples to be sampled to 1,
this prevents sampling.
E.g. create and run a bash script with the following content:

.. code::
    INPUT_DIR=/path/to/your/dataset
    SHARED_DIR=/path/to/shared
    OUTPUT_DIR=/path/to/output

    TOKEN=YOUR_TOKEN # put your token here
    N_SAMPLES=1

    docker run --ulimit nofile=32:32 --gpus all --rm -it \
      -v ${INPUT_DIR}:/home/input_dir:ro  \
      -v ${SHARED_DIR}:/home/shared_dir:ro \
      -v ${OUTPUT_DIR}:/home/output_dir \
      --ipc="host" --network="host" \
      lightly/sampling:latest \
      token=${TOKEN} \
      lightly.loader.num_workers=4     \
      stopping_condition.n_samples=${N_SAMPLES}\
      method=coreset \
      remove_exact_duplicates=True     \
      enable_corruptness_check=True \
      enable_training=True     \
      lightly.trainer.max_epochs=100 \
      dump_dataset=False \
      upload_dataset=False

After running it, this will create a embeddings.csv file
in the output directory. Locate it and save the path to it.
E.g. It may be found under
`/path/to/output/2021-07-28/12:00:00/data/embeddings.cv`

"""
# %%
# # Step 2:  Add active learning scores to the embeddings.cv file.
# -----------------
#
# If you want to use your use predictions from your model as active learning scores,
# you can use the Scorers from the lightly pip package.


from typing import Iterable
import csv
import os

"""
Run your detection model here
Use the scorers offered by lightly to generate active learning scores.
"""

# Let's assume that you have one active learning score for every image:
# The order of the scores must match the order of filenames
# in the embeddings.csv.
scores: Iterable[float]

# define the function to add the scores to the embeddings.cv
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
# adapt the following line to use the correct path to the embedding.csv
input_embeddings_csv = '/path/to/output/2021-07-28/12:00:00/data/embeddings.csv'
output_embeddings_csv = input_embeddings_csv.replace('.csv', '_al.csv')
add_al_scores_to_csv(input_embeddings_csv, output_embeddings_csv, scores)

print("Use the following path to the embedding.csv in the next step:")
print(output_embeddings_csv)

'''

# Step 3:
Run the docker and use the generated embedding file from the last step.
E.g. use the following bash script.


.. code::
    #!/bin/bash -e
    
    INPUT_DIR=/path/to/your/dataset/
    SHARED_DIR=/path/to/shared/
    OUTPUT_DIR=/path/to/output/
    
    EMBEDDING_FILE = # insert the path printed in the last step here.
    cp INPUT_EMBEDDING_FILE SHARED_DIR # copy the embedding file to the shared directory
    EMBEDDINGS_REL_TO_SHARED = embeddings_al.csv
    

    TOKEN= # put your token here
    N_SAMPLES= # Choose how many samples you want to use here, e.g. 0.1 for 10 percent.

    docker run --ulimit nofile=32:32 --gpus all --rm -it \
      -v ${INPUT_DIR}:/home/input_dir:ro  \
      -v ${SHARED_DIR}:/home/shared_dir:ro \
      -v ${OUTPUT_DIR}:/home/output_dir \
      --ipc="host" --network="host" \
      lightly/sampling:latest \
      token=${TOKEN} \
      lightly.loader.num_workers=4     \
      stopping_condition.n_samples=${N_SAMPLES}\
      method=coral \
      enable_training=False     \
      dump_dataset=True \
      upload_dataset=False \
      embeddings=${EMBEDDINGS_REL_TO_SHARED} \
      active_learning_score_column_name="al_score"
      

'''
