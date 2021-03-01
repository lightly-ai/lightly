INPUT_DIR="/Users/malteebnerlightly/Documents/datasets/clothing-dataset-small-master/test"
export LIGHTLY_SERVER_LOCATION="https://app-dev.lightly.ai/"
# create a dataset at https://app-dev.lightly.ai/
TOKEN="f9b60358d529bdd824e3c2df"
DATASET_ID="603917b420b7700032301176"

# install lightly from a branch
pip uninstall lightly --yes
pip install "git+https://github.com/lightly-ai/lightly.git@develop_active_learning"


# basic tests
lightly-train --help
lightly-embed --help
lightly-upload --help
lightly-download --help

# train a model for 1 epoch and then use it to embed
lightly-train input_dir=$INPUT_DIR trainer.max_epochs=1 loader.num_workers=6
lightly-embed input_dir=$INPUT_DIR checkpoint=mycheckpoint.ckpt embeddings=test_embeddings.csv
# upload the dataset and embeddings
lightly-upload input_dir=$INPUT_DIR token=$TOKEN dataset_id=$DATASET_ID
lightly-upload input_dir=$INPUT_DIR token=$TOKEN dataset_id=$DATASET_ID embeddings=test_embeddings.csv

# perform a sampling on the server -> tag_name_subsampled

# download it
lightly-download token=$TOKEN dataset_id=$DATASET_ID


