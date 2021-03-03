# install lightly from a branch
pip uninstall lightly --yes
pip install "git+https://github.com/lightly-ai/lightly.git@develop_active_learning"

# basic tests
lightly-train --help
lightly-embed --help
lightly-upload --help
lightly-magic --help
lightly-download --help

# test on a real dataset
#git clone https://github.com/alexeygrigorev/clothing-dataset-small clothing_dataset_small

# test with unnested input dir
INPUT_DIR_1="clothing_dataset_small/test/dress"
lightly-train input_dir=$INPUT_DIR_1 trainer.max_epochs=1 loader.num_workers=6
lightly-embed input_dir=$INPUT_DIR_1


