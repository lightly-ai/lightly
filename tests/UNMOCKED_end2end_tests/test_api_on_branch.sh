#!/bin/bash
set -e

cd ../../../lightly # ensure you are in the top directory

# install lightly from this branch
pip uninstall -y lightly
pip install . --use-feature=in-tree-build

# Get the parameters
export INPUT_DIR=$1
export TOKEN=$2
echo "input_dir: ${INPUT_DIR}"
echo "token: ${TOKEN}"

# Run the tests
echo "############################### Test 1"
lightly-magic token=${TOKEN} input_dir=${INPUT_DIR} trainer.max_epochs=0 new_dataset_name=test_unmocked_cli_1

echo "############################### Test 2"
lightly-magic token=${TOKEN} input_dir=${INPUT_DIR} trainer.max_epochs=1 new_dataset_name=test_unmocked_cli_2

echo "############################### Test 3"
lightly-upload token=${TOKEN} input_dir=${INPUT_DIR} new_dataset_name=test_unmocked_cli_3
lightly-upload token=${TOKEN} input_dir=${INPUT_DIR} new_dataset_name=test_unmocked_cli_3

echo "############################### Test 4"
lightly-magic token=${TOKEN} input_dir=${INPUT_DIR} trainer.max_epochs=0 new_dataset_name=test_unmocked_cli_4
lightly-magic token=${TOKEN} input_dir=${INPUT_DIR} trainer.max_epochs=0 new_dataset_name=test_unmocked_cli_4

echo "############################### Test 5"
lightly-upload token=${TOKEN} input_dir=${INPUT_DIR} new_dataset_name=test_unmocked_cli_5 upload=metadata

echo "############################### Test 6"
lightly-upload token=${TOKEN} input_dir=${INPUT_DIR} new_dataset_name=test_unmocked_cli_6 upload=thumbnails




echo "SUCCESS!SUCCESS!SUCCESS!SUCCESS!SUCCESS!SUCCESS!SUCCESS!"



