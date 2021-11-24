#!/bin/bash
set -e

# Get the parameters
export TOKEN=$1
[[ -z "$TOKEN" ]] && { echo "Error: token is empty" ; exit 1; }
echo "############################### token: ${TOKEN}"

echo "############################### Download the clothing_dataset_small"

DIR_DATASET=clothing_dataset_small
if [ -d $DIR_DATASET ]; then
  echo "Skipping download of dataset, it already exists."
else
  git clone https://github.com/alexeygrigorev/clothing-dataset-small $DIR_DATASET
fi
INPUT_DIR="${DIR_DATASET}/test/dress"

# Run the tests
echo "############################### Test 1"
lightly-magic token=$TOKEN input_dir=$INPUT_DIR trainer.max_epochs=0 new_dataset_name=test_unmocked_cli_1

echo "############################### Test 2"
lightly-magic token=$TOKEN input_dir=$INPUT_DIR trainer.max_epochs=1 new_dataset_name=test_unmocked_cli_2

echo "############################### Test 3"
lightly-upload token=$TOKEN input_dir=$INPUT_DIR new_dataset_name=test_unmocked_cli_3
lightly-upload token=$TOKEN input_dir=$INPUT_DIR new_dataset_name=test_unmocked_cli_3

echo "############################### Test 4"
lightly-magic token=$TOKEN input_dir=$INPUT_DIR trainer.max_epochs=0 new_dataset_name=test_unmocked_cli_4
lightly-magic token=$TOKEN input_dir=$INPUT_DIR trainer.max_epochs=0 new_dataset_name=test_unmocked_cli_4

echo "############################### Test 5"
lightly-upload token=$TOKEN input_dir=$INPUT_DIR new_dataset_name=test_unmocked_cli_5 upload=metadata

echo "############################### Test 6"
lightly-upload token=$TOKEN input_dir=$INPUT_DIR new_dataset_name=test_unmocked_cli_6 upload=thumbnails

echo "############################### Deleting all datasets again"
NUMBER_OF_DATASETS=6
python tests/UNMOCKED_end2end_tests/delete_datasets_test_unmocked_cli.py $NUMBER_OF_DATASETS $TOKEN

echo "############################### Test active learning"
INPUT_DIR="${PWD}/clothing_dataset_small/test"
python tests/UNMOCKED_end2end_tests/test_api.py $INPUT_DIR $TOKEN


echo "############################### Delete dataset again"
rm -rf $DIR_DATASET