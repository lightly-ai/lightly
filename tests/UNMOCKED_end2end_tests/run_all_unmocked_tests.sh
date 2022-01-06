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
CUSTOM_METADATA_FILENAME="${DIR_DATASET}/custom_metadata.json"
python tests/UNMOCKED_end2end_tests/create_custom_metadata_from_input_dir.py $INPUT_DIR $CUSTOM_METADATA_FILENAME

NUMBER_OF_DATASETS=0
# Run the tests
echo "############################### Test 1"
lightly-magic token=$TOKEN input_dir=$INPUT_DIR trainer.max_epochs=0 new_dataset_name=test_unmocked_cli_1
((NUMBER_OF_DATASETS=NUMBER_OF_DATASETS+1))

echo "############################### Test 2"
lightly-magic token=$TOKEN input_dir=$INPUT_DIR trainer.max_epochs=1 new_dataset_name=test_unmocked_cli_2
((NUMBER_OF_DATASETS=NUMBER_OF_DATASETS+1))

echo "############################### Test 3"
lightly-upload token=$TOKEN input_dir=$INPUT_DIR new_dataset_name=test_unmocked_cli_3
lightly-upload token=$TOKEN input_dir=$INPUT_DIR new_dataset_name=test_unmocked_cli_3
((NUMBER_OF_DATASETS=NUMBER_OF_DATASETS+1))

echo "############################### Test 4"
lightly-magic token=$TOKEN input_dir=$INPUT_DIR trainer.max_epochs=0 new_dataset_name=test_unmocked_cli_4
lightly-magic token=$TOKEN input_dir=$INPUT_DIR trainer.max_epochs=0 new_dataset_name=test_unmocked_cli_4
((NUMBER_OF_DATASETS=NUMBER_OF_DATASETS+1))

echo "############################### Test 5"
lightly-upload token=$TOKEN input_dir=$INPUT_DIR new_dataset_name=test_unmocked_cli_5 upload=metadata
((NUMBER_OF_DATASETS=NUMBER_OF_DATASETS+1))

echo "############################### Test 6"
lightly-upload token=$TOKEN input_dir=$INPUT_DIR new_dataset_name=test_unmocked_cli_6 upload=thumbnails
((NUMBER_OF_DATASETS=NUMBER_OF_DATASETS+1))

echo "############################### Test 7"
lightly-upload token=$TOKEN input_dir=$INPUT_DIR new_dataset_name=test_unmocked_cli_7 upload=metadata custom_metadata=$CUSTOM_METADATA_FILENAME
((NUMBER_OF_DATASETS=NUMBER_OF_DATASETS+1))

echo "############################### Test 8"
lightly-upload token=$TOKEN input_dir=$INPUT_DIR new_dataset_name=test_unmocked_cli_8 upload=metadata
lightly-upload token=$TOKEN new_dataset_name=test_unmocked_cli_8 custom_metadata=$CUSTOM_METADATA_FILENAME
((NUMBER_OF_DATASETS=NUMBER_OF_DATASETS+1))


echo "############################### Deleting all datasets again"
python tests/UNMOCKED_end2end_tests/delete_datasets_test_unmocked_cli.py $NUMBER_OF_DATASETS $TOKEN

echo "############################### Test active learning"
INPUT_DIR="${PWD}/clothing_dataset_small/test"
python tests/UNMOCKED_end2end_tests/test_api.py $INPUT_DIR $TOKEN


echo "############################### Delete dataset again"
rm -rf $DIR_DATASET