#!/bin/bash
set -e

# Get the parameters
export LIGHTLY_TOKEN=$1
[[ -z "$LIGHTLY_TOKEN" ]] && { echo "Error: token is empty" ; exit 1; }
echo "############################### token: ${LIGHTLY_TOKEN}"

DATE_TIME=$(date +%Y-%m-%d-%H-%M-%S)

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

# Run the tests
echo "############################### Test 1"
lightly-magic input_dir=$INPUT_DIR trainer.max_epochs=0

echo "############################### Test 2"
lightly-magic input_dir=$INPUT_DIR trainer.max_epochs=1

echo "############################### Delete dataset again"
rm -rf $DIR_DATASET
