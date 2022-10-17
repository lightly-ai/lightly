import os
import json

import numpy as np

TASK_NAME = "lightly_semantic_segmentation"
CATEGORIES = ["background", "car", "person"]


def get_dummy_prediction(height: int = 500, width: int = 500):
    """Returns a dummy prediction of shape h x w x n_classes.

    Height and width are in pixels.
    """
    return np.random.rand(height, width, len(CATEGORIES))


def filename_to_json(filename: str):
    """Turns an image filename into the respective json filename."""
    root, _ = os.path.splitext(filename)
    return f"{root}.json"


def binary_to_rle(binary_mask: np.ndarray) -> np.ndarray:
    """Converts a binary segmentation mask to RLE."""
    # Flatten mask and add -1 at beginning and end of array
    flat = np.concatenate(([-1], np.ravel(binary_mask), [-1]))
    # Find indices where a change to 0 or 1 happens
    borders = np.nonzero(np.diff(flat))[0]
    # Find counts of subsequent 0s and 1s
    rle = np.diff(borders)
    if flat[1]:
        # The first value in the encoding must always be the count
        # of initial 0s. If the mask starts with a 1 we must set
        # this count to 0.
        rle = np.concatenate(([0], rle))
    return rle


def convert_to_lightly_prediction(filename: str, seg_map: np.ndarray):
    """Converts a segmentation map of shape W x H x C to Lightly format."""
    seg_map_argmax = np.argmax(seg_map, axis=-1)

    prediction = {"file_name": filename, "predictions": []}
    for category_id in np.unique(seg_map_argmax):
        rle = binary_to_rle(seg_map_argmax == category_id)
        logits = np.mean(seg_map[seg_map_argmax == category_id], axis=0)
        assert np.argmax(logits) == category_id
        probabilities = np.exp(logits) / np.sum(np.exp(logits))
        assert abs(np.sum(probabilities) - 1.0) < 1e-6

        prediction["predictions"].append(
            {
                "category_id": int(category_id),
                "segmentation": [int(r) for r in rle],
                "score": float(probabilities[category_id]),
                "probabilities": [float(p) for p in probabilities],
            }
        )

    return prediction


# The following code will generate a tasks.json, a schema.json, and a dummy
# prediction file called my_image.json. To use them with the Lightly worker,
# arrange them as follows in a .lightly directory
#
# .lightly/
#  L predictions/
#     L tasks.json
#     L lightly_semantic_segmentation/
#        L schema.json
#        L // add the real prediction files here
#


# add tasks.json
tasks = [TASK_NAME]
with open("tasks.json", "w") as f:
    json.dump(tasks, f)

# add schema.json
schema = {
    "task_type": "semantic-segmentation",
    "categories": [
        {
            "id": i,
            "name": name,
        }
        for i, name in enumerate(CATEGORIES)
    ],
}
with open("schema.json", "w") as f:
    json.dump(schema, f)

# generate a dummy prediction
filename = "my_image.png"
prediction = get_dummy_prediction()  # this is a h x w x n_classes numpy array
category_ids = np.argmax(prediction, axis=-1)

lightly_prediction = {"file_name": filename, "predictions": []}
for category_id in np.unique(category_ids):
    # get the run-length encoding
    rle = binary_to_rle(category_ids == category_id)
    # get the logits
    logits = np.mean(prediction[category_ids == category_id], axis=0)
    assert np.argmax(logits) == category_id
    probabilities = np.exp(logits) / np.sum(np.exp(logits))
    assert abs(np.sum(probabilities) - 1.0) < 1e-6

    lightly_prediction["predictions"].append(
        {
            "category_id": int(category_id),
            "segmentation": [int(r) for r in rle],
            "score": float(probabilities[category_id]),
            "probabilities": [float(p) for p in probabilities],
        }
    )

with open(filename_to_json(filename), "w") as f:
    json.dump(lightly_prediction, f)
