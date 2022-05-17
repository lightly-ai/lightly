.. _ref-docker-datasource-metadata:

Add Metadata to a Datasource
===============================

Lightly can make use of metadata collected alongside your images or videos. Provided
metadata can be used to steer the selection process and to analyze the selected subset
in the web-app.


Metadata Folder Structure
----------------------------

In the following, we will outline the format in which metadata can be added to a
Lightly datasource. Everything regarding metadata will take place in a subdirectory
of your configured datasource called `.lightly/metadata`. The general structure
of this directory will look like this:


.. code-block:: bash

    datasource/my_dataset
        + image_1.png
        + image_2.png
        + ...
        + image_N.png
        + .lightly/metadata
            + schema.json
            + image_1.json
            ...
            + image_N.json


All of the `.json` files are explained in the next sections.




Metadata Schema
---------------
The schema defines the format of the metadata and helps the Lightly Platform to correctly identify 
and display different types of metadata.

You can provide this information to Lightly by adding a `schema.json` to the 
`.lightly/metadata` directory. The `schema.json` file must contain a list of
configuration entries. Each of the entries is a dictionary with the following keys:

 - `name`: Identifier of the metadata in the UI.
 - `path`: Concatenation of the keys to access the metadata in a dictionary.
 - `defaultValue`: The fallback value if there is no metadata available.
 - `valueDataType`: One of

   - `NUMERIC_INT`
   - `NUMERIC_FLOAT`
   - `CATEGORICAL_INT`
   - `CATEGORICAL_STRING`
   - `CATEGORICAL_BOOLEAN`


For example, let's say we have additional information about the weather for each
of the images we have collected. A possible schema could look like this:

.. code-block:: javascript
    :caption: .lightly/metadata/schema.json

    [
        {
            "name": "Is special frame",
            "path": "special_frame_flag",
            "defaultValue": false,
            "valueDataType": "CATEGORICAL_BOOLEAN"
        },
        {
            "name": "Weather description",
            "path": "weather.description",
            "defaultValue": "nothing",
            "valueDataType": "CATEGORICAL_STRING"
        },
        {
            "name": "Temperature",
            "path": "weather.temperature",
            "defaultValue": 0.0,
            "valueDataType": "NUMERIC_FLOAT"
        },
        {
            "name": "Air pressure",
            "path": "weather.air_pressure",
            "defaultValue": 0,
            "valueDataType": "NUMERIC_INT"
        },
        {
            "name": "Vehicle ID",
            "path": "vehicle_id",
            "defaultValue": 0,
            "valueDataType": "CATEGORICAL_INT"
        }
    ]




Metadata Files
--------------
Lightly requires a single metadata file per image or video. If a metadata file is provided
for a full video, Lightly assumes that the metadata is valid for all frames in that video.

To provide metadata for an image or a video, place a metadata file with the same name
as the image or video in the `.lightly/metadata` directory but change the file type to
`.json`. The file should contain the metadata in the format defined under :ref:`ref-metadata-format`.


.. code-block:: bash

    # filename of the metadata for file FILENAME.EXT
    .lightly/metadata/${FILENAME}.json

    # example: my_image.png
    .lightly/metadata/my_image.json

    # example: my_video.mp4
    .lightly/metadata/my_video.json


When working with videos it's also possible to provide metadata on a per-frame basis.
Then, Lightly requires a metadata file per frame. Lightly uses a naming convention to
identify frames: The filename of a frame consists of the video filename, the video format,
and the frame number (padded to the length of the number of frames in the video) separated
by hyphens. For example, for a video with 200 frames, the frame number will be padded
to length three. For a video with 1000 frames, the frame number will be padded to length four (99 becomes 0099).


.. code-block:: bash

    # filename of the metadata of the Xth frame of video FILENAME.EXT
    # with 200 frames (padding: len(str(200)) = 3)
    .lightly/metadata/${FILENAME}-${X:03d}-${EXT}.json

    # example: my_video.mp4, frame 99/200
    .lightly/metadata/my_video-099-mp4.json

    # example: my_subdir/my_video.mp4, frame 99/200
    .lightly/metadata/my_subdir/my_video-099-mp4.json


.. _ref-metadata-format:

Metadata Format
---------------

Metadata for images or videos must have a `file_name`, `type`, and `metadata`` key.
Here, `file_name`` serves as a unique identifier to retrieve the original file for which the metadata was collected,
`type` indicates whether the metadata is per "video", "frame", or "image", and `metadata` contains the actual metadata.

For our example from above, a corresponding metadata file should look like this:

.. code-block:: javascript
    :caption: .lightly/metadata/my_video.json

    {
        "file_name": "my_video.mp4",
        "type": "video",
        "metadata": {
            "weather": {
                "description": "sunny",
                "temperature": 23.2,
                "air_pressure": 1
            },
            "vehicle_id": 321,
        }
    }


Next Steps
----------

If metadata is provided, the Lightly worker will automatically detect and load it into
the web-app where it can be visualized and analyzed after running a selection.
