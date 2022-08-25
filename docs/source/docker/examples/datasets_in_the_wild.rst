Extract Diverse Video Frames
=============================

The following example is a showcase how the Lightly Worker can be used 
to extract frames from a video based on their uniqueness rather than based on timestamps.

.. note:: For all examples we assume that the Lightly Worker is configured and running. See :ref:`docker-setup` for more information.


Using ffmpeg
------------

Using tools such as ffmpeg we can extract frames from a video 
using a simple one-liner like this:

.. code-block:: console

    # extract all frames from video.mp4 as .png files and store in frames/ folder
    ffmpeg -i video.mp4 frames/%d.png

ffmpeg allows us to use various flags to choose framerate, crop the images, 
resize the images or set the quality as shown here:

.. code-block:: console

    # set framerate to 5 fps
    ffmpeg -i video.mp4 -filter:v "fps=5" frames/%d.png

    # resize image to 256x256 pixels
    ffmpeg -i video.mp4 -s 256x256 frames/%d.png

    # extract frames as .jpg files
    # high quality jpg compression
    ffmpeg -i video.mp4 -qscale:v 1 frames/%d.jpg

    # lower quality jpg compression
    ffmpeg -i video.mp4 -qscale:v 5 frames/%d.jpg

    # crop a 480x480 image with 80 pixels offset in x direction
    ffmpeg -i video.mp4 -filter:v "crop=480:480:80:0" frames/%d.png

    # and many more ...

However, the problem is the extracted frames sum up and use lots of storage.
For most training tasks, we don't even want to extract all the frames. Limiting
the framerate is very easy and helps us reduce the amount of extracted data. 
On the other hand, even a video with 5 fps might contain lots of similar frames
or even worse, we might miss some frames with lots of "action". 

Using the Lightly Worker
------------------------

The Lightly Worker has been designed to give engineers an alternative to using
fixed framerates for frame extraction. 

How about selecting frames based on their similarity? 

In this example, we use the following video: https://www.pexels.com/de-de/video/3719157/

We store the video in a storage bucket, e.g. under *s3://dataset/video/*. We can use wget in 
a terminal under linux or MacOS to download the video and then either upload it via drag and drop
or with the `aws cli <https://aws.amazon.com/cli/>`_.


Now, let's extract 99 frames using the Lightly Worker. We start by creating a dataset and configuring the S3 bucket as 
a datasource. We call the dataset `frame-extraction-example` and use the input type `VIDEOS`. We configure the datasource to point at `s3://dataset/video/`.

.. code-block:: python

  from lightly.api import ApiWorkflowClient
  from lightly.openapi_generated.swagger_client.models.dataset_type import DatasetType
  from lightly.openapi_generated.swagger_client.models.datasource_purpose import DatasourcePurpose

  # Create the Lightly client to connect to the API.
  client = ApiWorkflowClient(token="MY_AWESOME_TOKEN")

  # Create a new dataset on the Lightly Platform.
  client.create_new_dataset_with_unique_name(
      'frame-extraction-example',
      DatasetType.VIDEOS,
  )

  ## AWS S3
  # Input bucket
  client.set_s3_config(
      resource_path="s3://dataset/video/",
      region='eu-central-1',
      access_key='S3-ACCESS-KEY',
      secret_access_key='S3-SECRET-ACCESS-KEY',
      purpose=DatasourcePurpose.INPUT
  )
  # Output bucket
  client.set_s3_config(
      resource_path="s3://output/",
      region='eu-central-1',
      access_key='S3-ACCESS-KEY',
      secret_access_key='S3-SECRET-ACCESS-KEY',
      purpose=DatasourcePurpose.LIGHTLY
  )


Next, we schedule a job which extracts 99 frames with a strategy to
select a diverse set of frames:


.. code-block:: python

  client.schedule_compute_worker_run(
      worker_config={
          "enable_corruptness_check": True,
          "remove_exact_duplicates": True
      },
      selection_config = {
          "n_samples": 99,
          "strategies": [
              {
                  "input": {
                      "type": "EMBEDDINGS"
                  },
                  "strategy": {
                      "type": "DIVERSIFY"
                  }
              }
          ]
      }
  )

The extracted frames can now be found in the output bucket (`s3://output`) and can easily be accessed from the `Lightly Platform <https://app.lightly.ai>`_.


For comparison, we extracted frames from the video using ffmpeg with the following command:

.. code-block:: console

    ffmpeg -i raw/video.mp4 -filter:v "fps=5" frames_ffmpeg/%d.png


The table below shows a comparison of the different extraction methods:

.. list-table::
   :widths: 50 50 50 50 50
   :header-rows: 1

   * - Metric
     - original dataset
     - after ffmpeg
     - after random
     - after coreset
   * - Number of Samples
     - 475
     - 99
     - 99
     - 99
   * - L2 Distance (Mean)
     - 1.2620
     - 1.2793
     - 1.2746
     - 1.3711
   * - L2 Distance (Min)
     - 0.0000
     - 0.0000
     - 0.0586
     - 0.2353
   * - L2 Distance (Max)
     - 1.9835
     - 1.9693
     - 1.9704
     - 1.9470
   * - L2 Distance (10th Percentile)
     - 0.5851
     - 0.5891
     - 0.5994
     - 0.8690
   * - L2 Distance (90th Percentile)
     - 1.8490
     - 1.8526
     - 1.8525
     - 1.7822


We notice the following when looking at this table:

- The **min distance** between two samples was 0 after ffmpeg selection whereas the
  min distance significantly increased using coreset selection strategy.

  - 0 distance means that there are at least two samples completely identical
    (e.g. two frames in the video are the same)

- The **mean distance** between the original dataset, ffmpeg, as well as 
  random selection, is very similar. The coreset selection however differs
  significantly with a higher mean (higher diversity) in the selected dataset.

- The **10th percentile** shows similar behavior to the mean distance.

As you see in this example just selecting every N-th frame is similar to
selecting frames randomly. More sophisticated selection strategies, such as the coreset selection strategy, result in
much higher sample diversity. The docker has been optimized for these selection strategies.


.. note:: Note that by default the embeddings of the dataset will be normalized
          to unit vector length. Max L2 distance between two vectors is 
          therefore 2.0 (two vectors pointing in opposite directions). 


Now let's take a look at the storage requirements. If we would extract all frames from the video
and then run a selection algorithm on them we would need 553.4 MBytes. However, the Lightly Worker
can process the video directly so we require only 6.4 MBytes of storage. This means it requires 70x less storage!


.. list-table::
   :widths: 50 50 50 30
   :header-rows: 1

   * - Metric
     - ffmpeg extracted frames
     - Lightly using video
     - Reduction
   * - Storage Consumption
     - 447 MBytes + 6.4 MBytes
     - 6.4 MBytes
     - 70.84x

.. note:: Why not extract the frames as compressed .jpg images? Extracting the 
          frames as .jpg would indeed reduce storage consumption. The video from 
          our example would end up using (14 MBytes + 6.4 MBytes). However, for 
          critical applications where robustness and accuracy of the model are 
          key, we have to think about the final system in production. Is your 
          production system working with the raw camera stream (uncompressed) or 
          with compressed frames (e.g. .jpg)? Very often we don’t have time to 
          compress a frame in real-time systems or don’t want to introduce 
          compression artifacts. You should also think about whether you want 
          to train a model on compressed data whereas in production is runs 
          using raw data.
