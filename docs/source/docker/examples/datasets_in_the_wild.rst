Extract Diverse Video Frames
=============================

The following example is a showcase how the lightly docker solution can be used 
to extract frames from a video based on their uniqueness 
rather than based on timestamps.


Using ffmpeg
-----------------------------

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

Using Lightly Docker
-----------------------------

Lightly Docker has been designed to give engineers an alternative to using
fixed framerates for frame extraction. 

How about selecting frames based on their similarity? 

In this example, we use the following video: https://www.pexels.com/de-de/video/3719157/

We download the video to a local folder */dataset/video/*. We can use wget in 
a terminal under linux or MacOS to download the video (just make sure you 
navigated to the directory where you want to download the video to).

Let us extract frames from the video using ffmpeg. We want to get 5 frames per
second (fps). Create a new directory called */dataset/frames_ffmpeg/*. Using ffmpeg we can 
extract the frames with the following command:

.. code-block:: console

    ffmpeg -i raw/video.mp4 -filter:v "fps=5" frames_ffmpeg/%d.png


Extracting the frames without introducing compression artifacts is using lots of 
storage. In this example, we have a small video of 6.4 MBytes. Once extracted, 
the .png frames together with the video consume 553.4 MBytes. This is a 
70x increase!

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

Now we want to do the same using Lightly Docker. Since the ffmpeg command
extracted 99 frames let's extract 99 frames as well:

.. code-block:: console

    docker run --gpus all --rm -it -v /dataset/video/:/home/input_dir:ro \
        -v \/datasets/videos/docker_out:/home/output_dir \
        -v /datasets/docker_shared_dir:/home/shared_dir -e --ipc="host" \
        --network="host" lightly/sampling:latest token=MYAWESOMETOKEN \
        lightly.collate.input_size=64 lightly.loader.batch_size=32 '
        lightly.loader.num_workers=8 lightly.trainer.max_epochs=10 \
        stopping_condition.n_samples=100 remove_exact_duplicates=True \
        enable_corruptness_check=False enable_training=False dump_dataset=True \
        method=coreset

To perform a random selection we can simply replace "coreset" with "random" as
our selected method. Note that coreset is the default method.

Let's have a look at some statistics of the two obtained datasets:

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
  RANDOM selection, is very similar. The coreset selection however differs
  significantly with a higher mean (higher diversity) in the selected dataset.

- The **10th percentile** shows similar behavior to the mean distance.

As you see in this example just selecting every N-th frame is similar to
selecting frames randomly. More sophisticated selection methods, such as the coreset which has been optimized for Lightly Docker, result in
much higher sample diversity.

.. note:: Note that by default the embeddings of the dataset will be normalized
          to unit vector length. Max L2 distance between two vectors is 
          therefore 2.0 (two vectors pointing in opposite directions). 