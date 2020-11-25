Datasets in the Wild
=============================

Extract Diverse Video Frames
-----------------------------

Lots of data we use to train computer vision models has been collected using 
video cameras. Since we don't work with video files directly due to various 
reasons (performance, usability) we extract the invidividual frames first.

Using ffmpeg
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using tools such as ffmpeg this can be done using a simple one liner like this:

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
    # high quality
    ffmpeg -i video.mp4 -qscale:v 1 frames/%d.jpg

    # lower quality
    ffmpeg -i video.mp4 -qscale:v 5 frames/%d.jpg

    # crop a 480x480 image with 80 pixels offset in x direction
    ffmpeg -i video.mp4 -filter:v "crop=480:480:80:0" frames/%d.png

    # and many more ...

However, the problem is the extracted frames sum up and use lots of storage.
For most training tasks we don't even want to extract all the frames. Limiting
the framerate is very easy and helps us reduce the amount of extracted data. 
On the other hand, even a video with 5 fps might contain lots of similar frames
or even worse, might miss some important frames with lots of "action". 

Using Lightly Docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Lightly Docker has been designed to give engineers an alternative to using
fixed framerates for frame extraction. 

How about selecting frames based on their similarity? 

In this example we use the following video: 

We download the video to a local folder */dataset/video/*. We can use wget in 
a terminal under linux or MacOS to download the video (just make sure you 
navigated to the directory where you want to download the video to):

.. code-block:: console

    wget url


Let us extract frames from the video using ffmpeg. We want to get 5 frame per
second (fps). Create a new directory called */dataset/frames_ffmpeg/*. Using ffmpeg we can 
extract the frames with the following command:

.. code-block:: console

    ffmpeg -i raw/video.mp4 -filter:v "fps=5" frames_ffmpeg/%d.png


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
        enable_corruptness_check=False enable_training=False dump_dataset=True
