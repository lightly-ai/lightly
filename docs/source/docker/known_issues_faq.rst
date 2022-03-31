.. _rst-docker-known-issues-faq:

Known Issues and FAQ
===================================

Docker is slow when working with long videos
---------------------------------------------------

We are working on this issue internally. For now we suggest to split the large
videos into chunks. You can do this using ffmpeg and without losing quality.
The following code just breaks up the video in a way that no re-encoding is needed.

.. code-block:: console

    ffmpeg -i input.mp4 -c copy -map 0 -segment_time 01:00:00 -f segment -reset_timestamps 1 output%03d.mp4

What exactly happens here?

- `input.mp4`, this is your input video
- `-c copy -map 0`, this makes sure we just copy and don't re-encode the video
- `-segment_time 01:00:00 -f segment`, defines that we want chunks of 1h each
- `-reset_timestamps 1`, makes sure we reset the timestamps (each video starts from 0)
- `output%03d.mp4`, name of the output vidoes (output001.mp4, output002.mp4, ...)

Docker Crashes when running with GPUs
-------------------------------------

You run the docker with `--gpus all` and encounter the following error?

.. code-block:: console

    Error response from daemon: could not select device driver "" with capabilities: [[gpu]].

This error might be caused because your docker installation does not support GPUs.

Try to install `nvidia-docker` following the guide 
`here <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker>`_.


Shared Memory Error when running Lightly Docker
-----------------------------------------------

The following error message appears when the docker runtime has not enough
shared memory. By default Docker uses 64 MBytes. However, when using multiple 
workers for data fetching `lightly.loader.num_workers` there might be not enough.

.. code-block:: console

    ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).                                                                                                
    Traceback (most recent call last):                                                                                                                                                                          
    File "/opt/conda/envs/env/lib/python3.7/multiprocessing/queues.py", line 236, in _feed                                                                                                                    
        obj = _ForkingPickler.dumps(obj)                                                                                                                                                                        
    File "/opt/conda/envs/env/lib/python3.7/multiprocessing/reduction.py", line 51, in dumps                                                                                                                  
        cls(buf, protocol).dump(obj)                                                                                                                                                                            
    File "/opt/conda/envs/env/lib/python3.7/site-packages/torch/multiprocessing/reductions.py", line 321, in reduce_storage                                                                                   
        fd, size = storage._share_fd_()                                                                                                                                                                         
    RuntimeError: unable to write to file </torch_31_1030151126> 

To solve this problem we need to increase the shared memory for the docker runtime.

You can change the shared memory to 512 MBytes by adding `--shm-size="512m"` to 
the docker run command:

.. code-block:: console

    # example of docker run with setting shared memory to 512 MBytes
    docker run --shm-size="512m" --gpus all


Docker crashes because of too many open files
-----------------------------------------------

The following error message appears when the docker runtime has not enough
file handlers. By default Docker uses 1024. However, when using multiple
workers for data fetching `lightly.loader.num_workers` this might be not
enough. As file handlers are used at many different parts of the code,
the actual error message may differ. Internet connections like for
connecting to the Lightly API also use file handlers.

.. code-block:: console

    <Error [Errno 24] Too many open files>

To solve this problem we need to increase the number of file handlers for the
docker runtime.

You can change the number of file handlers to 90000 by adding
`--ulimit nofile=90000:90000` to the docker run command:

.. code-block:: console

    # example of docker run with 90000 file handlers
    docker run --ulimit nofile=90000:90000 --gpus all

More documentation on docker file handlers is providided `here.
<https://docs.docker.com/engine/reference/commandline/run/#set-ulimits-in-container---ulimit>`_


Permission denied for input created with sudo
-----------------------------------------------

There are some problems if the input directory was created with root/ sudo and
the container tries to access it. This can be solved by making the files readable:

.. code-block:: console

    # make subdirectories browsable
    find MY_INPUT_DIR -type d -exec chmod 755 {} +

    # make the files themselves readable
    find MY_INPUT_DIR -type f -exec chmod 644 {} +


Error when using S3 fuse and mounting to docker
------------------------------------------------

If you use docker in combination with S3 fuse you might stumble across an issue 
that the docker container can't create the mount path for the input directory.

.. code-block:: console

    docker: Error response from daemon: error while creating mount source path \
    '/home/ubuntu/mydataset/': mkdir /home/ubuntu/mydataset: file exists.

You can resolve this problem by following the guide here: 
https://stackoverflow.com/a/61686833

1. uncomment **user_allow_other** option in the **/etc/fuse.conf** file
2. when you mount the bucket using s3fs use the **-o allow_other** option. 
   
   .. code-block:: console
   
       s3fs my-s3-bucket /s3-mount -o allow_other -o use_cache=/tmp


Token printed to shared stdout or logs
-------------------------------

The token (along with other Hydra configuration) will be printed to stdout, and so could appear in logs in an automated setup.

.. code-block:: console

    docker run --rm -it \
        -v {INPUT_DIR}:/home/input_dir:ro \
        -v {OUTPUT_DIR}:/home/shared_dir \
        --ipc="host" --network="host" \
        lightly/sampling:latest \
        token=MYAWESOMETOKEN \
        ...

This can be avoided by setting your `token` via the `LIGHTLY_TOKEN` environment variable:

.. code-block:: console

    docker run --rm -it \
        -e LIGHTLY_TOKEN=MYAWESOMETOKEN
        -v {INPUT_DIR}:/home/input_dir:ro \
        -v {OUTPUT_DIR}:/home/shared_dir \
        --ipc="host" --network="host" \
        lightly/sampling:latest \
        ...


