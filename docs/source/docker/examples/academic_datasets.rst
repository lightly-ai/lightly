ImageNet
===================================

Let's have a look at how to run the docker container to analyze and filter the famous
ImageNet dataset. You can reproduce the sample report using the following
command.

.. code-block:: console

    docker run --gpus all --rm -it \
        -v /datasets/imagenet/train/:/home/input_dir:ro \
        -v /datasets/docker_imagenet_500k:/home/output_dir \
        --ipc="host" \
        lightly/worker:latest \
        token=MYAWESOMETOKEN \
        lightly.collate.input_size=64 \
        lightly.loader.batch_size=256 \
        lightly.loader.num_workers=8 \
        lightly.trainer.max_epochs=0 \
        stopping_condition.n_samples=500000 \
        remove_exact_duplicates=True \
        enable_corruptness_check=False

The complete **processing time** was **04h 37m 02s**. The machine used for this experiment is a cloud instance with
8 cores, 30GB of RAM, and a V100 GPU. The dataset was stored on an SSD drive.

You can also use the direct link for the
`ImageNet <https://uploads-ssl.webflow.com/5f7ac1d59a6fc13a7ce87963/5facf14359b56365e817a773_report_imagenet_500k.pdf>`_ report.





Combining Cityscapes with Kitti
================================

Using Lightly Docker and the datapool feature we can combine two datasets and 
ensure that we only keep the unique samples.

.. code-block:: console

    docker run --shm-size="512m" --gpus all --rm -it \
        -v /datasets/cityscapes/leftImg8bit/train/:/home/input_dir:ro \
        -v /datasets/docker_out_cityscapes:/home/output_dir \
        -v /datasets/docker_out_cityscapes:/home/shared_dir \
        -e --ipc="host" --network="host" lightly/worker:latest \
        token=MYAWESOMETOKEN lightly.loader.num_workers=8 \
        stopping_condition.min_distance=0.2 remove_exact_duplicates=True \
        enable_corruptness_check=False enable_training=True \
        lightly.trainer.max_epochs=20 lightly.optimizer.lr=1.0 \
        lightly.trainer.precision=32 lightly.loader.batch_size=256 \
        lightly.collate.input_size=64 datapool.name=autonomous_driving

The report for running the command can be found here:
:download:`Cityscapes.pdf <../resources/datapool_example_cityscapes.pdf>` 

Since the Cityscapes dataset has subfolders for the different cities Lightly
Docker uses them as weak labels for the embedding plot as shown below.

.. figure:: ../resources/cityscapes_scatter_umap_k_15_no_overlay.png
    :align: center
    :alt: some alt text

    Scatterplot of Cityscapes. Each color represents one of the 18 
    subfolders (cities) of the Cityscapes dataset.


Now we can use the datapool and pre-trained model to select the interesting
frames from Kitti and add them to Cityscapes:

.. code-block:: console

    docker run --shm-size="512m" --gpus all --rm -it \
        -v /datasets/kitti/training/image_2/:/home/input_dir:ro \
        -v /datasets/docker_out_cityscapes:/home/output_dir \
        -v /datasets/docker_out_cityscapes:/home/shared_dir \
        -e --ipc="host" --network="host" lightly/worker:latest \
        token=MYAWESOMETOKEN lightly.loader.num_workers=8 \
        stopping_condition.min_distance=0.2 remove_exact_duplicates=True \
        enable_corruptness_check=False enable_training=False \
        lightly.trainer.max_epochs=20 lightly.optimizer.lr=1.0 \
        lightly.trainer.precision=32 lightly.loader.batch_size=256 \
        lightly.collate.input_size=64 datapool.name=autonomous_driving


We will end up with new plots in the report due to the datapool. The plots show
the embeddings and highlight with blue color the samples which have been added
from the new dataset. In our experiment, we see that Lighlty Docker added several 
new samples outside of the previous embedding distribution. This is great, since it
shows that Cityscapes and Kitti have different data and we can combine the two datasets.

.. figure:: ../resources/datapool_umap_scatter_before_threshold_0.2.png
    :align: center
    :alt: An example of the newly selected examples when we use 
          stopping_condition.min_distance=0.2

    An example of the newly selected examples when we use 
    stopping_condition.min_distance=0.2. 7089 samples from Kitti have been added
    to our existing datapool.

.. figure:: ../resources/datapool_umap_scatter_before_threshold_0.05.png
    :align: center
    :alt: An example of the newly selected examples when we use 
          stopping_condition.min_distance=0.05

    An example of the newly selected examples when we use 
    stopping_condition.min_distance=0.05. 3598 samples from Kitti have been added
    to our existing datapool.


The report for running the command can be found here:
:download:`kitti_with_min_distance=0.2.pdf <../resources/datapool_example_kitti_threshold_0.2.pdf>` 

And the report for stopping condition mininum distance of 0.05:
:download:`kitti_with_min_distance=0.05.pdf <../resources/datapool_example_kitti_threshold_0.05.pdf>` 