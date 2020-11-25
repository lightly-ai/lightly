Examples on Academic Datasets
===================================


ImageNet
-----------------------------------

Let's have a look at how to run the docker container to analyze and filter the famous
ImageNet dataset. The results provided in de sample report have been obtained using the following
command.

.. code-block:: console

    docker run --gpus all --rm -it \
        -v /datasets/imagenet/train/:/home/input_dir:ro \
        -v /datasets/docker_imagenet_500k:/home/output_dir \
        --ipc="host" \
        lightly/sampling:latest \
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