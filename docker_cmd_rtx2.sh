img="nvcr.io/nvidia/pytorch:25.02-py3" 
# img="padim:0.1"

docker run  --gpus all --privileged=true  --workdir /git --name "lightly"  -e DISPLAY --ipc=host -d  -p 5433:8889  \
-v /mnt/data/shezhen_code/lightly:/git/lightly/ \
-v /home/fengyun/datasets/shezhen_original_data:/git/datasets/shezhen_original_data \
$img sleep infinity

docker exec -it lightly /bin/bash

# --resume-from /git/lightly/checkpoints/simclr_imagenet1k_epoch=99-step=500400.ckpt \


