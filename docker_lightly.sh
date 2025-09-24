docker run  --gpus all --privileged=true  --workdir /git --name "lightly"  -e DISPLAY --ipc=host -d  -p 5433:8889  \
-v /mnt/data/shezhen_code/lightly:/git/lightly/ \
-v /home/fengyun/datasets/shezhen_original_data:/git/datasets/shezhen_original_data \
lightly_image \
$img sleep infinity

docker exec -it lightly /bin/bash
