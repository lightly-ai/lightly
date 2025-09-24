img="nvcr.io/nvidia/pytorch:24.01-py3" 
# img="padim:0.1"

docker run --rm  --gpus all --privileged=true  --workdir /git --name "ligthly"  -e DISPLAY --ipc=host -d --rm  -p 5433:8889  \
-v /mnt/newdisk/she_zhen_code/lightly:/git/ligthly/ \
-v /mnt/newdisk/shezhen_original_data:/git/datasets/shezhen_original_data \
$img sleep infinity


docker exec -it lightly /bin/bash


