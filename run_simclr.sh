#!/bin/bash

# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0  # 如果不是 eth0，请改成你的网卡名

mkdir -p Logs

datatime=$(date +%Y%m%d_%H%M%S)
model=simclr

CUDA_VISIBLE_DEVICES=0,1 NCCL_DEBUG=INFO python  ./benchmarks/imagenet/resnet50/main_mutilabel.py  \
--train-dir /git/datasets/shezhen_original_data/shezhen_unlabeled_data  \
--val-dir /git/datasets/shezhen_original_data/shezhen_label_data_2 \
--methods  $model  --batch-size-per-devic 156  --strategy  ddp  --epochs 100 --precision 16 \
--resume /git/lightly//checkpoints/simclr_epoch=83-step=1621872.ckpt \
| tee Logs/train_resnet_${model}_${datatime}.log 2>&1