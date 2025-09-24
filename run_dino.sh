#!/bin/bash

mkdir -p Logs

datatime=$(date +%Y%m%d_%H%M%S)
model=dino

CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=INFO python  ./benchmarks/imagenet/vitb16/main_mutilabel.py  \
--train-dir /git/datasets/shezhen_original_data/shezhen_unlabeled_data  \
--val-dir /git/datasets/shezhen_original_data/shezhen_label_data_2000 \
--ckpt-path /git/lightly/checkpoints/dino_shezhen-epoch=55.ckpt \
--methods  $model  --batch-size-per-devic 22  --strategy  ddp  --epochs 60 --precision 32 --devices 2 \
| tee Logs/${model}/train_imagenet_${model}_base_${datatime}.log 2>&1
