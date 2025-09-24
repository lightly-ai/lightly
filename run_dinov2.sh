#!/bin/bash

mkdir -p Logs

datatime=$(date +%Y%m%d_%H%M%S)
model=dinov2

CUDA_VISIBLE_DEVICES=1 NCCL_DEBUG=INFO python  ./benchmarks/imagenet/vitb16/main_mutilabel.py  \
--train-dir /git/datasets/shezhen_original_data/shezhen_unlabeled_data  \
--val-dir /git/datasets/shezhen_original_data/shezhen_label_data_2000 \
--ckpt-path /git/lightly/checkpoints/dinov2_shezhen-epoch=49+45.ckpt \
--methods  $model  --batch-size-per-devic 38  --strategy  ddp  --epochs 50 --precision 32 --devices 1 \
| tee Logs/${model}/train_shezhen_${model}_${datatime}.log 2>&1