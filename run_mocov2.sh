#!/bin/bash

mkdir -p Logs

datatime=$(date +%Y%m%d_%H%M%S)
model=mocov2

CUDA_VISIBLE_DEVICES=1,0 NCCL_DEBUG=INFO python  ./benchmarks/imagenet/resnet50/main_mutilabel.py  \
--train-dir /git/datasets/shezhen_original_data/shezhen_unlabeled_data  \
--val-dir /git/datasets/shezhen_original_data/shezhen_label_data_2 \
--methods  $model  --batch-size-per-devic 140  --strategy  ddp  --epochs 100 --precision 32 \
--resume /git/lightly/checkpoints/mocov2_imagenet_epoch=99+12.ckpt \
| tee Logs/${model}/train_imagenet_${model}_${datatime}.log 2>&1