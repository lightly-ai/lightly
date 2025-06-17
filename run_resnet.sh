#!/bin/bash

mkdir -p Logs

datatime=$(date +%Y%m%d_%H%M%S)

CUDA_VISIBLE_DEVICES=1 python  ./benchmarks/imagenet/resnet50/main.py  \
--train-dir /git/datasets/shezhen_original_data/shezhen_unlabeled_data  \
--val-dir /git/datasets/shezhen_original_data/shezhen_label_data \
--methods  simclr  --batch-size-per-devic 32  --strategy  ddp_find_unused_parameters_true  --epochs 10 --precision 16 \
| tee Logs/train_${datatime}.log 2>&1

