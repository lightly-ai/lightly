#!/bin/bash

mkdir -p Logs

rm Logs/*.log

datatime=$(date +%Y%m%d_%H%M%S)
model= dino

CUDA_VISIBLE_DEVICES=1 python  ./benchmarks/imagenet/vitb16/main.py  \
--train-dir /git/datasets/shezhen_original_data/shezhen_unlabeled_data  \
--val-dir /git/datasets/shezhen_original_data/shezhen_label_data \
--methods $model   --batch-size-per-devic 32  --strategy  ddp   --precision 16 \
| tee Logs/train_${model}_${datatime}.log 2>&1


# mae 不适合做图像分类任务
# rino 算法 不要batch 比较大
