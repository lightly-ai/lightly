#!/bin/bash

mkdir -p Logs
rm Logs/*.log

datatime=$(date +%Y%m%d_%H%M%S)
model=mocov2

CUDA_VISIBLE_DEVICES=0,1 python  ./benchmarks/imagenet/resnet50/main.py  \
--train-dir /git/datasets/shezhen_original_data/shezhen_unlabeled_data  \
--val-dir /git/datasets/shezhen_original_data/shezhen_label_data \
--methods  $model  --batch-size-per-devic 32  --strategy  ddp  --epochs 10 --precision 16 \
| tee Logs/train_${model}_${datatime}.log 2>&1


