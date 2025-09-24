#!/bin/bash

mkdir -p Logs

datatime=$(date +%Y%m%d_%H%M%S)
model=dino

CUDA_VISIBLE_DEVICES=1 python  ./benchmarks/imagenet/vitb16/main.py  \
--train-dir /git/datasets/shezhen_original_data/shezhen_unlabeled_data  \
--val-dir /git/datasets/shezhen_original_data/shezhen_label_data \
--methods $model   --batch-size-per-devic 32  --strategy  ddp   --precision 32 \
--ckpt-path ./benchmark_logs/dino/2025-06-13_14-30-30/pretrain/version_0/checkpoints/epoch=21-step=177430.ckpt \
| tee Logs/train_vit_${model}_${datatime}.log 2>&1


# mae 不适合做图像分类任务
# rino 需要batch size 比较大
