from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, Sequence, Union, Optional
import barlowtwins
import byol
import dcl
import dclw
import dino
import finetune_eval
import knn_eval
import linear_eval
import mocov2
import simclr_muti
import simclr
import swav
import tico
import torch
import vicreg

import numpy as np
import os
import traceback
from tqdm import tqdm

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    DeviceStatsMonitor,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.serialization import add_safe_globals
from torch.utils.data import DataLoader
from torchvision import transforms as T

from lightly.data import LightlyDataset, MultiLabelDataset
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking import MetricCallback
from lightly.utils.dist import print_rank_zero

def image_inference(
    model: LightningModule,
    image_path: str,
    device: torch.device,
) -> list:
    """对单张图像进行推理并返回分类结果"""
    # 图像预处理（使用验证集转换）
    val_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_NORMALIZE["mean"], std=IMAGENET_NORMALIZE["std"]),
    ])
    
    # 加载并预处理图像
    image = Image.open(image_path).convert('RGB')
    image_tensor = val_transform(image).unsqueeze(0).to(device)
    
    # 推理
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        print(logits)
        print(logits.shape)
        # probs = torch.nn.functional.softmax(logits, dim=1)
        # top_probs, top_indices = probs.topk(top_k, dim=1)
    
    # 整理结果
    results = []
    class_names = ['气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '平和质']  # 可替换为实际类别名
    for i in range(top_k):
        results.append({
            "class_index": top_indices[0, i].item(),
            "class_name": class_names[top_indices[0, i].item()],
            "confidence": top_probs[0, i].item()
        })
    return results

model = METHODS[method]["model"](
        batch_size_per_device=batch_size_per_device, num_classes=num_classes)

device = torch.device(accelerator if accelerator != "cpu" else "cpu")
model = model.to(device)

all_classes = ['气虚质', '阳虚质', '阴虚质', '痰湿质', '湿热质', '血瘀质', '气郁质', '平和质']
results = image_inference(
    model=model,
    image_path=infer_image,
    device=device,
    )
print_rank_zero(f"Image inference results for {infer_image}:")
