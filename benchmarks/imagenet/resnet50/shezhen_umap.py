import simclr
import torch
import numpy as np
import os
import umap.umap_ as umap
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

from PIL import Image
from pathlib import Path
from torchvision import transforms as T
from pytorch_lightning import LightningModule, Trainer, seed_everything
from sklearn.cluster import KMeans
from lightly.transforms.utils import IMAGENET_NORMALIZE

os.environ["USE_LIBUV"] = "0"
os.environ["PL_TORCH_DISTRIBUTED_BACKEND"] = "gloo"

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

    return logits.cpu().numpy()

resume_from = "checkpoints/simclr_epoch=83+79.ckpt"
val_dir = "/git/datasets/shezhen_original_data/shezhen_unlabeled_data"
# load data path
image_extensions = ('.png', '.jpg', '.jpeg')
image_paths = [
    str(path) for path in Path(val_dir).rglob('*')
    if path.suffix.lower() in image_extensions and
    "模糊" not in path.parts and
    "模糊图片" not in path.parts
]
# load model
model = simclr.SimCLR(batch_size_per_device=1, num_classes=8)
device = torch.device('cuda')
model = model.to(device)
checkpoint = torch.load(resume_from, weights_only=False)["state_dict"]
model.load_state_dict(checkpoint)
embs=[]

for image_path in image_paths:
    result = image_inference(model, image_path, device)

    embs.append(result)

embs = np.concatenate(embs, axis=0)
reducer = umap.UMAP(n_components=3)
embs_3d = reducer.fit_transform(embs)
# k-means
num_clusters = 8  # 聚类数量
kmeans = KMeans(n_clusters=num_clusters)
labels = kmeans.fit_predict(embs_3d)
# 可视化降维结果
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制散点图
# colors = plt.cm.get_cmap('viridis', num_clusters)
# for i in range(num_clusters):
#     ax.scatter(embs_3d[labels == i, 0], embs_3d[labels == i, 1], embs_3d[labels == i, 2],
#                s=5, color=colors(i), label=f'Cluster {i}')
#
# ax.set_title('3D UMAP 降维可视化')
# ax.set_xlabel('UMAP 1')
# ax.set_ylabel('UMAP 2')
# ax.set_zlabel('UMAP 3')
# plt.show()

df = pd.DataFrame(embs_3d, columns=['UMAP 1', 'UMAP 2', 'UMAP 3'])
df['Cluster'] = labels

# 使用Plotly绘制3D散点图
fig = px.scatter_3d(df, x='UMAP 1', y='UMAP 2', z='UMAP 3', color='Cluster',
                    title='3D UMAP 降维可视化与聚类标记',
                    color_continuous_scale=px.colors.sequential.Viridis)

# 保存为HTML文件
fig.write_html('visiual/simclr_83+79e_shezhen_umap.html')

# 显示图形（可选）
fig.show()