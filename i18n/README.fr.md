![Logo d'apprentissage auto-supervisé LightlySSL](docs/logos/lightly_SSL_logo_crop.png)

![GitHub](https://img.shields.io/github/license/lightly-ai/lightly)
![Tests unitaires](https://github.com/lightly-ai/lightly/workflows/Unit%20Tests/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/lightly)](https://pypi.org/project/lightly/)
[![Téléchargements](https://static.pepy.tech/badge/lightly)](https://pepy.tech/project/lightly)
[![Style de code : black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Discord](https://img.shields.io/discord/752876370337726585?logo=discord&logoColor=white&label=discord&color=7289da)](https://discord.gg/xvNJW94)
<h4 align="center">
  <p>
    <a href="https://github.com/lightly-ai/lightly/blob/master/README.md">English</a> |
    <b>Français</b>
  </p>
</h4>
Lightly**SSL** est un framework de vision par ordinateur pour l'apprentissage auto-supervisé.

- [Documentation](https://docs.lightly.ai/self-supervised-learning/)
- [Github](https://github.com/lightly-ai/lightly)
- [Discord](https://discord.gg/xvNJW94) (Nous avons des sessions papier hebdomadaires !)

Pour une version commerciale avec plus de fonctionnalités, notamment la prise en charge de Docker et la pré-formation
modèles pour les tâches d'intégration, de classification, de détection et de segmentation avec
une seule commande, veuillez contacter sales@lightly.ai.

Nous avons également construit toute une plateforme, avec des fonctionnalités supplémentaires pour l'apprentissage actif.
et [conservation des données](https://docs.lightly.ai/docs/what-is-lightly). Si vous êtes intéressé par le
Solution Lightly Worker pour traiter facilement des millions d'échantillons et exécuter des [algorithmes puissants](https://docs.lightly.ai/docs/customize-a-selection)
sur vos données, consultez [lightly.ai](https://www.lightly.ai). C'est gratuit pour commencer !

## Caractéristiques

Ce cadre d'apprentissage auto-supervisé offre les fonctionnalités suivantes :

- Cadre modulaire, qui expose les éléments de base de bas niveau tels que les fonctions de perte et
  têtes de modèles.
- Facile à utiliser et écrit dans un style de type PyTorch.
- Prend en charge les modèles de base personnalisés pour une pré-formation auto-supervisée.
- Prise en charge de la formation distribuée à l'aide de PyTorch Lightning.

### Modèles pris en charge

Vous pouvez [trouver des exemples de code pour tous les modèles pris en charge ici.](https://docs.lightly.ai/self-supervised-learning/examples/models.html) Nous fournissons PyTorch, PyTorch Lightning,
et PyTorch Lightning ont distribué des exemples pour tous les modèles afin de démarrer votre projet.

**Modèles** :

| Modèle | Année | Papier | Documents | Colab (PyTorch) | Colab (PyTorch Lightning) |
|----------------|------|-------|------|-------------- -------|----------------------------|
| OBJECTIF | 2024 | [papier](https://arxiv.org/abs/2401.08541) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/aim.html) | [![Ouvrir dans Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ légèrement/blob/master/examples/notebooks/pytorch/aim.ipynb) | [![Ouvrir dans Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ légèrement/blob/master/examples/notebooks/pytorch_lightning/aim.ipynb) |
| Jumeaux Barlow | 2021 | [papier](https://arxiv.org/abs/2103.03230) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/barlowtwins.html) | [![Ouvrir dans Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ légèrement/blob/master/examples/notebooks/pytorch/barlowtwins.ipynb) | [![Ouvrir dans Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ légèrement/blob/master/examples/notebooks/pytorch_lightning/barlowtwins.ipynb) |
| BYOL | 2020 | [article](https://arxiv.org/abs/2006.07733) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/byol.html) | [![Ouvrir dans Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ légèrement/blob/master/examples/notebooks/pytorch/byol.ipynb) | [![Ouvrir dans Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ légèrement/blob/master/examples/notebooks/pytorch_lightning/byol.ipynb) |
| DCL & DCLW | 2021 | [paper](https://arxiv.org/abs/2110.06848) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/dcl.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch/dcl.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch_lightning/dcl.ipynb) |
| DenseCL | 2021 | [paper](https://arxiv.org/abs/2011.09157) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/densecl.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch/densecl.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch_lightning/densecl.ipynb) |
| DINO | 2021 | [paper](https://arxiv.org/abs/2104.14294) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/dino.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch/dino.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch_lightning/dino.ipynb) |
| MAE | 2021 | [paper](https://arxiv.org/abs/2111.06377) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/mae.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch/mae.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch_lightning/mae.ipynb) |
| MSN | 2022 | [paper](https://arxiv.org/abs/2204.07141) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/msn.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch/msn.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch_lightning/msn.ipynb) |
| MoCo | 2019 | [paper](https://arxiv.org/abs/1911.05722) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/moco.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch/moco.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch_lightning/moco.ipynb) |
| NNCLR | 2021 | [paper](https://arxiv.org/abs/2104.14548) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/nnclr.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch/nnclr.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch_lightning/nnclr.ipynb) |
| PMSN | 2022 | [paper](https://arxiv.org/abs/2210.07277) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/pmsn.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch/pmsn.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch_lightning/pmsn.ipynb) |
| SimCLR | 2020 | [paper](https://arxiv.org/abs/2002.05709) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/simclr.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch/simclr.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch_lightning/simclr.ipynb) |
| SimMIM | 2022 | [paper](https://arxiv.org/abs/2111.09886) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/simmim.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch/simmim.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch_lightning/simmim.ipynb) |
| SimSiam | 2021 | [paper](https://arxiv.org/abs/2011.10566) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/simsiam.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch/simsiam.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch_lightning/simsiam.ipynb) |
| SwaV | 2020 | [paper](https://arxiv.org/abs/2006.09882) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/swav.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch/swav.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ lightly/blob/master/examples/notebooks/pytorch_lightning/swav.ipynb) |
| VICReg | 2021 | [article](https://arxiv.org/abs/2105.04906) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/vicreg.html) | [![Ouvrir dans Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ légèrement/blob/master/examples/notebooks/pytorch/vicreg.ipynb) | [![Ouvrir dans Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/ légèrement/blob/master/examples/notebooks/pytorch_lightning/vicreg.ipynb) |

## Tutoriels

Vous voulez passer aux didacticiels et voir Lightly en action ?

- [Former MoCo sur CIFAR-10](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html)
- [Former SimCLR sur les données vestimentaires](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simclr_clothing.html)
- [Former SimSiam sur les images satellite](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simsiam_esa.html)
- [Utilisez Lightly avec des augmentations personnalisées](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_custom_augmentations.html)
-[Pré-entraîner un backbone Detectron2 avec Lightly](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_pretrain_detectron2.html)
- [Finetuning Lightly Checkpoints](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_checkpoint_finetuning.html)
- [Utilisation des modèles Timm comme backbones](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_timm_backbone.html)

Projets communautaires et partenaires :

- [Apprentissage profond sur appareil avec Lightly sur un microcontrôleur ARM](https://github.com/ARM-software/EndpointAI/tree/master/ProofOfConcepts/Vision/OpenMvMaskDefaults)

## Démarrage rapide

Nécessite légèrement **Python 3.7+**. Nous vous recommandons d'installer Lightly dans un environnement **Linux** ou **OSX**. Python 3.12 n'est pas encore pris en charge, car PyTorch lui-même n'est pas compatible avec Python 3.12.

### Dépendances

En raison de la nature modulaire du package Lightly, certains modules peuvent être utilisés avec des versions plus anciennes de dépendances. Cependant, pour utiliser toutes les fonctionnalités d'aujourd'hui avec légèreté, il faut les dépendances suivantes :

- [PyTorch](https://pytorch.org/)>=1.11.0
- [Torchvision](https://pytorch.org/vision/stable/index.html)>=0.12.0
- [PyTorch Lightning](https://www.pytorchlightning.ai/index.html)>=1.7.1

Lightly est compatible avec PyTorch et PyTorch Lightning v2.0+ !

###Installation

Vous pouvez installer Lightly et ses dépendances depuis PyPI avec :

```
pip3 installe légèrement
```

Nous vous recommandons fortement d'installer Lightly dans un environnement virtuel dédié pour éviter les conflits avec vos packages système.

### Légèrement en action

Avec Lightly, vous pouvez utiliser les dernières méthodes d'apprentissage auto-supervisé de manière modulaire.
manière en utilisant toute la puissance de PyTorch. Expérimentez avec différents backbones,
modèles et fonctions de perte. Le framework a été conçu pour être facile à utiliser
de bas en haut. [Trouvez plus d'exemples dans nos documents](https://docs.lightly.ai/self-supervised-learning/examples/models.html).

```python
import torch
import torchvision

from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads


# Create a PyTorch module for the SimCLR model.
class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=512,  # Resnet18 features have 512 dimensions.
            hidden_dim=512,
            output_dim=128,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z


# Use a resnet backbone from torchvision.
backbone = torchvision.models.resnet18()
# Ignore the classification head as we only want the features.
backbone.fc = torch.nn.Identity()

# Build the SimCLR model.
model = SimCLR(backbone)

# Prepare transform that creates multiple random views for every image.
transform = transforms.SimCLRTransform(input_size=32, cj_prob=0.5)


# Create a dataset from your image folder.
dataset = LightlyDataset(input_dir="./my/cute/cats/dataset/", transform=transform)

# Build a PyTorch dataloader.
dataloader = torch.utils.data.DataLoader(
    dataset,  # Pass the dataset to the dataloader.
    batch_size=128,  # A large batch size helps with the learning.
    shuffle=True,  # Shuffling is important!
)

# Lightly exposes building blocks such as loss functions.
criterion = loss.NTXentLoss(temperature=0.5)

# Get a PyTorch optimizer.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-6)

# Train the model.
for epoch in range(10):
    for (view0, view1), targets, filenames in dataloader:
        z0 = model(view0)
        z1 = model(view1)
        loss = criterion(z0, z1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"loss: {loss.item():.5f}")
```

Vous pouvez facilement utiliser un autre modèle comme SimSiam en échangeant le modèle et le
fonction de perte.

```python
# PyTorch module for the SimSiam model.
class SimSiam(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimSiamProjectionHead(512, 512, 128)
        self.prediction_head = heads.SimSiamPredictionHead(128, 64, 128)

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p


model = SimSiam(backbone)

# Use the SimSiam loss function.
criterion = loss.NegativeCosineSimilarity()
```

Vous pouvez [trouver un exemple plus complet pour SimSiam ici.](https://docs.lightly.ai/self-supervised-learning/examples/simsiam.html)

Utilisez PyTorch Lightning pour entraîner le modèle :

```python
from pytorch_lightning import LightningModule, Trainer

class SimCLR(LightningModule):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18()
        resnet.fc = torch.nn.Identity()
        self.backbone = resnet
        self.projection_head = heads.SimCLRProjectionHead(512, 512, 128)
        self.criterion = loss.NTXentLoss()

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z

    def training_step(self, batch, batch_index):
        (view0, view1), _, _ = batch
        z0 = self.forward(view0)
        z1 = self.forward(view1)
        loss = self.criterion(z0, z1)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim


model = SimCLR()
trainer = Trainer(max_epochs=10, devices=1, accelerator="gpu")
trainer.fit(model, dataloader)
```

Consultez [nos documents pour un exemple complet de PyTorch Lightning.](https://docs.lightly.ai/self-supervised-learning/examples/simclr.html)

Ou entraînez le modèle sur 4 GPU :

```python

# Use distributed version of loss functions.
criterion = loss.NTXentLoss(gather_distributed=True)

trainer = Trainer(
    max_epochs=10,
    devices=4,
    accelerator="gpu",
    strategy="ddp",
    sync_batchnorm=True,
    use_distributed_sampler=True,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
)
trainer.fit(model, dataloader)
```

Nous fournissons des exemples de formation multi-GPU avec collecte distribuée et BatchNorm synchronisé.
[Jetez un œil à nos documents concernant la formation distribuée.](https://docs.lightly.ai/self-supervised-learning/getting_started/distributed_training.html)

## Repères

Modèles implémentés et leurs performances sur divers ensembles de données. Les hyperparamètres ne sont pas
réglé pour une précision maximale. Pour des résultats détaillés et plus d’informations sur les benchmarks, cliquez sur
[ici](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html).

### ImageNet1k

[Benchmarks ImageNet1k](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html#imagenet1k)

**Remarque** : Les paramètres d'évaluation sont basés sur ces articles :

- Linéaire : [SimCLR](https://arxiv.org/abs/2002.05709)
- Réglage fin : [SimCLR](https://arxiv.org/abs/2002.05709)
-KNN : [InstDisc](https://arxiv.org/abs/1805.01978)

Voir les [scripts d'analyse comparative](./benchmarks/imagenet/resnet50/) pour plus de détails.

| Modèle | Colonne vertébrale | Taille du lot | Époques | Linéaire Top1 | Affiner Top1 | kNN Top1 | Tableau tensoriel | Point de contrôle |
| --------------- | -------- | ---------- | ------ | ----------- | ------------- | -------- | -------------------------------------------------- -------------------------------------------------- -------------------------------------------------- -------------------- | -------------------------------------------------- -------------------------------------------------- -------------------------------------------------- ----------------- |
| BarlowTwins | Rés50 | 256 | 100 | 62,9 | 72,6 | 45.6 | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_barlowtwins_2023-08-18_00-11-03/pretrain/version_0/events.out.tfevents.1692310273.Machine2.569794.0) | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_barlowtwins_2023-08-18_00-11-03/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| BYOL | Rés50 | 256 | 100 | 62,5 | 74,5 | 46,0 | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_byol_2024-02-14_16-10-09/pretrain/version_0/events.out.tfevents.1707923418.Machine2.3205.0) | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_byol_2024-02-14_16-10-09/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| DINOSAURE | Rés50 | 128 | 100 | 68.2 | 72,5 | 49,9 | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dino_2023-06-06_13-59-48/pretrain/version_0/events.out.tfevents.1686052799.Machine2.482599.0) | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dino_2023-06-06_13-59-48/pretrain/version_0/checkpoints/epoch%3D99-step%3D1000900.ckpt) |
| MAE | ViT-B/16 | 256 | 100 | 46,0 | 81.3 | 11.2 | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_vitb16_mae_2024-02-25_19-57-30/pretrain/version_0/events.out.tfevents.1708887459.Machine2.1092409.0) | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_vitb16_mae_2024-02-25_19-57-30/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| MoCoV2 | Rés50 | 256 | 100 | 61,5 | 74.3 | 41,8 | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_mocov2_2024-02-18_10-29-14/pretrain/version_0/events.out.tfevents.1708248562.Machine2.439033.0) | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_mocov2_2024-02-18_10-29-14/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| SimCLR\* | Rés50 | 256 | 100 | 63.2 | 73,9 | 44,8 | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_simclr_2023-06-22_09-11-13/pretrain/version_0/events.out.tfevents.1687417883.Machine2.33270.0) | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_simclr_2023-06-22_09-11-13/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| SimCLR\* + DCL | Rés50 | 256 | 100 | 65.1 | 73,5 | 49.6 | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dcl_2023-07-04_16-51-40/pretrain/version_0/events.out.tfevents.1688482310.Machine2.247807.0) | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dcl_2023-07-04_16-51-40/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| SimCLR\* + DCLW | Rés50 | 256 | 100 | 64,5 | 73.2 | 48,5 | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dclw_2023-07-07_14-57-13/pretrain/version_0/events.out.tfevents.1688734645.Machine2.3176.0) | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dclw_2023-07-07_14-57-13/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| SWAV | Rés50 | 256 | 100 | 67.2 | 75.4 | 49,5 | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_swav_2023-05-25_08-29-14/pretrain/version_0/events.out.tfevents.1684996168.Machine2.1445108.0) | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_swav_2023-05-25_08-29-14/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| TiCo | Rés50 | 256 | 100 | 49,7 | 72,7 | 26.6 | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_tico_2024-01-07_18-40-57/pretrain/version_0/events.out.tfevents.1704649265.Machine2.1604956.0) | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_tico_2024-01-07_18-40-57/pretrain/version_0/checkpoints/epoch%3D99-step%3D250200.ckpt) |
| VICReg | Rés50 | 256 | 100 | 63,0 | 73,7 | 46.3 | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_vicreg_2023-09-11_10-53-08/pretrain/version_0/events.out.tfevents.1694422401.Machine2.556563.0) | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_vicreg_2023-09-11_10-53-08/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |

_\*Nous utilisons la mise à l'échelle du taux d'apprentissage de la racine carrée au lieu de la mise à l'échelle linéaire, car elle donne
de meilleurs résultats pour des lots plus petits. Voir l'annexe B.1 dans le [article SimCLR](https://arxiv.org/abs/2002.05709)._

### ImageNet100

[Résultats détaillés des tests ImageNet100](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html#imagenet100)

### Imagenette

[Résultats détaillés des tests d'Imagenette](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html#imagenette)

### CIFAR-10

[Résultats détaillés des tests de référence CIFAR-10](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html#cifar-10)

## Terminologie
Ci-dessous, vous pouvez voir un aperçu schématique des différents concepts du package.

Les termes en gras sont expliqués plus en détail dans notre [documentation](https://docs.lightly.ai/self-supervised-learning/).

<img src="/docs/source/getting_started/images/lightly_overview.png" alt="Présentation du package Lightly pip"/></a>

### Prochaines étapes

Rendez-vous sur la [documentation](https://docs.lightly.ai/self-supervised-learning/) et voyez les choses que vous pouvez réaliser avec Lightly !

## Développement

Pour installer des dépendances de développement (par exemple pour contribuer au framework) vous pouvez utiliser la commande suivante :

```
pip3 install -e ".[dev]"
```

Pour plus d'informations sur la façon de contribuer, jetez un œil [ici] (CONTRIBUTING.md).

### Exécution de tests

Les tests unitaires se trouvent dans le [répertoire tests](tests/) et nous vous recommandons de les exécuter en utilisant
[pytest](https://docs.pytest.org/en/stable/). Il existe deux configurations de test
disponible. Par défaut, seul un sous-ensemble sera exécuté :

```
faire des tests rapides
```

Pour exécuter tous les tests (y compris les plus lents), vous pouvez utiliser la commande suivante :

```
faire un test
```

Pour tester un fichier ou un répertoire spécifique, utilisez :

```
pytest <chemin d'accès au fichier ou au répertoire>
```

### Formatage du code

Pour formater le code avec [black](https://black.readthedocs.io/en/stable/) et [isort](https://docs.pytest.org), exécutez :

```
créer un format
```

## Lectures complémentaires

**Apprentissage auto-supervisé** :

- Jetez un œil à notre [chaîne #papers sur Discord](https://discord.com/channels/752876370337726585/815153188487299083)
  pour les derniers documents d'apprentissage auto-supervisés.
- [Un livre de recettes d'apprentissage auto-supervisé, 2023](https://arxiv.org/abs/2304.12210)
- [Les auto-encodeurs masqués sont des apprenants en vision évolutive, 2021](https://arxiv.org/abs/2111.06377)
- [Propriétés émergentes des transformateurs de vision auto-supervisés, 2021](https://arxiv.org/abs/2104.14294)
- [Apprentissage non supervisé des fonctionnalités visuelles par affectations de clusters contrastées, 2021](https://arxiv.org/abs/2006.09882)
- [Ce qui ne devrait pas être contrastif dans l'apprentissage contrastif, 2020](https://arxiv.org/abs/2008.05659)
- [Un cadre simple pour l'apprentissage contrastif des représentations visuelles, 2020](https://arxiv.org/abs/2002.05709)
- [Momentum Contrast pour l'apprentissage de la représentation visuelle non supervisé, 2020](https://arxiv.org/abs/1911.05722)

##FAQ

- Pourquoi devrais-je me soucier de l'apprentissage auto-supervisé ? Les modèles pré-entraînés d'ImageNet ne sont-ils pas bien meilleurs pour l'apprentissage par transfert ?

- L'apprentissage auto-supervisé est devenu de plus en plus populaire parmi les scientifiques ces dernières années car les représentations apprises fonctionnent extraordinairement bien sur les tâches en aval. Cela signifie qu'ils capturent mieux les informations importantes dans une image que les autres types de modèles pré-entraînés. En entraînant un modèle auto-supervisé sur _votre_ ensemble de données, vous pouvez vous assurer que les représentations contiennent toutes les informations nécessaires sur vos images.

- Comment puis-je contribuer ?

  - Créez un problème si vous rencontrez des bugs ou si vous avez des idées de fonctionnalités que nous devrions implémenter. Vous pouvez également ajouter votre propre code en créant ce référentiel et en créant un PR. Plus de détails sur la façon de contribuer avec du code se trouvent dans notre [guide de contribution] (CONTRIBUTING.md).

- Ce framework est-il gratuit ?

  - Oui, ce framework est totalement gratuit et nous fournissons le code source. Nous pensons que nous devons rendre la formation des modèles d’apprentissage profond plus efficace en matière de données pour parvenir à une adoption généralisée. Une étape pour atteindre cet objectif consiste à tirer parti de l’apprentissage auto-supervisé. La société derrière Lightly s'engage à garder ce framework open source.

- Si ce framework est gratuit, comment l'entreprise derrière Lightly gagne-t-elle de l'argent ?
  - La formation de modèles auto-supervisés n'est qu'une partie de notre solution.
    [La société derrière Lightly](https://lightly.ai/) se concentre sur le traitement et l'analyse des intégrations créées par des modèles auto-supervisés.
    En créant ce que nous appelons une boucle d'apprentissage actif auto-supervisé, nous aidons les entreprises à comprendre et à utiliser leurs données plus efficacement.
    La [Solution Lightly](https://docs.lightly.ai) étant un produit freemium, vous pouvez l'essayer gratuitement. Cependant, nous facturerons certaines fonctionnalités.
  - Dans tous les cas, ce framework sera toujours libre d'utilisation, même à des fins commerciales.

## Légèrement dans la recherche

- [Apprentissage auto-supervisé par ingénierie inverse, 2023](https://arxiv.org/abs/2305.15614)
- [Apprentissage des représentations visuelles via un échantillonnage guidé par le langage, 2023](https://arxiv.org/pdf/2302.12248.pdf)
- [Méthodes d'apprentissage auto-supervisées pour la classification des caries dentaires efficace sur le label, 2022](https://www.mdpi.com/2075-4418/12/5/1237)
- [DPCL : Apprentissage de représentation contrastive avec confidentialité différentielle, 2022](https://assets.researchsquare.com/files/rs-1516950/v1_covered.pdf?c=1654486158)
- [Apprentissage contrastif découplé, 2021](https://arxiv.org/abs/2110.06848)
- [apprentissage en solo : une bibliothèque de méthodes auto-supervisées pour l'apprentissage de la représentation visuelle, 2021](https://www.jmlr.org/papers/volume23/21-1155/21-1155.pdf)

## Entreprise derrière ce framework Open Source

[Lightly](https://www.lightly.ai) est une spin-off de l'ETH Zurich qui aide les entreprises
créer des pipelines d'apprentissage actif efficaces pour sélectionner les données les plus pertinentes pour leurs modèles.

Vous pouvez en savoir plus sur l'entreprise et ses services en suivant les liens ci-dessous :

- [Page d'accueil](https://www.lightly.ai)
- [Application Web](https://app.lightly.ai)
- [Documentation de la solution Lightly (Lightly Worker et API)](https://docs.lightly.ai/)
