![Logo LightlySSL auto-apprentissage supervisé](docs/logos/lightly_SSL_logo_crop.png)

![GitHub](https://img.shields.io/github/license/lightly-ai/lightly)
![Tests Unitaires](https://github.com/lightly-ai/lightly/workflows/Unit%20Tests/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/lightly)](https://pypi.org/project/lightly/)
[![Téléchargements](https://static.pepy.tech/badge/lightly)](https://pepy.tech/project/lightly)
[![Style de code : black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Discord](https://img.shields.io/discord/752876370337726585?logo=discord&logoColor=white&label=discord&color=7289da)](https://discord.gg/xvNJW94)

**LightlySSL** est un cadre d'apprentissage auto-supervisé pour la vision par ordinateur.

- [Documentation](https://docs.lightly.ai/self-supervised-learning/)
- [Github](https://github.com/lightly-ai/lightly)
- [Discord](https://discord.gg/xvNJW94) (Nous avons des sessions hebdomadaires sur les articles !)

Pour une version commerciale avec plus de fonctionnalités, y compris le support de Docker et des modèles de pré-entraînement pour l'intégration, la classification, la détection et les tâches de segmentation avec une seule commande, veuillez contacter sales@lightly.ai.

Nous avons également construit une plateforme complète avec des fonctionnalités supplémentaires pour l'apprentissage actif et la [curation de données](https://docs.lightly.ai/docs/what-is-lightly). Si vous êtes intéressé par la solution Lightly Worker pour traiter facilement des millions d'échantillons et exécuter des [algorithmes puissants](https://docs.lightly.ai/docs/customize-a-selection) sur vos données, consultez [lightly.ai](https://www.lightly.ai). C'est gratuit pour commencer !

## Fonctionnalités

Ce cadre d'apprentissage auto-supervisé offre les fonctionnalités suivantes :

- Cadre modulaire, qui expose des blocs de construction bas-niveau tels que les fonctions de perte et les têtes de modèles.
- Facile à utiliser et écrit dans un style proche de PyTorch.
- Supporte les modèles personnalisés de backbone pour le pré-entraînement auto-supervisé.
- Prise en charge de l'entraînement distribué avec PyTorch Lightning.


### Modèles pris en charge

Vous pouvez [trouver un code d'exemple pour tous les modèles pris en charge ici.](https://docs.lightly.ai/self-supervised-learning/examples/models.html) Nous fournissons des exemples pour PyTorch, PyTorch Lightning et PyTorch Lightning distribué pour tous les modèles afin de démarrer votre projet.

**Modèles**:

| Modèle         | Année | Papier | Docs | Colab (PyTorch) | Colab (PyTorch Lightning) |
|----------------|-------|--------|------|-----------------|---------------------------|
| AIM            | 2024  | [papier](https://arxiv.org/abs/2401.08541) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/aim.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/aim.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/aim.ipynb) |
| Barlow Twins   | 2021  | [papier](https://arxiv.org/abs/2103.03230) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/barlowtwins.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/barlowtwins.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/barlowtwins.ipynb) |
| BYOL           | 2020  | [papier](https://arxiv.org/abs/2006.07733) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/byol.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/byol.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/byol.ipynb) |
| DCL & DCLW     | 2021  | [papier](https://arxiv.org/abs/2110.06848) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/dcl.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/dcl.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/dcl.ipynb) |
| DenseCL        | 2021  | [papier](https://arxiv.org/abs/2011.09157) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/densecl.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/densecl.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/densecl.ipynb) |
| DINO           | 2021  | [papier](https://arxiv.org/abs/2104.14294) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/dino.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/dino.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/dino.ipynb) |
| MAE            | 2021  | [papier](https://arxiv.org/abs/2111.06377) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/mae.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/mae.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/mae.ipynb) |
| MSN            | 2022  | [papier](https://arxiv.org/abs/2204.07141) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/msn.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/msn.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/msn.ipynb) |
| MoCo           | 2019  | [papier](https://arxiv.org/abs/1911.05722) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/moco.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/moco.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/moco.ipynb) |
| NNCLR          | 2021  | [papier](https://arxiv.org/abs/2104.14548) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/nnclr.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/nnclr.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/nnclr.ipynb) |
| PMSN           | 2022  | [papier](https://arxiv.org/abs/2210.07277) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/pmsn.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/pmsn.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/pmsn.ipynb) |
| SimCLR         | 2020  | [papier](https://arxiv.org/abs/2002.05709) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/simclr.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/simclr.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/simclr.ipynb) |
| SwAV           | 2021  | [papier](https://arxiv.org/abs/2006.09852) | [docs](https://docs.lightly.ai/self-supervised-learning/examples/swav.html) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch/swav.ipynb) | [![Open In Colab](https://img.shields.io/badge/Colab-PyTorch_Lightning-blue?logo=googlecolab)](https://colab.research.google.com/github/lightly-ai/lightly/blob/master/examples/notebooks/pytorch_lightning/swav.ipynb) |




## Tutoriels

Vous souhaitez passer aux tutoriels et voir Lightly en action ?

- [Entraîner MoCo sur CIFAR-10](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html)
- [Entraîner SimCLR sur des données de vêtements](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simclr_clothing.html)
- [Entraîner SimSiam sur des images satellites](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simsiam_esa.html)
- [Utiliser Lightly avec des augmentations personnalisées](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_custom_augmentations.html)
- [Pré-entraînement d'un backbone Detectron2 avec Lightly](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_pretrain_detectron2.html)
- [Affinage des checkpoints Lightly](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_checkpoint_finetuning.html)
- [Utiliser des modèles timm comme backbones](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_timm_backbone.html)

Projets communautaires et partenaires :

- [Apprentissage profond sur appareil avec Lightly sur un microcontrôleur ARM](https://github.com/ARM-software/EndpointAI/tree/master/ProofOfConcepts/Vision/OpenMvMaskDefaults)

## Démarrage rapide

Lightly nécessite **Python 3.7+**. Nous recommandons d'installer Lightly dans un environnement **Linux** ou **OSX**. Python 3.12 n'est pas encore pris en charge, car PyTorch lui-même n'est pas compatible avec Python 3.12.

### Dépendances

En raison de la nature modulaire du package Lightly, certains modules peuvent être utilisés avec des versions plus anciennes des dépendances. Cependant, pour utiliser toutes les fonctionnalités actuelles, Lightly nécessite les dépendances suivantes :

- [PyTorch](https://pytorch.org/)>=1.11.0
- [Torchvision](https://pytorch.org/vision/stable/index.html)>=0.12.0
- [PyTorch Lightning](https://www.pytorchlightning.ai/index.html)>=1.7.1

Lightly est compatible avec PyTorch et PyTorch Lightning v2.0+ !

### Installation

Vous pouvez installer Lightly et ses dépendances à partir de PyPI avec :

```
pip3 install lightly

```

Nous recommandons fortement d'installer Lightly dans un environnement virtuel dédié pour éviter les conflits avec vos packages système.

### Lightly en Action

Avec Lightly, vous pouvez utiliser les dernières méthodes d'apprentissage auto-supervisé de manière modulaire en utilisant toute la puissance de PyTorch. Expérimentez avec différents backbones, modèles et fonctions de perte. Le framework a été conçu pour être facile à utiliser dès le départ. [Trouvez plus d'exemples dans notre documentation](https://docs.lightly.ai/self-supervised-learning/examples/models.html).

```python
import torch
import torchvision

from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.models.modules import heads


# Créez un module PyTorch pour le modèle SimCLR.
class SimCLR(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = heads.SimCLRProjectionHead(
            input_dim=512,  # Les caractéristiques de Resnet18 ont 512 dimensions.
            hidden_dim=512,
            output_dim=128,
        )

    def forward(self, x):
        features = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(features)
        return z


# Utilisez un backbone resnet de torchvision.
backbone = torchvision.models.resnet18()
# Ignorez la tête de classification car nous voulons seulement les caractéristiques.
backbone.fc = torch.nn.Identity()

# Construisez le modèle SimCLR.
model = SimCLR(backbone)

# Préparez une transformation qui crée plusieurs vues aléatoires pour chaque image.
transform = transforms.SimCLRTransform(input_size=32, cj_prob=0.5)


# Créez un ensemble de données à partir de votre dossier d'images.
dataset = LightlyDataset(input_dir="./my/cute/cats/dataset/", transform=transform)

# Construisez un dataloader PyTorch.
dataloader = torch.utils.data.DataLoader(
    dataset,  # Passez l'ensemble de données au dataloader.
    batch_size=128,  # Une grande taille de lot aide à l'apprentissage.
    shuffle=True,  # Le mélange est important !
)

# Lightly expose des blocs de construction tels que des fonctions de perte.
criterion = loss.NTXentLoss(temperature=0.5)

# Obtenez un optimiseur PyTorch.
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=1e-6)

# Entraînez le modèle.
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
Vous pouvez facilement utiliser un autre modèle comme SimSiam en échangeant le modèle et la fonction de perte.

```python
# Module PyTorch pour le modèle SimSiam.
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

# Utilisez la fonction de perte SimSiam.
criterion = loss.NegativeCosineSimilarity()
```

# Exemple de modèle SimCLR avec PyTorch Lightning

Cet exemple montre comment entraîner un modèle SimCLR à l'aide de PyTorch Lightning.

## Code

Voici le code pour le modèle SimCLR :

```python
import torchvision
import torch
from pytorch_lightning import LightningModule, Trainer
from lightly.models import heads, loss

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

Nous fournissons des exemples d'entraînement multi-GPU avec collecte distribuée et BatchNorm synchronisé.  
[Consultez notre documentation concernant l'entraînement distribué.](https://docs.lightly.ai/self-supervised-learning/getting_started/distributed_training.html)

## Références

Modèles implémentés et leurs performances sur divers ensembles de données. Les hyperparamètres ne sont pas ajustés pour une précision maximale. Pour des résultats détaillés et plus d'informations sur les références, cliquez [ici](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html).

### ImageNet1k

[Références ImageNet1k](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html#imagenet1k)

**Remarque** : Les paramètres d'évaluation sont basés sur ces articles :

- Linéaire : [SimCLR](https://arxiv.org/abs/2002.05709)
- Affinage : [SimCLR](https://arxiv.org/abs/2002.05709)
- KNN : [InstDisc](https://arxiv.org/abs/1805.01978)

Voir les [scripts de référence](./benchmarks/imagenet/resnet50/) pour plus de détails.

| Modèle          | Backbone | Taille de lot | Époques | Linear Top1 | Finetune Top1 | kNN Top1 | Tensorboard                                                                                                                                                                    | Checkpoint                                                                                                                                                              |
| ---------------- | -------- | ------------- | ------- | ----------- | ------------- | -------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| BarlowTwins      | Res50    | 256           | 100     | 62.9        | 72.6          | 45.6     | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_barlowtwins_2023-08-18_00-11-03/pretrain/version_0/events.out.tfevents.1692310273.Machine2.569794.0) | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_barlowtwins_2023-08-18_00-11-03/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |
| BYOL             | Res50    | 256           | 100     | 62.5        | 74.5          | 46.0     | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_byol_2024-02-14_16-10-09/pretrain/version_0/events.out.tfevents.1707923418.Machine2.3205.0)          | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_byol_2024-02-14_16-10-09/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)        |
| DINO             | Res50    | 128           | 100     | 68.2        | 72.5          | 49.9     | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dino_2023-06-06_13-59-48/pretrain/version_0/events.out.tfevents.1686052799.Machine2.482599.0)        | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dino_2023-06-06_13-59-48/pretrain/version_0/checkpoints/epoch%3D99-step%3D1000900.ckpt)       |
| MAE              | ViT-B/16 | 256           | 100     | 46.0        | 81.3          | 11.2     | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_vitb16_mae_2024-02-25_19-57-30/pretrain/version_0/events.out.tfevents.1708887459.Machine2.1092409.0)          | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_vitb16_mae_2024-02-25_19-57-30/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)           |
| MoCoV2           | Res50    | 256           | 100     | 61.5        | 74.3          | 41.8     | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_mocov2_2024-02-18_10-29-14/pretrain/version_0/events.out.tfevents.1708248562.Machine2.439033.0)      | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_mocov2_2024-02-18_10-29-14/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)      |
| SimCLR\*         | Res50    | 256           | 100     | 63.2        | 73.9          | 44.8     | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_simclr_2023-06-22_09-11-13/pretrain/version_0/events.out.tfevents.1687417883.Machine2.33270.0)       | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_simclr_2023-06-22_09-11-13/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)      |
| SimCLR\* + DCL   | Res50    | 256           | 100     | 65.1        | 73.5          | 49.6     | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dcl_2023-07-04_16-51-40/pretrain/version_0/events.out.tfevents.1688482310.Machine2.247807.0)         | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dcl_2023-07-04_16-51-40/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)         |
| SimCLR\* + DCLW  | Res50    | 256           | 100     | 64.5        | 73.2          | 48.5     | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dclw_2023-07-07_14-57-13/pretrain/version_0/events.out.tfevents.1688734645.Machine2.3176.0)          | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_dclw_2023-07-07_14-57-13/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)        |
| SwAV             | Res50    | 256           | 100     | 67.2        | 75.4          | 49.5     | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_swav_2023-05-25_08-29-14/pretrain/version_0/events.out.tfevents.1684996168.Machine2.1445108.0)       | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_swav_2023-05-25_08-29-14/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)        |
| TiCo             | Res50    | 256           | 100     | 49.7        | 72.7          | 26.6     | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_tico_2024-01-07_18-40-57/pretrain/version_0/events.out.tfevents.1704649265.Machine2.1604956.0)       | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_tico_2024-01-07_18-40-57/pretrain/version_0/checkpoints/epoch%3D99-step%3D250200.ckpt)        |
| VICReg           | Res50    | 256           | 100     | 63.0        | 73.7          | 46.3     | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_vicreg_2023-09-11_10-53-08/pretrain/version_0/events.out.tfevents.1694422401.Machine2.556563.0)      | [lien](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_vicreg_2023-09-11_10-53-08/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt)      |

_\*Nous utilisons un scaling du taux d'apprentissage par racine carrée au lieu d'un scaling linéaire car cela donne de meilleurs résultats pour des tailles de lot plus petites. Voir l'Annexe B.1 dans le [document SimCLR](https://arxiv.org/abs/2002.05709)._

### ImageNet100

[Résultats détaillés des benchmarks ImageNet100](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html#imagenet100)

### Imagenette

[Résultats détaillés des benchmarks Imagenette](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html#imagenette)

### CIFAR-10

[Résultats détaillés des benchmarks CIFAR-10](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html#cifar-10)

## Terminologie

Vous pouvez voir ci-dessous un aperçu schématique des différents concepts dans le paquet. Les termes en gras sont expliqués plus en détail dans notre [documentation](https://docs.lightly.ai/self-supervised-learning/).

<img src="/docs/source/getting_started/images/lightly_overview.png" alt="Aperçu du paquet pip Lightly"/></a>

### Prochaines Étapes

Rendez-vous sur la [documentation](https://docs.lightly.ai/self-supervised-learning/) et découvrez ce que vous pouvez réaliser avec Lightly !

## Développement

Pour installer les dépendances de développement (par exemple, pour contribuer au framework), vous pouvez utiliser la commande suivante :

```
pip3 install -e ".[dev]"
```


Pour plus d'informations sur la façon de contribuer, jetez un œil [ici](CONTRIBUTING.md).

### Exécution des Tests

Les tests unitaires se trouvent dans le répertoire [tests](tests/) et nous recommandons de les exécuter en utilisant [pytest](https://docs.pytest.org/en/stable/). Il existe deux configurations de test disponibles. Par défaut, seuls un sous-ensemble sera exécuté :

```
make test-fast

```

Pour exécuter tous les tests (y compris les tests lents), vous pouvez utiliser la commande suivante :

```
make test
```
Pour tester un fichier ou un répertoire spécifique, utilisez :
```
pytest <chemin vers le fichier ou le répertoire>
```

### Formatage du Code

Pour formater le code avec [black](https://black.readthedocs.io/en/stable/) et [isort](https://docs.pytest.org), exécutez :



