
![Lightly Logo](docs/logos/lightly_logo_crop.png)


![GitHub](https://img.shields.io/github/license/lightly-ai/lightly)
![Unit Tests](https://github.com/lightly-ai/lightly/workflows/Unit%20Tests/badge.svg)
![codecov](https://codecov.io/gh/lightly-ai/lightly/branch/master/graph/badge.svg?token=1NEAVROK3W)

Lightly is a computer vision framework for self-supervised learning.

> We, at [Lightly](https://www.lightly.ai), are passionate engineers who want to make deep learning more efficient. That's why - together with our community - we want to popularize the use of self-supervised methods to understand and curate raw image data. Our solution can be applied before any data annotation step and the learned representations can be used to visualize and analyze datasets. This allows to select the best core set of samples for model training through advanced filtering.

- [Homepage](https://www.lightly.ai)
- [Web-App](https://app.lightly.ai)
- [Documentation](https://docs.lightly.ai)
- [Github](https://github.com/lightly-ai/lightly)
- [Discord](https://discord.gg/xvNJW94)

### Features

Lightly offers features like

- modular framework which exposes low-level building blocks such as loss functions
- support for multi-gpu training using PyTorch Lightning
- easy to use and written in a PyTorch like style
- supports custom backbone models for self-supervised pre-training

#### Supported Models

You can [find sample code for all the supported models here.](https://docs.lightly.ai/examples/models.html)
We provide PyTorch, PyTorch Lightning and PyTorch Lightning distributed examples for each of the models
to kickstart your project.

Some of our supported models:

- [Barlow Twins, 2021](https://arxiv.org/abs/2103.03230)
- [BYOL, 2020](https://arxiv.org/abs/2006.07733)
- [DCL & DCLW, 2021](https://arxiv.org/abs/2110.06848)
- [DINO, 2021](https://arxiv.org/abs/2104.14294)
- [MAE, 2021](https://arxiv.org/abs/2111.06377)
- [MSN, 2022](https://arxiv.org/abs/2204.07141)
- [MoCo, 2019](https://arxiv.org/abs/1911.05722)
- [NNCLR, 2021](https://arxiv.org/abs/2104.14548)
- [SimCLR, 2020](https://arxiv.org/abs/2002.05709)
- [SimSiam, 2021](https://arxiv.org/abs/2011.10566)
- [SMoG, 2022](https://arxiv.org/abs/2207.06167)
- [SwaV, 2020](https://arxiv.org/abs/2006.09882)


### Tutorials

Want to jump to the tutorials and see lightly in action?

- [Train MoCo on CIFAR-10](https://docs.lightly.ai/tutorials/package/tutorial_moco_memory_bank.html)
- [Train SimCLR on clothing data](https://docs.lightly.ai/tutorials/package/tutorial_simclr_clothing.html)
- [Train SimSiam on satellite images](https://docs.lightly.ai/tutorials/package/tutorial_simsiam_esa.html)
- [Use lightly with custom augmentations](https://docs.lightly.ai/tutorials/package/tutorial_custom_augmentations.html)
- [Pre-train a Detectron2 Backbone with Lightly](https://docs.lightly.ai/tutorials/package/tutorial_pretrain_detectron2.html)

Tutorials of using the lightly packge together with the Lightly Platform:

- [Active Learning using Detectron2 on Comma10k](https://docs.lightly.ai/tutorials/platform/tutorial_active_learning_detectron2.html)
- [Active Learning with the Nvidia TLT](https://github.com/lightly-ai/NvidiaTLTActiveLearning)

Community and partner projects:

- [On-Device Deep Learning with Lightly on an ARM microcontroller](https://github.com/ARM-software/EndpointAI/tree/master/ProofOfConcepts/Vision/OpenMvMaskDefaults)

## Quick Start

Lightly requires **Python 3.6+** but we recommend using **Python 3.7+**.
We recommend installing Lightly in a **Linux** or **OSX** environment.

### Dependencies

- hydra-core>=1.0.0
- numpy>=1.18.1
- pytorch_lightning>=1.5
- requests>=2.23.0
- torchvision
- tqdm

### Installation
You can install Lightly and its dependencies from PyPI with:
```
pip3 install lightly
```

We strongly recommend that you install Lightly in a dedicated virtualenv, to avoid conflicting with your system packages.


### Lightly in Action

With lightly, you can use the latest self-supervised learning methods in a modular
way using the full power of PyTorch. Experiment with different backbones,
models, and loss functions. The framework has been designed to be easy to use
from the ground up. [Find more examples in our docs](https://docs.lightly.ai/examples/models.html).

```python
import torch
import torchvision
import lightly.models as models
import lightly.loss as loss
import lightly.data as data

# the collate function applies random transforms to the input images
collate_fn = data.ImageCollateFunction(input_size=32, cj_prob=0.5)

# create a dataset from your image folder
dataset = data.LightlyDataset(input_dir='./my/cute/cats/dataset/')

# build a PyTorch dataloader
dataloader = torch.utils.data.DataLoader(
    dataset,                # pass the dataset to the dataloader
    batch_size=128,         # a large batch size helps with the learning
    shuffle=True,           # shuffling is important!
    collate_fn=collate_fn)  # apply transformations to the input images


# create a PyTorch module for the SimCLR model
class SimCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = models.modules.SimCLRProjectionHead(
          input_dim=512, 
          hidden_dim=512,
          output_dim=128
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z

# use a resnet backbone
resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])

# build the simclr model
model = SimCLR(backbone)

# lightly exposes building blocks such as loss functions
criterion = loss.NTXentLoss(temperature=0.5)

# get a PyTorch optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=1e-0, weight_decay=1e-5)
```

You can easily use another model like SimSiam by swapping the model and the
loss function.
```python
# PyTorch module for the SimSiam model
class SimSiam(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head = SimSiamProjectionHead(512, 512, 128)
        self.prediction_head = SimSiamPredictionHead(128, 64, 128)

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p

model = SimSiam(backbone)

# use the SimSiam loss function
criterion = loss.NegativeCosineSimilarity()
```
You can [find a more complete example for SimSiam here.](https://docs.lightly.ai/examples/simsiam.html)


Use PyTorch Lightning to train the model:

```python
trainer = pl.Trainer(max_epochs=max_epochs, gpus=1)
trainer.fit(
    model,
    dataloader
)
```

Or train the model on 4 GPUs:
```python

# use distributed version of loss functions
criterion = NTXentLoss(gather_distributed=True)

trainer = pl.Trainer(
    max_epochs=max_epochs, 
    gpus=4, 
    distributed_backend='ddp'
)
trainer.fit(
    model,
    dataloader
)
```

We provide proper multi-GPU training with distributed gather and synchronized BatchNorm.

[Have a look at our docs regarding distributed training](https://docs.lightly.ai/getting_started/distributed_training.html)



### Benchmarks

Currently implemented models and their accuracy on cifar10 and imagenette. All models have been evaluated using kNN. We report the max test accuracy over the epochs as well as the maximum GPU memory consumption. All models in this benchmark use the same augmentations as well as the same ResNet-18 backbone. Training precision is set to FP32 and SGD is used as an optimizer with cosineLR.
One epoch on cifar10 takes ~35 seconds on a V100 GPU. [Learn more about the cifar10 and imagenette benchmark here](https://docs.lightly.ai/getting_started/benchmarks.html)

#### ImageNette

| Model        | Epochs | Batch Size | Test Accuracy |
|--------------|--------|------------|---------------|
| BarlowTwins  |    800 |        256 |         0.789 |
| BYOL         |    800 |        256 |         0.851 |
| DCL          |    800 |        256 |         0.816 |
| DCLW         |    800 |        256 |         0.827 |
| DINO (Res18) |    800 |        256 |         0.881 |
| Moco         |    800 |        256 |         0.832 |
| NNCLR        |    800 |        256 |         0.848 |
| SimCLR       |    800 |        256 |         0.858 |
| SimSiam      |    800 |        256 |         0.852 |
| SwaV         |    800 |        256 |         0.899 |


#### Cifar10

| Model         | Epochs | Batch Size | Test Accuracy |
|---------------|--------|------------|---------------|
| BarlowTwins   |    800 |        512 |         0.857 |
| BYOL          |    800 |        512 |         0.911 |
| DCL           |    800 |        512 |         0.873 |
| DCLW          |    800 |        512 |         0.873 | 
| DINO          |    800 |        512 |         0.884 |
| Moco          |    800 |        512 |         0.900 |
| NNCLR         |    800 |        512 |         0.896 |
| SimCLR        |    800 |        512 |         0.875 |
| SimSiam       |    800 |        512 |         0.906 |
| SwaV          |    800 |        512 |         0.881 |

## Terminology

Below you can see a schematic overview of the different concepts present in the lightly Python package. The terms in bold are explained in more detail in our [documentation](https://docs.lightly.ai).

<img src="/docs/source/getting_started/images/lightly_overview.png" alt="Overview of the lightly pip package"/></a>


### Next Steps
Head to the [documentation](https://docs.lightly.ai) and see the things you can achieve with lightly!


## Development

To install dev dependencies (for example to contribute to the framework)
you can use the following command:
```
pip3 install -e ".[dev]"
```

For more information about how to contribute have a look [here](CONTRIBUTING.md).

### Running Tests

Unit tests are within the [tests folder](tests/) and we recommend running them using 
[pytest](https://docs.pytest.org/en/stable/).
There are two test configurations available. By default, only a subset will be run.
This is faster and should take less than a minute. You can run it using
```
python -m pytest -s -v
```

To run all tests (including the slow ones) you can use the following command.
```
python -m pytest -s -v --runslow
```

### Code Linting
We provide a [Pylint](https://github.com/PyCQA/pylint) config following the
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md).

You can run the linter from your terminal either on a folder
```
pylint lightly/
```
or on a specific file
```
pylint lightly/core.py
```


## Further Reading

**Self-supervised Learning:**
- [A Simple Framework for Contrastive Learning of Visual Representations (2020)](https://arxiv.org/abs/2002.05709)
- [Momentum Contrast for Unsupervised Visual Representation Learning (2020)](https://arxiv.org/abs/1911.05722)
- [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments (2020)](https://arxiv.org/abs/2006.09882)
- [What Should Not Be Contrastive in Contrastive Learning (2020)](https://arxiv.org/abs/2008.05659)

## FAQ

- Why should I care about self-supervised learning? Aren't pre-trained models from ImageNet much better for transfer learning?
  - Self-supervised learning has become increasingly popular among scientists over the last year because the learned representations perform extraordinarily well on downstream tasks. This means that they capture the important information in an image better than other types of pre-trained models. By training a self-supervised model on *your* dataset, you can make sure that the representations have all the necessary information about your images.

- How can I contribute?
  - Create an issue if you encounter bugs or have ideas for features we should implement. You can also add your own code by forking this repository and creating a PR. More details about how to contribute with code is in our [contribution guide](CONTRIBUTING.md).

- Is this framework for free?
  - Yes, this framework completely free to use and we provide the code. We believe that
  we need to make training deep learning models more data efficient to achieve widespread adoption. One step to achieve this goal is by leveraging self-supervised learning. The company behind lightly commited to keep this framework open-source.

- If this framework is free, how is the company behind lightly making money?
  - Training self-supervised models is only one part of our solution. [The company behind lightly](https://lightly.ai/) focuses on processing and analyzing embeddings created by self-supervised models. 
  By building, what we call a self-supervised active learning loop we help companies understand and work with their data more efficiently. This framework acts as an interface
  for our platform to easily upload and download datasets, embeddings and models. Whereas 
  the platform will cost for additional features this frameworks will always remain free of charge (even for commercial use).


## Lightly in Research

- [Decoupled Contrastive Learning](https://arxiv.org/abs/2110.06848)
- [DPCL: Constrative Representation Learning with Differential Privacy](https://assets.researchsquare.com/files/rs-1516950/v1_covered.pdf?c=1654486158)
- [Self-Supervised Learning Methods for Label-Efficient Dental Caries Classification](https://www.mdpi.com/2075-4418/12/5/1237)
- [solo-learn: A Library of Self-supervised Methods for Visual Representation Learning](https://www.jmlr.org/papers/volume23/21-1155/21-1155.pdf)


## BibTeX
If you want to cite the framework feel free to use this:

```bibtex
@article{susmelj2020lightly,
  title={Lightly},
  author={Igor Susmelj, Matthias Heller, Philipp Wirth, Jeremy Prescott, Malte Ebner et al.},
  journal={GitHub. Note: https://github.com/lightly-ai/lightly},
  year={2020}
}
```
