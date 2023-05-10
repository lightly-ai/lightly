
![Lightly Logo](docs/logos/lightly_logo_crop.png)

![GitHub](https://img.shields.io/github/license/lightly-ai/lightly)
![Unit Tests](https://github.com/lightly-ai/lightly/workflows/Unit%20Tests/badge.svg)
[![PyPI](https://img.shields.io/pypi/v/lightly)](https://pypi.org/project/lightly/)
[![Downloads](https://pepy.tech/badge/lightly)](https://pepy.tech/project/lightly)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Lightly is a computer vision framework for self-supervised learning.

> We, at [Lightly](https://www.lightly.ai), are passionate engineers who want to make deep learning more efficient. That's why - together with our community - we want to popularize the use of self-supervised methods to understand and curate raw image data. Our solution can be applied before any data annotation step and the learned representations can be used to visualize and analyze datasets. This allows to select the best set of samples for model training through advanced filtering.

- [Homepage](https://www.lightly.ai)
- [Web-App](https://app.lightly.ai)
- [Documentation](https://docs.lightly.ai/self-supervised-learning/)
- [Lightly Solution Documentation (Lightly Worker & API)](https://docs.lightly.ai/)
- [Github](https://github.com/lightly-ai/lightly)
- [Discord](https://discord.gg/xvNJW94) (We have weekly paper sessions!)


## Features

Lightly offers features like

- Modular framework which exposes low-level building blocks such as loss functions and
  model heads.
- Easy to use and written in a PyTorch like style.
- Supports custom backbone models for self-supervised pre-training.
- Support for distributed training using PyTorch Lightning.


### Supported Models

You can [find sample code for all the supported models here.](https://docs.lightly.ai/self-supervised-learning/examples/models.html) We provide PyTorch, PyTorch Lightning,
and PyTorch Lightning distributed examples for all models to kickstart your project.

**Models**:

- Barlow Twins, 2021 [paper](https://arxiv.org/abs/2103.03230) [docs](https://docs.lightly.ai/self-supervised-learning/examples/barlowtwins.html)
- BYOL, 2020 [paper](https://arxiv.org/abs/2006.07733) [docs](https://docs.lightly.ai/self-supervised-learning/examples/byol.html)
- DCL & DCLW, 2021 [paper](https://arxiv.org/abs/2110.06848) [docs](https://docs.lightly.ai/self-supervised-learning/examples/dcl.html)
- DINO, 2021 [paper](https://arxiv.org/abs/2104.14294) [docs](https://docs.lightly.ai/self-supervised-learning/examples/dino.html)
- MAE, 2021 [paper](https://arxiv.org/abs/2111.06377) [docs](https://docs.lightly.ai/self-supervised-learning/examples/mae.html)
- MSN, 2022 [paper](https://arxiv.org/abs/2204.07141) [docs](https://docs.lightly.ai/self-supervised-learning/examples/msn.html)
- MoCo, 2019 [paper](https://arxiv.org/abs/1911.05722) [docs](https://docs.lightly.ai/self-supervised-learning/examples/moco.html)
- NNCLR, 2021 [paper](https://arxiv.org/abs/2104.14548) [docs](https://docs.lightly.ai/self-supervised-learning/examples/nnclr.html)
- PMSN, 2022 [paper](https://arxiv.org/abs/2210.07277) [docs](https://docs.lightly.ai/self-supervised-learning/examples/pmsn.html)
- SimCLR, 2020 [paper](https://arxiv.org/abs/2002.05709) [docs](https://docs.lightly.ai/self-supervised-learning/examples/simclr.html)
- SimMIM, 2021 [paper](https://arxiv.org/abs/2111.09886) [docs](https://docs.lightly.ai/self-supervised-learning/examples/simmim.html)
- SimSiam, 2021 [paper](https://arxiv.org/abs/2011.10566) [docs](https://docs.lightly.ai/self-supervised-learning/examples/simsiam.html)
- SMoG, 2022 [paper](https://arxiv.org/abs/2207.06167) [docs](https://docs.lightly.ai/self-supervised-learning/examples/smog.html)
- SwaV, 2020 [paper](https://arxiv.org/abs/2006.09882) [docs](https://docs.lightly.ai/self-supervised-learning/examples/swav.html)
- TiCo, 2022 [paper](https://arxiv.org/abs/2206.10698) [docs](https://docs.lightly.ai/self-supervised-learning/examples/tico.html)
- VICReg, 2022 [paper](https://arxiv.org/abs/2105.04906) [docs](https://docs.lightly.ai/self-supervised-learning/examples/vicreg.html)
- VICRegL, 2022 [paper](https://arxiv.org/abs/2210.01571) [docs](https://docs.lightly.ai/self-supervised-learning/examples/vicregl.html)


## Tutorials

Want to jump to the tutorials and see Lightly in action?

- [Train MoCo on CIFAR-10](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_moco_memory_bank.html)
- [Train SimCLR on Clothing Data](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simclr_clothing.html)
- [Train SimSiam on Satellite Images](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_simsiam_esa.html)
- [Use Lightly with Custom Augmentations](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_custom_augmentations.html)
- [Pre-train a Detectron2 Backbone with Lightly](https://docs.lightly.ai/self-supervised-learning/tutorials/package/tutorial_pretrain_detectron2.html)

Tutorials for the Lightly Solution (Lightly Worker & API):

- [General Docs of Lightly Solution](https://docs.lightly.ai)
- [Active Learning Using YOLOv7 and Comma10k](https://docs.lightly.ai/docs/active-learning-yolov7)
- [Active Learning for Driveable Area Segmentation Using Cityscapes](https://docs.lightly.ai/docs/active-learning-for-driveable-area-segmentation-using-cityscapes)
- [Active Learning for Transactions of Images](https://docs.lightly.ai/docs/active-learning-for-transactions-of-images)
- [Improving YOLOv8 using Active Learning on Videos](https://docs.lightly.ai/docs/active-learning-yolov8-video)
- [Assertion-based Active Learning with YOLOv8](https://docs.lightly.ai/docs/assertion-based-active-learning-tutorial)
- and more ...


Community and partner projects:

- [On-Device Deep Learning with Lightly on an ARM microcontroller](https://github.com/ARM-software/EndpointAI/tree/master/ProofOfConcepts/Vision/OpenMvMaskDefaults)


## Quick Start

Lightly requires **Python 3.6+** but we recommend using **Python 3.7+**. We recommend installing Lightly in a **Linux** or **OSX** environment.

### Dependencies

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [PyTorch Lightning](https://www.pytorchlightning.ai/index.html) v1.5+

Lightly is compatible with PyTorch and PyTorch Lightning v2.0+!

Vision transformer based models require Torchvision v0.12+.

### Installation
You can install Lightly and its dependencies from PyPI with:
```
pip3 install lightly
```

We strongly recommend that you install Lightly in a dedicated virtualenv, to avoid
conflicting with your system packages.

If you only want to install the API client without torch and torchvision dependencies
follow the docs on [how to install the Lightly Python Client](https://docs.lightly.ai/docs/install-lightly#install-the-lightly-python-client).


### Lightly in Action

With Lightly, you can use the latest self-supervised learning methods in a modular
way using the full power of PyTorch. Experiment with different backbones,
models, and loss functions. The framework has been designed to be easy to use
from the ground up. [Find more examples in our docs](https://docs.lightly.ai/self-supervised-learning/examples/models.html).

```python
import torch
import torchvision

from lightly import loss
from lightly import transforms
from lightly.data import LightlyDataset
from lightly.data.multi_view_collate import MultiViewCollate
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


# Use a resnet backbone.
backbone = torchvision.models.resnet18()
# Ignore the classification head as we only want the features.
backbone.fc = torch.nn.Identity()

# Build the SimCLR model.
model = SimCLR(backbone)

# Prepare transform that creates multiple random views for every image.
transform = transforms.SimCLRTransform(input_size=32, cj_prob=0.5)

# Combine views from multiple images into a batch.
collate_fn = MultiViewCollate()

# Create a dataset from your image folder.
dataset = data.LightlyDataset(input_dir="./my/cute/cats/dataset/", transform=transform)

# Build a PyTorch dataloader.
dataloader = torch.utils.data.DataLoader(
    dataset,  # Pass the dataset to the dataloader.
    batch_size=128,  # A large batch size helps with the learning.
    shuffle=True,  # Shuffling is important!
    collate_fn=collate_fn,
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

You can easily use another model like SimSiam by swapping the model and the
loss function.

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

You can [find a more complete example for SimSiam here.](https://docs.lightly.ai/self-supervised-learning/examples/simsiam.html)


Use PyTorch Lightning to train the model:

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

See [our docs for a full PyTorch Lightning example.](https://docs.lightly.ai/self-supervised-learning/examples/simclr.html)

Or train the model on 4 GPUs:
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

We provide multi-GPU training examples with distributed gather and synchronized BatchNorm.
[Have a look at our docs regarding distributed training.](https://docs.lightly.ai/self-supervised-learning/getting_started/distributed_training.html)


## Benchmarks

Implemented models and their performance on various datasets. Hyperparameters are not
tuned for maximum accuracy. For detailed results and more info about the benchmarks click
[here](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html).


### Imagenet

| Model       | Backbone | Batch Size | Epochs | Linear Top1 | Linear Top1 Online | KNN Top1 | Tensorboard | Checkpoint |
|-------------|----------|------------|--------|-------------|--------------------|----------|-------------|------------|
| SimCLR      | Res50    |        256 |    100 |        63.2 |               63.1 |     44.9 |      [link](https://tensorboard.dev/experiment/JwNs9E02TeeQkS7aljh8dA) |       [link](https://lightly-ssl-checkpoints.s3.amazonaws.com/imagenet_resnet50_simclr_2023-05-04_09-02-54/pretrain/version_0/checkpoints/epoch%3D99-step%3D500400.ckpt) |


### ImageNette

| Model       | Backbone | Batch Size | Epochs | KNN Top1 |
|-------------|----------|------------|--------|----------|
| BarlowTwins | Res18    |        256 |    800 |    0.852 |
| BYOL        | Res18    |        256 |    800 |    0.887 |
| DCL         | Res18    |        256 |    800 |    0.861 |
| DCLW        | Res18    |        256 |    800 |    0.865 |
| DINO        | Res18    |        256 |    800 |    0.888 |
| FastSiam    | Res18    |        256 |    800 |    0.873 |
| MAE         | ViT-S    |        256 |    800 |    0.610 |
| MSN         | ViT-S    |        256 |    800 |    0.828 |
| Moco        | Res18    |        256 |    800 |    0.874 |
| NNCLR       | Res18    |        256 |    800 |    0.884 |
| PMSN        | ViT-S    |        256 |    800 |    0.822 |
| SimCLR      | Res18    |        256 |    800 |    0.889 |
| SimMIM      | ViT-B32  |        256 |    800 |    0.343 |
| SimSiam     | Res18    |        256 |    800 |    0.872 |
| SwaV        | Res18    |        256 |    800 |    0.902 |
| SwaVQueue   | Res18    |        256 |    800 |    0.890 |
| SMoG        | Res18    |        256 |    800 |    0.788 |
| TiCo        | Res18    |        256 |    800 |    0.856 |
| VICReg      | Res18    |        256 |    800 |    0.845 |
| VICRegL     | Res18    |        256 |    800 |    0.778 |


### Cifar10

| Model       | Backbone | Batch Size | Epochs | KNN Top1 |
|-------------|----------|------------|--------|----------|
| BarlowTwins | Res18    |        512 |    800 |    0.859 |
| BYOL        | Res18    |        512 |    800 |    0.910 |
| DCL         | Res18    |        512 |    800 |    0.874 |
| DCLW        | Res18    |        512 |    800 |    0.871 |
| DINO        | Res18    |        512 |    800 |    0.848 |
| FastSiam    | Res18    |        512 |    800 |    0.902 |
| Moco        | Res18    |        512 |    800 |    0.899 |
| NNCLR       | Res18    |        512 |    800 |    0.892 |
| SimCLR      | Res18    |        512 |    800 |    0.879 |
| SimSiam     | Res18    |        512 |    800 |    0.904 |
| SwaV        | Res18    |        512 |    800 |    0.884 |
| SMoG        | Res18    |        512 |    800 |    0.800 |


## Terminology

Below you can see a schematic overview of the different concepts in the package.
The terms in bold are explained in more detail in our [documentation](https://docs.lightly.ai/self-supervised-learning/).

<img src="/docs/source/getting_started/images/lightly_overview.png" alt="Overview of the Lightly pip package"/></a>


### Next Steps

Head to the [documentation](https://docs.lightly.ai) and see the things you can achieve with Lightly!


## Development

To install dev dependencies (for example to contribute to the framework) you can use the following command:
```
pip3 install -e ".[dev]"
```

For more information about how to contribute have a look [here](CONTRIBUTING.md).


### Running Tests

Unit tests are within the [tests directory](tests/) and we recommend running them using 
[pytest](https://docs.pytest.org/en/stable/). There are two test configurations
available. By default, only a subset will be run:
```
make test-fast
```

To run all tests (including the slow ones) you can use the following command:
```
make test
```

To test a specific file or directory use:
```
pytest <path to file or directory>
```


### Code Formatting

To format code with [black](https://black.readthedocs.io/en/stable/) and [isort](https://docs.pytest.org) run:
```
make format
```


## Further Reading

**Self-Supervised Learning**:
- Have a look at our [#papers channel on discord](https://discord.com/channels/752876370337726585/815153188487299083)
  for the newest self-supervised learning papers.
- [A Cookbook of Self-Supervised Learning, 2023](https://arxiv.org/abs/2304.12210)
- [Masked Autoencoders Are Scalable Vision Learners, 2021](https://arxiv.org/abs/2111.06377)
- [Emerging Properties in Self-Supervised Vision Transformers, 2021](https://arxiv.org/abs/2104.14294)
- [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments, 2021](https://arxiv.org/abs/2006.09882)
- [What Should Not Be Contrastive in Contrastive Learning, 2020](https://arxiv.org/abs/2008.05659)
- [A Simple Framework for Contrastive Learning of Visual Representations, 2020](https://arxiv.org/abs/2002.05709)
- [Momentum Contrast for Unsupervised Visual Representation Learning, 2020](https://arxiv.org/abs/1911.05722)


## FAQ

- Why should I care about self-supervised learning? Aren't pre-trained models from ImageNet much better for transfer learning?
  - Self-supervised learning has become increasingly popular among scientists over the last years because the learned representations perform extraordinarily well on downstream tasks. This means that they capture the important information in an image better than other types of pre-trained models. By training a self-supervised model on *your* dataset, you can make sure that the representations have all the necessary information about your images.

- How can I contribute?
  - Create an issue if you encounter bugs or have ideas for features we should implement. You can also add your own code by forking this repository and creating a PR. More details about how to contribute with code is in our [contribution guide](CONTRIBUTING.md).

- Is this framework for free?
  - Yes, this framework is completely free to use and we provide the source code. We believe that we need to make training deep learning models more data efficient to achieve widespread adoption. One step to achieve this goal is by leveraging self-supervised learning. The company behind Lightly is committed to keep this framework open-source.

- If this framework is free, how is the company behind Lightly making money?
  - Training self-supervised models is only one part of our solution. 
  [The company behind Lightly](https://lightly.ai/) focuses on processing and analyzing embeddings created by self-supervised models. 
  By building, what we call a self-supervised active learning loop we help companies understand and work with their data more efficiently. 
  As the [Lightly Solution](https://docs.lightly.ai) is a freemium product, you can try it out for free. However, we will charge for some features.
  - In any case this framework will always be free to use, even for commercial purposes.


## Lightly in Research

- [Learning Visual Representations via Language-Guided Sampling, 2023](https://arxiv.org/pdf/2302.12248.pdf)
- [Self-Supervised Learning Methods for Label-Efficient Dental Caries Classification, 2022](https://www.mdpi.com/2075-4418/12/5/1237)
- [DPCL: Constrative Representation Learning with Differential Privacy, 2022](https://assets.researchsquare.com/files/rs-1516950/v1_covered.pdf?c=1654486158)
- [Decoupled Contrastive Learning, 2021](https://arxiv.org/abs/2110.06848)
- [solo-learn: A Library of Self-supervised Methods for Visual Representation Learning, 2021](https://www.jmlr.org/papers/volume23/21-1155/21-1155.pdf)


## BibTeX
If you want to cite the framework feel free to use this:

```bibtex
@article{susmelj2020lightly,
  title={Lightly},
  author={Igor Susmelj and Matthias Heller and Philipp Wirth and Jeremy Prescott and Malte Ebner et al.},
  journal={GitHub. Note: https://github.com/lightly-ai/lightly},
  year={2020}
}
```
