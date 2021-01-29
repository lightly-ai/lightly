
![Lightly Logo](docs/logos/lightly_logo_crop.png)


![GitHub](https://img.shields.io/github/license/lightly-ai/lightly)
![Unit Tests](https://github.com/lightly-ai/lightly/workflows/Unit%20Tests/badge.svg)

Lightly is a computer vision framework for self-supervised learning.

> We, at [Lightly](https://www.lightly.ai), are passionate engineers who want to make deep learning more efficient. We want to help popularize the use of self-supervised methods to understand and filter raw image data. Our solution can be applied before any data annotation step and the learned representations can be used to analyze and visualize datasets as well as for selecting a core set of samples.

- [Homepage](https://www.lightly.ai)
- [Web-App](https://app.lightly.ai)
- [Documentation](https://docs.lightly.ai)
- [Github](https://github.com/lightly-ai/lightly)
- [Discord](https://discord.gg/xvNJW94)


### Tutorials

Want to jump to the tutorials and see lightly in action?

- [Train MoCo on CIFAR-10](https://docs.lightly.ai/tutorials/package/tutorial_moco_memory_bank.html)
- [Train SimCLR on clothing data](https://docs.lightly.ai/tutorials/package/tutorial_simclr_clothing.html)
- [Train SimSiam on satellite images](https://docs.lightly.ai/tutorials/package/tutorial_simsiam_esa.html)


### Benchmarks

Currently implemented models and their accuracy on cifar10. All models have been evaluated using kNN. We report the max test accuracy over the epochs as well as the maximum GPU memory consumption. All models in this benchmark use the same augmentations as well as the same ResNet-18 backbone. Training precision is set to FP32 and SGD is used as an optimizer with cosineLR.

| Model   | Epochs | Batch Size | Test Accuracy | Peak GPU usage |
|---------|--------|------------|---------------|----------------|
| MoCo    |  200   | 128        | 0.83          | 2.1 GBytes     |
| SimCLR  |  200   | 128        | 0.78          | 2.0 GBytes     |
| SimSiam |  200   | 128        | 0.73          | 3.0 GBytes     |
| MoCo    |  200   | 512        | 0.85          | 7.4 GBytes     |
| SimCLR  |  200   | 512        | 0.83          | 7.8 GBytes     |
| SimSiam |  200   | 512        | 0.81          | 7.0 GBytes     |
| MoCo    |  800   | 512        | 0.90          | 7.2 GBytes     |
| SimCLR  |  800   | 512        | 0.89          | 7.7 GBytes     |
| SimSiam |  800   | 512        | 0.91          | 6.9 GBytes     |


## Terminology

Below you can see a schematic overview of the different concepts present in the lightly Python package. The terms in bold are explained in more detail in our [documentation](https://docs.lightly.ai).

<img src="docs/source/images/lightly_overview.png" alt="Overview of the lightly pip package"/></a>



## Quick Start

Lightly requires **Python 3.6+**. We recommend installing Lightly in a **Linux** or **OSX** environment.

### Requirements

- hydra-core>=1.0.0
- numpy>=1.18.1
- pytorch_lightning>=0.10.0   
- requests>=2.23.0
- torchvision
- tqdm

### Installation
You can install Lightly and its dependencies from PyPI with:
```
pip3 install lightly
```

We strongly recommend that you install Lightly in a dedicated virtualenv, to avoid conflicting with your system packages.

### Command-Line Interface

Lightly is accessible also through a command-line interface (CLI).
To train a SimCLR model on a folder of images you can simply run
the following command:

```
lightly-train input_dir=/mydataset
```

To create an embedding of a dataset you can use:

```
lightly-embed input_dir=/mydataset checkpoint=/mycheckpoint
```

The embeddings with the corresponding filename are stored in a human-readable .csv file.

### Next Steps
Head to the [documentation](https://docs.lightly.ai) and see the things you can achieve with Lightly!


## Development

To install dev dependencies (for example to contribute to the framework)
you can use the following command:
```
pip3 install -e ".[dev]"
```

For more information about how to contribute have a look [here](CONTRIBUTING.md).

### Running Tests

Unit tests are within the `tests` folder and we recommend to run them using 
[pytest](https://docs.pytest.org/en/stable/).
There are two test configurations available. By default only a subset will be run.
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
  - Training self-supervised models is only part of the solution. The company behind lightly focuses on processing and analyzing embeddings created by self-supervised models. 
  By building, what we call a self-supervised active learning loop we help companies understand and work with their data more efficiently. This framework acts as an interface
  for our platform to easily upload and download datasets, embeddings and models. Whereas 
  the platform will cost for additional features this frameworks will always remain free of charge (even for commercial use).