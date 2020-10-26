
![Lightly Logo](docs/logos/lightly_logo_crop.png)

Lightly is a computer vision framework for self-supervised learning.

> We, at [Lightly](https://www.lightly.ai), are passionate engineers who want to make deep learning more efficient. We want to help popularize the use of self-supervised methods to understand and filter raw image data. Our solution can be applied before any data annotation step and the learned representations can be used to analyze and visualize datasets as well as for selecting a core set of samples.

- [Homepage](https://www.lightly.ai)
- [Web-App](https://app.lightly.ai)
- [Documentation](https://lightly.readthedocs.io)
- [Github](https://github.com/lightly-ai/lightly)
- [Discord](https://discord.gg/xvNJW94)

## Terminology
- **Dataset:** A collection of raw images.
- **Embedding:** Representation of an image in a vector space.
- **Embedding Model:** Function (typically a convolutional neural network) to create embeddings from images.
- **Self-supervised Learning:** A form of unsupervised learning where the data provides the supervision.

## Quick Start

Lightly requires Python 3.5+. We recommend installing Lightly in a **Linux** or **OSX** environment.

### Installation
You can install Lightly and its dependencies from PyPI with:
```
pip install lightly
```

We strongly recommend that you install Lightly in a dedicated virtualenv, to avoid conflicting with your system packages.

### Next Steps
Head to the [documentation](https://lightly.readthedocs.io) and see the things you can achieve with Lightly!


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
  - Create an issue if you encounter bugs or have ideas for features we should implement. You can also add your own code by forking this repository and creating a PR. More details about how to contribute with code is in our contribution guide.

- Is this framework for free?
  - Yes, this framework completely free to use and we provide the code. We believe that
  we need to make training deep learning models more data efficient to achieve widespread adoption. One step to achieve this goal is by leveraging self-supervised learning. The company behind lightly commited to keep this framework open-source.

- If this framework is free, how is the company behind lightly making money?
  - Training self-supervised models is only part of the solution. The company behind lightly focuses on processing and analyzing embeddings created by self-supervised models. 
  By building, what we call a self-supervised active learning loop we help companies understand and work with their data more efficiently. This framework acts as an interface
  for our platform to easily upload and download datasets, embeddings and models. Whereas 
  the platform will cost for additional features this frameworks will always remain free of charge (even for commercial use).