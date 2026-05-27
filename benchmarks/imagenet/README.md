# ImageNet Benchmarks

Reference implementations for self-supervised learning (SSL) methods on ImageNet.
The benchmark scripts are organized by model family and are intended to be easy
to read, run, and adapt.

**Note**
> The benchmarks are still in beta phase and there will be breaking changes and
> frequent updates. PRs for new methods are highly welcome!

**Goals**
* Provide easy to use/adapt reference implementations of SSL methods.
* Keep methods self-contained while using Lightly building blocks. See
  [resnet50/simclr.py](resnet50/simclr.py) for a compact example.
* Remain as framework agnostic as possible. The benchmarks mainly rely on PyTorch and PyTorch Lightning; some ViT methods require `timm` (install via `pip install lightly[timm]`).

**Non-Goals**
* Lightly doesn't strive to be an end-to-end SSL framework with vast
  configuration options. Instead, we provide building blocks and examples to make
  it as easy as possible to build on top of existing SSL methods.

You can find benchmark results in our
[docs](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html).

## Available Benchmarks

| Directory | Backbone family | Methods |
| --- | --- | --- |
| [`resnet50/`](resnet50/) | ResNet50 | BarlowTwins, BYOL, DCL, DCLW, DINO, MoCoV2, SimCLR, SwAV, TiCo, VICReg |
| [`vitb16/`](vitb16/) | Vision Transformers | AIM, DINO, DINOv2, iBOT, MAE |

The `vitb16/` directory name is historical. Check the individual method files for
the exact ViT variant used by each method.

## Dataset Setup

Download the ImageNet ILSVRC2012 split from
https://www.image-net.org/challenges/LSVRC/2012/ and pass the training and
validation folders with `--train-dir` and `--val-dir`.

All examples below assume the following layout:

```text
/datasets/imagenet/train
/datasets/imagenet/val
```

## Run ResNet50 Benchmarks

From the repository root:

```bash
cd benchmarks/imagenet/resnet50
python main.py \
    --epochs 100 \
    --train-dir /datasets/imagenet/train \
    --val-dir /datasets/imagenet/val \
    --num-workers 12 \
    --devices 2 \
    --batch-size-per-device 128 \
    --skip-finetune-eval
```

To run only specific methods, use `--methods`:

```bash
python main.py --epochs 100 --batch-size-per-device 128 --methods simclr byol
```

## Run ViT Benchmarks

From the repository root:

```bash
cd benchmarks/imagenet/vitb16
python main.py \
    --epochs 100 \
    --train-dir /datasets/imagenet/train \
    --val-dir /datasets/imagenet/val \
    --num-workers 12 \
    --devices 2 \
    --batch-size-per-device 128 \
    --skip-finetune-eval
```

To run only specific methods, use `--methods`:

```bash
python main.py --epochs 100 --batch-size-per-device 128 --methods mae dino
```

ViT linear and fine-tuning evaluation supports different protocols via
`--eval-method`:

```bash
python main.py --eval-method mae      # default
python main.py --eval-method simclr
```

## Run with SLURM

Create a script such as `run_imagenet.sh` in the benchmark directory you want to
run (`resnet50/` or `vitb16/`):

```bash
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:2            # Must match --devices argument
#SBATCH --ntasks-per-node=2     # Must match --devices argument
#SBATCH --cpus-per-task=16      # Must be >= --num-workers argument
#SBATCH --mem=0

eval "$(conda shell.bash hook)"

conda activate lightly-env
srun python main.py \
    --epochs 100 \
    --train-dir /datasets/imagenet/train \
    --val-dir /datasets/imagenet/val \
    --num-workers 12 \
    --devices 2 \
    --batch-size-per-device 128
conda deactivate
```

Run it with:

```bash
sbatch run_imagenet.sh
```

## Configuration

Common flags supported by both benchmark families:

* `--methods`: Run a subset of methods.
* `--epochs`: Number of pretraining epochs. Set to `0` to skip pretraining.
* `--batch-size-per-device`: Batch size on each device.
* `--devices`: Number of devices used by PyTorch Lightning.
* `--accelerator`: Accelerator used by PyTorch Lightning. Defaults to `gpu`.
* `--precision`: PyTorch Lightning precision. Defaults to `16-mixed`.
* `--ckpt-path`: Load or resume from a checkpoint.
* `--num-classes`: Number of dataset classes. Defaults to `1000`.
* `--skip-knn-eval`: Skip kNN evaluation.
* `--skip-linear-eval`: Skip linear evaluation.
* `--skip-finetune-eval`: Skip fine-tuning evaluation.
* `--seed`: Set a random seed.

Training and evaluation steps can be skipped as follows:

```bash
python main.py --batch-size-per-device 128 \
    --epochs 0 \
    --skip-knn-eval \
    --skip-linear-eval \
    --skip-finetune-eval
```

Architecture-specific defaults differ slightly. For example, ResNet50 defaults
to `--knn-t 0.1`, while ViT defaults to `--knn-t 0.07` and adds the
`--eval-method` flag.

## ImageNet100

For ImageNet100, adapt the dataset location and set the number of classes to
100:

```bash
python main.py \
    --train-dir /datasets/imagenet100/train \
    --val-dir /datasets/imagenet100/val \
    --num-classes 100 \
    --epochs 100 \
    --num-workers 12 \
    --devices 2 \
    --batch-size-per-device 128
```

## Imagenette

For [Imagenette](https://github.com/fastai/imagenette), adapt the dataset
location and set the number of classes to 10:

```bash
python main.py \
    --train-dir /datasets/imagenette2-320/train \
    --val-dir /datasets/imagenette2-320/val \
    --num-classes 10 \
    --epochs 100 \
    --num-workers 12 \
    --devices 2 \
    --batch-size-per-device 128
```
