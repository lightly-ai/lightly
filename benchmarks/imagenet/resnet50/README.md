# ImageNet ResNet50

Reference implementations for self-supervised learning (SSL) methods on ImageNet with
ResNet50 backbones.

**Note**
> The benchmarks are still in beta phase and there will be breaking changes and
frequent updates. PRs for new methods are highly welcome!

**Goals**
* Provide easy to use/adapt reference implementations of SSL methods.
* Implemented methods should be self-contained and use the Lightly building blocks.
See [simclr.py](simclr.py).
* Remain as framework agnostic as possible. The benchmarks currently only rely on PyTorch and PyTorch Lightning.


**Non-Goals**
* Lightly doesn't strive to be an end-to-end SSL framework with vast configuration options.
Instead, we try to provide building blocks and examples to make it as easy as possible to
build on top of existing SSL methods.

You can find benchmark resuls in our [docs](https://docs.lightly.ai/self-supervised-learning/getting_started/benchmarks.html).

## Run Benchmark

To run the benchmark first download the ImageNet ILSVRC2012 split from here: https://www.image-net.org/challenges/LSVRC/2012/.


Then start the benchmark with:
```
python main.py --epochs 100 --train-dir /datasets/imagenet/train --val-dir /datasets/imagenet/val --num-workers 12 --devices 2 --batch-size 256 --skip-finetune-eval
```

Or with SLURM, create the following script (`run_imagenet.sh`):
```
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:2            # Must match --devices argument
#SBATCH --ntasks-per-node=2     # Must match --devices argument
#SBATCH --cpus-per-task=16      # Must be >= --num-workers argument
#SBATCH --mem=0

eval "$(conda shell.bash hook)"

conda lightly-env
srun python main.py --epochs 100 --train-dir /datasets/imagenet/train --val-dir /datasets/imagenet/val --num-workers 12 --devices 2 --batch-size 256
conda deactivate
```

And run it with sbatch: `sbatch run_imagenet.sh`.


## Configuration

To run the benchmark on specific methods use the `--methods` flag:
```
python main.py --epochs 100 --batch-size 256 --methods simclr byol
```

Training/evaluation steps can be skipped as follows:
```
python main.py --batch-size 256 \
    --epochs 0              # no pretraining
    --skip-knn-eval         # no KNN evaluation
    --skip-linear-eval      # no linear evaluation
    --skip-finetune-eval    # no finetune evaluation
```


## ImageNet100

For ImageNet100 you have to adapt the dataset location and set number of classes to 100:
```
python main.py --train-dir /datasets/imagenet100/train --val-dir /datasets/imagenet100/val --num-classes 100 --epochs 100 --num-workers 12 --devices 2 --batch-size 256
```


## Imagenette

For [Imagenette](https://github.com/fastai/imagenette) you have to adapt the dataset location and set number of classes to 10:

```
python main.py --train-dir /datasets/imagenette2-320/train --val-dir /datasets/imagenette2-320/val --num-classes 10 --epochs 100 --num-workers 12 --devices 2 --batch-size 256
```