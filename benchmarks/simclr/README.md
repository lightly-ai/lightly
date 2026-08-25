# SimCLR

```bash
torchrun --nproc_per_node=8 -m benchmarks.simclr.benchmark \
    --train-dir /datasets/imagenet/train --val-dir /datasets/imagenet/val
```

```bash
python -m benchmarks.simclr.benchmark --dataset cifar10 \
    --train-dir ./cifar/train --val-dir ./cifar/val --devices 1 --strategy auto
```

`datasets.py` holds one row per dataset. Each row states every field, so two rows
read side by side show every difference there is. `benchmark.py` takes one row.

## Numbers

| Row | Backbone | Batch | Epochs | Linear top1 | kNN top1 | Produced by |
| :-- | :-- | --: | --: | --: | --: | :-- |
| imagenet | ResNet-50 | 4096 | 100 | — | — | not run yet |
| cifar10 | ResNet-18 | 256 | 100 | — | — | not run yet |

The number in the repository README — 63.2 linear, 73.9 kNN — came from
`benchmarks/imagenet/resnet50/simclr.py`, last touched in c7928120, on the run
logged as `imagenet_resnet50_simclr_2023-06-22_09-11-13`. That file is deleted
here and the row above has not reproduced it. Three things differ, so treat the
old number as a target rather than as this benchmark's result:

- **Batch size and learning rate.** The old run used 256 with square-root
  scaling. The `imagenet` row is the paper's 4096 at lr 4.8.
- **The online probe.** It ran on both views' features out of the SSL forward and
  its head rode the method's optimiser. Here it sees one view under `no_grad`, so
  `val_online_cls_top*` moves. The SSL loss is unaffected: the probe's input was
  already detached.
- **The kNN probe.** Same settings, k=200 and t=0.1, but `bench/probes.py`
  replaces the dataloader-index state machine with `reset`, `add`, `build` and
  `score`, called from `validation_step`.

## What is not here yet

`bench/probes.py` is the smallest thing that serves this benchmark.
`lightly.eval` is where a public, method-agnostic probe goes, and the online
linear probe still lives inside `benchmark.py` until then.
