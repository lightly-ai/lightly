""" Embedding Strategies """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from lightly.data import LightlyDataset
from lightly.embedding._base import BaseEmbedding
from lightly.utils.benchmarking import BenchmarkModule
from lightly.utils.reordering import sort_items_by_keys

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SelfSupervisedEmbedding(BaseEmbedding):
    """Implementation of self-supervised embedding models.

    Implements an embedding strategy based on self-supervised learning. A
    model backbone, self-supervised criterion, optimizer, and dataloader are
    passed to the constructor. The embedding itself is a pytorch-lightning
    module.

    The implementation is based on contrastive learning.

    * SimCLR: https://arxiv.org/abs/2002.05709
    * MoCo: https://arxiv.org/abs/1911.05722
    * SimSiam: https://arxiv.org/abs/2011.10566

    Attributes:
        model:
            A backbone convolutional network with a projection head.
        criterion:
            A contrastive loss function.
        optimizer:
            A PyTorch optimizer.
        dataloader:
            A torchvision dataloader.
        scheduler:
            A PyTorch learning rate scheduler.

    Examples:
        >>> # define a model, criterion, optimizer, and dataloader above
        >>> import lightly.embedding as embedding
        >>> encoder = SelfSupervisedEmbedding(
        >>>     model,
        >>>     criterion,
        >>>     optimizer,
        >>>     dataloader,
        >>> )
        >>> # train the self-supervised embedding with default settings
        >>> encoder.train_embedding()
        >>> # pass pytorch-lightning trainer arguments as kwargs
        >>> encoder.train_embedding(max_epochs=10)

    """

    def __init__(
        self,
        model: BenchmarkModule,
        criterion: Module,
        optimizer: Optimizer,
        dataloader: DataLoader[LightlyDataset],
        scheduler: Optional[_LRScheduler] = None,
    ) -> None:
        super(SelfSupervisedEmbedding, self).__init__(
            model, criterion, optimizer, dataloader, scheduler
        )

    def embed(
        self,
        dataloader: DataLoader[LightlyDataset],
        device: Optional[torch.device] = None,
    ) -> Tuple[NDArray[np.float64], List[int], List[str]]:
        """Embeds images in a vector space.

        Args:
            dataloader:
                A PyTorch dataloader.
            device:
                Selected device (`cpu`, `cuda`, see PyTorch documentation)

        Returns:
            Tuple of (embeddings, labels, filenames) ordered by the
            samples in the dataset of the dataloader.
                embeddings:
                    Embedding of shape (n_samples, embedding_feature_size).
                    One embedding for each sample.
                labels:
                    Labels of shape (n_samples, ).
                filenames:
                    The filenames from dataloader.dataset.get_filenames().

        Examples:
            >>> # embed images in vector space
            >>> embeddings, labels, fnames = encoder.embed(dataloader)

        """

        self.model.eval()
        filenames = []

        dataset: LightlyDataset = dataloader.dataset

        pbar = tqdm(total=len(dataset), unit="imgs")

        efficiency = 0.0
        embeddings: List[NDArray[np.float64]] = []
        labels: List[int] = []
        with torch.no_grad():
            start_timepoint = time.time()
            for image_batch, label_batch, filename_batch in dataloader:
                batch_size = image_batch.shape[0]

                # the following 2 lines are needed to prevent a file handler leak,
                # see https://github.com/lightly-ai/lightly/pull/676
                image_batch = image_batch.to(device)
                label_batch = label_batch.clone()

                filenames += [*filename_batch]

                prepared_timepoint = time.time()

                embedding_batch = self.model.backbone(image_batch)
                embedding_batch = embedding_batch.detach().reshape(batch_size, -1)

                embeddings.extend(embedding_batch.cpu().numpy())
                labels.extend(label_batch.cpu().tolist())

                finished_timepoint = time.time()

                data_loading_time = prepared_timepoint - start_timepoint
                inference_time = finished_timepoint - prepared_timepoint
                total_batch_time = data_loading_time + inference_time

                efficiency = inference_time / total_batch_time
                pbar.set_description("Compute efficiency: {:.2f}".format(efficiency))
                start_timepoint = time.time()

                pbar.update(batch_size)

        sorted_filenames = dataset.get_filenames()
        sorted_embeddings = sort_items_by_keys(
            keys=filenames,
            items=embeddings,
            sorted_keys=sorted_filenames,
        )
        sorted_labels = sort_items_by_keys(
            keys=filenames, items=labels, sorted_keys=sorted_filenames
        )

        return np.stack(sorted_embeddings), sorted_labels, sorted_filenames
