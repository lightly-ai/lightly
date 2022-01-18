""" Embedding Strategies """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import time
from typing import List, Union, Tuple

import numpy as np
import torch
import lightly
from lightly.embedding._base import BaseEmbedding
from tqdm import tqdm

if lightly._is_prefetch_generator_available():
    from prefetch_generator import BackgroundGenerator


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
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dataloader: torch.utils.data.DataLoader,
        scheduler=None,
    ):

        super(SelfSupervisedEmbedding, self).__init__(
            model, criterion, optimizer, dataloader, scheduler
        )

    def embed(self,
              dataloader: torch.utils.data.DataLoader,
              device: torch.device = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
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
        embeddings, labels, fnames = None, None, []

        if lightly._is_prefetch_generator_available():
            pbar = tqdm(
                BackgroundGenerator(dataloader, max_prefetch=3),
                total=len(dataloader)
            )
        else:
            pbar = tqdm(dataloader, total=len(dataloader))

        efficiency = 0.0
        embeddings = []
        labels = []
        with torch.no_grad():

            start_timepoint = time.time()
            for (img, label, fname) in pbar:

                img = img.to(device)

                fnames += [*fname]

                batch_size = img.shape[0]
                prepared_timepoint = time.time()

                emb = self.model.backbone(img)
                emb = emb.detach().reshape(batch_size, -1)

                embeddings.append(emb)
                labels.append(label)

                finished_timepoint = time.time()

                data_loading_time = prepared_timepoint - start_timepoint
                inference_time = finished_timepoint - prepared_timepoint
                total_batch_time = data_loading_time + inference_time

                efficiency = inference_time / total_batch_time
                pbar.set_description("Compute efficiency: {:.2f}".format(efficiency))
                start_timepoint = time.time()

            embeddings = torch.cat(embeddings, 0)
            labels = torch.cat(labels, 0)

            embeddings = embeddings.cpu().numpy()
            labels = labels.cpu().numpy()

        to_order = zip(embeddings, labels)
        lookup_to_order_by_filenames = dict(zip(fnames, to_order))

        filenames_correct_order = dataloader.dataset.get_filenames()
        ordered = [
            lookup_to_order_by_filenames[filename] for filename in filenames_correct_order
        ]

        embeddings_ordered, labels_ordered = zip(*ordered)
        embeddings = np.stack(embeddings_ordered)
        labels = np.stack(labels_ordered)

        return embeddings, labels, filenames_correct_order
