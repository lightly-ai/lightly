""" Embedding Strategies """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import time

import torch
from lightly import is_prefetch_generator_available
from lightly.embedding._base import BaseEmbedding
from tqdm import tqdm

if is_prefetch_generator_available():
    from prefetch_generator import BackgroundGenerator


class SelfSupervisedEmbedding(BaseEmbedding):
    """Implementation of self-supervised embedding models.

    Implements an embedding strategy based on self-supervised learning. A 
    model backbone, self-supervised criterion, optimizer, and dataloader are 
    passed to the constructor. The embedding itself is a pytorch-lightning 
    module which can be trained very easily:

    https://pytorch-lightning.readthedocs.io/en/stable/

    The implementation is based on Contrastive Multiview Coding and SimCLR.

    https://arxiv.org/abs/1906.05849

    https://arxiv.org/abs/2002.05709

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

    def __init__(self,
                 model: torch.nn.Module,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 dataloader: torch.utils.data.DataLoader,
                 scheduler = None):

        super(SelfSupervisedEmbedding, self).__init__(
            model, criterion, optimizer, dataloader, scheduler)

    def embed(self,
              dataloader: torch.utils.data.DataLoader,
              device: torch.device = None,
              to_numpy: bool = True):
        """Embeds images in a vector space.

        Args:
            dataloader:
                A torchvision dataloader.
            device:
                Selected device (see PyTorch documentation)
            to_numpy:
                Whether to return the embeddings as numpy array.
        
        Returns:
            A tensor or ndarray of embeddings with shape n_images x num_ftrs

        Examples:
            >>> # embed images in vector space
            >>> embeddings, _, _ = encoder.embed(dataloader)

        """

        self.model.eval()
        embeddings, labels, fnames = None, None, []

        if is_prefetch_generator_available():
            pbar = tqdm(BackgroundGenerator(dataloader, max_prefetch=3),
                        total=len(dataloader))
        else:
            pbar = tqdm(dataloader, total=len(dataloader))

        start_time = time.time()
        with torch.no_grad():

            for (img, label, fname) in pbar:

                img = img.to(device)
                label = label.to(device)

                fnames += [*fname]

                batch_size = img.shape[0]
                prepare_time = time.time()

                emb = self.model.features(img)
                emb = emb.detach().reshape(batch_size, -1)

                if embeddings is None:
                    embeddings = emb
                else:
                    embeddings = torch.cat((embeddings, emb), 0)

                if labels is None:
                    labels = label
                else:
                    labels = torch.cat((labels, label), 0)

                process_time = time.time()

                pbar.set_description("Compute efficiency: {:.2f}".format(
                    process_time / (process_time + prepare_time)))

            if to_numpy:
                embeddings = embeddings.cpu().numpy()
                labels = labels.cpu().numpy()

        return embeddings, labels, fnames


class _VAEEmbedding(BaseEmbedding):
    """ Unsupervised embedding based on variational auto-encoders.

    """

    def embed(self, dataloader):
        """ TODO

        """
        raise NotImplementedError("This site is under construction...")
