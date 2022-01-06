""" Contrastive Loss Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import torch.nn as nn
import torch.distributed as dist


from lightly.loss.memory_bank import MemoryBankModule
from lightly.loss.gather import GatherLayer


class NTXentLoss(MemoryBankModule):
    """Implementation of the Contrastive Cross Entropy Loss.

    This implementation follows the SimCLR[0] paper. If you enable the memory
    bank by setting the `memory_bank_size` value > 0 the loss behaves like 
    the one described in the MoCo[1] paper.

    - [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709
    - [1] MoCo, 2020, https://arxiv.org/abs/1911.05722
    
    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        memory_bank_size:
            Number of negative samples to store in the memory bank. 
            Use 0 for SimCLR. For MoCo we typically use numbers like 4096 or 65536.

    Raises:
        ValueError: If abs(temperature) < 1e-8 to prevent divide by zero.

    Examples:

        >>> # initialize loss function without memory bank
        >>> loss_fn = NTXentLoss(memory_bank_size=0)
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimCLR or MoCo model
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(output)

    """

    def __init__(self,
                 temperature: float = 0.5,
                 memory_bank_size: int = 0):
        super(NTXentLoss, self).__init__(size=memory_bank_size)
        self.temperature = temperature
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        self.eps = 1e-8

        if abs(self.temperature) < self.eps:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'
                             .format(self.temperature))

    def forward(self,
                out0: torch.Tensor,
                out1: torch.Tensor):
        """Forward pass through Contrastive Cross-Entropy Loss.

        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as 
        negative samples.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)

        Returns:
            Contrastive Cross Entropy Loss value.

        """

        device = out0.device
        batch_size, _ = out0.shape

        # normalize the output to length 1
        out0 = torch.nn.functional.normalize(out0, dim=1)
        out1 = torch.nn.functional.normalize(out1, dim=1)

        # ask memory bank for negative samples and extend it with out1 if 
        # out1 requires a gradient, otherwise keep the same vectors in the 
        # memory bank (this allows for keeping the memory bank constant e.g.
        # for evaluating the loss on the test set)
        # out1: shape: (batch_size, embedding_size)
        # negatives: shape: (embedding_size, memory_bank_size)
        out1, negatives = \
            super(NTXentLoss, self).forward(out1, update=out0.requires_grad)

        # We use the cosine similarity, which is a dot product (einsum) here,
        # as all vectors are already normalized to unit length.
        # Notation in einsum: n = batch_size, c = embedding_size and k = memory_bank_size.

        if negatives is not None:
            # use negatives from memory bank
            negatives = negatives.to(device)


            # sim_pos is of shape (batch_size, 1) and sim_pos[i] denotes the similarity
            # of the i-th sample in the batch to its positive pair
            sim_pos = torch.einsum('nc,nc->n', out0, out1).unsqueeze(-1)

            # sim_neg is of shape (batch_size, memory_bank_size) and sim_neg[i,j] denotes the similarity
            # of the i-th sample to the j-th negative sample
            sim_neg = torch.einsum('nc,ck->nk', out0, negatives)

            # set the labels to the first "class", i.e. sim_pos,
            # so that it is maximized in relation to sim_neg
            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
            labels = torch.zeros(logits.shape[0], device=device, dtype=torch.long)

        else:
            
            if dist.is_initialized() and dist.get_world_size() > 1:
                # gather hidden representations from other processes
                out0_large = torch.cat(GatherLayer.apply(out0), 0)
                out1_large = torch.cat(GatherLayer.apply(out1), 0)
                rank = dist.get_rank()
            else:
                # TODO
                out0_large = out0
                out1_large = out1
                rank = 0

            # TODO
            logits_aa = torch.einsum('nc,mc->nm', out0, out0_large) / self.temperature
            logits_ab = torch.einsum('nc,mc->nm', out0, out1_large) / self.temperature
            logits_ba = torch.einsum('nc,mc->nm', out1, out0_large) / self.temperature
            logits_bb = torch.einsum('nc,mc->nm', out1, out1_large) / self.temperature

            # initialize labels and masks
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            labels = labels + rank * batch_size
            masks = torch.ones_like(logits_aa).bool()
            masks = masks.scatter_(0, labels.unsqueeze(0), False)

            # remove similarities of samples to themselves
            logits_aa = logits_aa[masks].view(batch_size, -1)
            logits_bb = logits_bb[masks].view(batch_size, -1)

            # TODO
            logits_abaa = torch.cat([logits_ab, logits_aa], 1)
            logits_babb = torch.cat([logits_ba, logits_bb], 1)
            logits = torch.cat([logits_abaa, logits_babb], 0)
            labels = torch.cat([labels, labels])

        loss = self.cross_entropy(logits, labels)

        return loss
