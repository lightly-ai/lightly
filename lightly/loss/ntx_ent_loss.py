""" Contrastive Loss Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import numpy as np

from lightly.loss.memory_bank import MemoryBankModule


class NTXentLoss(MemoryBankModule):
    """Implementation of the Contrastive Cross Entropy Loss.

    This implementation follows the SimCLR[0] paper. If you enable the memory
    bank by setting the `memory_bank_size` value > 0 the loss behaves like 
    the one described in the MoCo[1] paper.

    [0] SimCLR, 2020, https://arxiv.org/abs/2002.05709
    [1] MoCo, 2020, https://arxiv.org/abs/1911.05722
    
    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        use_cosine_similarity:
            Whether to use cosine similarity over L2 distance.
        memory_bank_size:
            Number of negative samples to store in the memory bank. 
            Use 0 for SimCLR. For MoCo we typically use numbers like 4096 or 65536.

    Raises:
        ValueError if abs(temperature) < 1e-8 to prevent divide by zero.

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
                use_cosine_similarity: bool = True,
                memory_bank_size: int = 0):
        super(NTXentLoss, self).__init__(size=memory_bank_size)
        self.temperature = temperature
        self.similarity_function = self._get_similarity_function(
                                        use_cosine_similarity)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="mean")
        self.correlated_mask = None
        self.eps = 1e-8

        if abs(self.temperature) < self.eps:
            raise ValueError('Illegal temperature: abs({}) < 1e-8'
                             .format(self.temperature))

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _torch_get_correlated_mask(self, batch_size, device=None):
        diag = torch.eye(2 * batch_size, device=device)
        diag[batch_size:, :batch_size] += torch.eye(batch_size, device=device)
        diag[:batch_size, batch_size:] += torch.eye(batch_size, device=device)
        mask = (1 - diag).type(torch.bool)
        return mask

    def _get_correlated_mask(self, batch_size):
        # TODO: deprecate
        diag = np.eye(2 * batch_size)
        l1 = np.eye((2 * batch_size), 2 * batch_size, k=-batch_size)
        l2 = np.eye((2 * batch_size), 2 * batch_size, k=batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        if torch.cuda.is_available():
            mask.to("cuda")
        return mask

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

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
                out1:
                    Output projections of the second set of transformed images.

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
        out1, negatives = \
            super(NTXentLoss, self).forward(out1, update=out0.requires_grad)

        if negatives is not None:
            negatives = negatives.to(device)
            # use negatives from memory bank
            l_pos = torch.einsum('nc,nc->n', [out0, out1]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [out0, negatives.clone().detach()])
        else:
            # use other samples from batch as negatives
            output = torch.cat((out0, out1), axis=0)
            similarity_matrix = self.similarity_function(output, output)

            # filter out the scores from the positive samples
            l_pos = torch.diag(similarity_matrix, batch_size)
            r_pos = torch.diag(similarity_matrix, -batch_size)
            l_pos = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

            if self.correlated_mask is None:
                self.correlated_mask = \
                    self._torch_get_correlated_mask(batch_size, device=device)
            if 2 * batch_size != self.correlated_mask.shape[0]:
                self.correlated_mask = \
                    self._torch_get_correlated_mask(batch_size, device=device)

            l_neg = similarity_matrix[
                self.correlated_mask].view(2 * batch_size, -1)

        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= self.temperature

        labels = torch.zeros(logits.shape[0]).long()
        loss = self.cross_entropy(logits, labels.to(device))

        return loss
