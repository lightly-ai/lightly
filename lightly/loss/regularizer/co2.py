""" CO2 Regularizer """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
from lightly.loss.memory_bank import MemoryBankModule


class CO2Regularizer(MemoryBankModule):
    """Implementation of the CO2 regularizer [0] for self-supervised learning.

    [0] CO2, 2021, https://arxiv.org/abs/2010.02217

    Attributes:
        alpha:
            Weight of the regularization term.
        t_consistency:
            Temperature used during softmax calculations.
        memory_bank_size:
            Number of negative samples to store in the memory bank.
            Use 0 to use the second batch for negative samples.

    Examples:
        >>> # initialize loss function for MoCo
        >>> loss_fn = NTXentLoss(memory_bank_size=4096)
        >>>
        >>> # initialize CO2 regularizer
        >>> co2 = CO2Regularizer(alpha=1.0, memory_bank_size=4096)
        >>>
        >>> # generate two random trasnforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through the MoCo model
        >>> out0, out1 = model(t0, t1)
        >>> 
        >>> # calculate loss and apply regularizer
        >>> loss = loss_fn(out0, out1) + co2(out0, out1)

    """

    def __init__(self,
                alpha: float = 1,
                t_consistency: float = 0.05,
                memory_bank_size: int = 0):

        super(CO2Regularizer, self).__init__(size=memory_bank_size)
        self.kl_div = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.t_consistency = t_consistency
        self.alpha = alpha

    def _get_pseudo_labels(self,
                           out0: torch.Tensor,
                           out1: torch.Tensor,
                           negatives: torch.Tensor = None):
        """Computes the soft pseudo labels across negative samples.

        Args:
            out0:
                Output projections of the first set of transformed images (query).
                Shape: bsz x n_ftrs
            out1:
                Output projections of the second set of transformed images (positive sample).
                Shape: bsz x n_ftrs
            negatives:
                Negative samples to compare against. If this is None, the second
                batch of images will be used as negative samples.
                Shape: memory_bank_size x n_ftrs

        Returns:
            Log probability that a positive samples will classify each negative
            sample as the positive sample.
            Shape: bsz x (bsz - 1) or bsz x memory_bank_size

        """
        batch_size, _ = out0.shape
        if negatives is None:
            # use second batch as negative samples
            # l_pos has shape bsz x 1 and l_neg has shape bsz x bsz
            l_pos = torch.einsum('nc,nc->n', [out0, out1]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [out0, out1.t()])
            # remove elements on the diagonal
            # l_neg has shape bsz x (bsz - 1)
            l_neg = l_neg.masked_select(
                ~torch.eye(batch_size, dtype=bool, device=l_neg.device)
            ).view(batch_size, batch_size - 1)
        else:
            # use memory bank as negative samples
            # l_pos has shape bsz x 1 and l_neg has shape bsz x memory_bank_size
            negatives = negatives.to(out0.device)
            l_pos = torch.einsum('nc,nc->n', [out0, out1]).unsqueeze(-1)
            l_neg = torch.einsum('nc,ck->nk', [out0, negatives.clone().detach()])
            
        # concatenate such that positive samples are at index 0
        logits = torch.cat([l_pos, l_neg], dim=1)
        # divide by temperature
        logits = logits / self.t_consistency

        # the input to kl_div is expected to be log(p) and we set the
        # flag log_target to True, so both probabilities should be passed as log
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs


    def forward(self,
                out0: torch.Tensor,
                out1: torch.Tensor):
        """Computes the CO2 regularization term for two model outputs.

        Args:
            out0:
                Output projections of the first set of transformed images.
            out1:
                Output projections of the second set of transformed images.

        Returns:
            The regularization term multiplied by the weight factor alpha.

        """

        # normalize the output to length 1
        out0 = torch.nn.functional.normalize(out0, dim=1)
        out1 = torch.nn.functional.normalize(out1, dim=1)

        # ask memory bank for negative samples and extend it with out1 if 
        # out1 requires a gradient, otherwise keep the same vectors in the 
        # memory bank (this allows for keeping the memory bank constant e.g.
        # for evaluating the loss on the test set)
        # if the memory_bank size is 0, negatives will be None
        out1, negatives = \
            super(CO2Regularizer, self).forward(out1, update=True)
        
        # get log probabilities
        p = self._get_pseudo_labels(out0, out1, negatives)
        q = self._get_pseudo_labels(out1, out0, negatives)
        
        # calculate kullback leibler divergence from log probabilities
        return self.alpha * 0.5 * (self.kl_div(p, q) + self.kl_div(q, p))
