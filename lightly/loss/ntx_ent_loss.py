""" Contrastive Loss Functions """

# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

import torch
import numpy as np


class NTXentLoss(torch.nn.Module):
    """Implementation of the Contrastive Cross Entropy Loss.
    
    Attributes:
        temperature:
            Scale logits by the inverse of the temperature.
        use_cosine_similarity:
            Whether to use cosine similarity over L2 distance.

    Raises:
        ValueError if abs(temperature) < 1e-8 to prevent divide by zero.

    """

    def __init__(self,
                temperature: float = 0.5,
                use_cosine_similarity: bool = True):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.similarity_function = self._get_similarity_function(
                                        use_cosine_similarity)
        self.cross_entropy = torch.nn.CrossEntropyLoss(reduction="sum")
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

    def forward(self, output: torch.Tensor, labels: torch.Tensor = None):
        """Forward pass through Contrastive Cross Entropy Loss.

            Args:
                output:
                    Output from the model with shape: 2*bsz x d.
                labels:
                    Labels associated with the inputs.

            Returns:
                Contrastive Cross Entropy Loss value.

            Raises:
                ValueError if shape of output is not multiple of batch_size.
        """

        output = torch.nn.functional.normalize(output, dim=1)
        device = output.device

        if output.shape[0] % 2:
            raise ValueError('Illegal shape, axis 0 not divisible by 2: {}'
                             .format(output.shape))

        batch_size = output.shape[0] // 2
        similarity_matrix = self.similarity_function(output, output)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, batch_size)
        r_pos = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * batch_size, 1)

        if self.correlated_mask is None:
            self.correlated_mask = \
                self._torch_get_correlated_mask(batch_size, device=device)
        if 2 * batch_size != self.correlated_mask.shape[0]:
            self.correlated_mask = \
                self._torch_get_correlated_mask(batch_size, device=device)

        negatives = similarity_matrix[
            self.correlated_mask
            ].view(2 * batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * batch_size).long()

        loss = self.cross_entropy(logits, labels.to(device))
        return loss / (2 * batch_size)
