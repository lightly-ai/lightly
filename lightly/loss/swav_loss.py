from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


@torch.no_grad()
def sinkhorn(out: torch.Tensor, iterations: int = 3, epsilon: float = 0.05):
    """Distributed sinkhorn algorithm.

    As outlined in [0] and implemented in [1].
    
    [0]: SwaV, 2020, https://arxiv.org/abs/2006.09882
    [1]: https://github.com/facebookresearch/swav/ 

    Args:
        out:
            Similarity of the features and the SwaV prototypes.
        iterations:
            Number of sinkhorn iterations.
        epsilon:
            Temperature parameter.

    Returns:
        Soft codes Q assigning each feature to a prototype.
    
    """

    # get the exponential matrix and make it sum to 1
    Q = torch.exp(out / epsilon).t()
    sum_Q = torch.sum(Q)
    Q /= sum_Q

    B = Q.shape[1]
    K = Q.shape[0] # number of prototypes

    for i in range(iterations):
        # normalize rows
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        Q /= sum_of_rows
        Q /= K
        # normalize columns
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B
    return Q.t()


class SwaVLoss(nn.Module):
    """Implementation of the SwaV loss.

    Attributes:
        temperature:
            Temperature parameter used for cross entropy calculations.
        sinkhorn_iterations:
            Number of iterations of the sinkhorn algorithm.
        sinkhorn_epsilon:
            Temperature parameter used in the sinkhorn algorithm.
    
    """

    def __init__(self,
                 temperature: float = 0.1,
                 sinkhorn_iterations: int = 3,
                 sinkhorn_epsilon: float = 0.05):
        super(SwaVLoss, self).__init__()
        self.temperature = temperature
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_epsilon = sinkhorn_epsilon


    def subloss(self, z: torch.Tensor, q: torch.Tensor):
        """Calculates the cross entropy for the SwaV prediction problem.

        Args:
            z:
                Similarity of the features and the SwaV prototypes.
            q:
                Codes obtained from Sinkhorn iterations.

        Returns:
            Cross entropy between predictions z and codes q.

        """
        return - torch.mean(
            torch.sum(q * F.log_softmax(z / self.temperature, dim=1), dim=1)
        )


    def forward(self,
                high_resolution_outputs: List[torch.Tensor],
                low_resolution_outputs: List[torch.Tensor]):
        """Computes the SwaV loss for a set of high and low resolution outputs.

        Args:
            high_resolution_outputs:
                List of similarities of features and SwaV prototypes for the
                high resolution crops.
            low_resolution_outputs:
                List of similarities of features and SwaV prototypes for the
                low resolution crops.

        Returns:
            Swapping assignments between views loss (SwaV) as described in [0].

        [0]: SwaV, 2020, https://arxiv.org/abs/2006.09882

        """
        n_crops = len(high_resolution_outputs) + len(low_resolution_outputs)

        # multi-crop iterations
        loss = 0.
        for i in range(len(high_resolution_outputs)):

            # compute codes of i-th high resolution crop
            with torch.no_grad():
                q = sinkhorn(
                    high_resolution_outputs[i].detach(),
                    iterations=self.sinkhorn_iterations,
                    epsilon=self.sinkhorn_epsilon
                )

            # compute subloss for each pair of crops
            subloss = 0.
            for v in range(len(high_resolution_outputs)):
                if v != i:
                    subloss += self.subloss(high_resolution_outputs[v], q)

            for v in range(len(low_resolution_outputs)):
                subloss += self.subloss(low_resolution_outputs[v], q)

            loss += subloss / (n_crops - 1)

        return loss / len(high_resolution_outputs)
