import torch
import numpy as np

class SymmetrizedLoss(torch.nn.Module):
    """Implementation of the Symmetrized Loss.
    
    Examples:

        >>> # initialize loss function
        >>> loss_fn = SymmetrizedLoss
        >>>
        >>> # generate two random transforms of images
        >>> t0 = transforms(images)
        >>> t1 = transforms(images)
        >>>
        >>> # feed through SimSiam model
        >>> # returns z1, z2, p1, p2 concat in one tensor
        >>> batch = torch.cat((t0, t1), dim=0)
        >>> output = model(batch)  
        >>>
        >>> # calculate loss
        >>> loss = loss_fn(output)

    """

    def __init__(self):
        super(SymmetrizedLoss, self).__init__()

    def _neg_cosine_simililarity(self, x, y):
        """Computes the negative cosine similarity between the predictions x 
        and projections y.

        Args:
            x:
                Predictions with shape: n x d.
            y: 
                Projections with shape: n x d. Gradients wll not flow through this
                variable.

        Returns:
            [type]: [description]
        """
        v = - torch.nn.functional.cosine_similarity(x, y.detach(), dim=-1).mean()
        return v

    def forward(self, output: torch.Tensor, labels: torch.Tensor = None):
        """Forward pass through Symmetric Loss.

            Args:
                output:
                    Output from the model with shape: 2*bsz x d. Expects the order
                    of the batches to be z1, z2, p1, p2.
                labels:
                    Labels associated with the inputs.

            Returns:
                Contrastive Cross Entropy Loss value.

            Raises:
                ValueError if shape of output is not multiple of batch_size.
        """

        if output.shape[0] % 4:
            raise ValueError('Expected output of shape 4*bsz x dim but got '
                             f'shape {output.shape[0]} x {output.shape[1]}.')

        # device = output.device
        batch_size, dim = output.shape
        batch_size = batch_size // 4

        # normalize the output to length 1
        output = torch.nn.functional.normalize(output, dim=1)

        z1 = output[:batch_size]
        z2 = output[batch_size : batch_size * 2]
        p1 = output[batch_size * 2 : batch_size * 3]
        p2 = output[batch_size * 3 : batch_size * 4]
        
        loss = self._neg_cosine_simililarity(p1, z2) / 2 + \
               self._neg_cosine_simililarity(p2, z1) / 2

        return loss


    