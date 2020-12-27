import torch

class SymNegCosineSimilarityLoss(torch.nn.Module):
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

    def forward(self, 
                out0: torch.Tensor, 
                out1: torch.Tensor):
        """Forward pass through Symmetric Loss.

            Args:
                out0:
                    Output projections of the first set of transformed images.
                    Expects the tuple to be of the form (z0, p0), where z0 is
                    the output of the backbone and projection mlp, and p0 is 
                out1:
                    Output projections of the second set of transformed images.

            Returns:
                Contrastive Cross Entropy Loss value.

            Raises:
                ValueError if shape of output is not multiple of batch_size.
        """
        z0, p0 = out0
        z1, p1 = out1

        loss = self._neg_cosine_simililarity(p0, z1) / 2 + \
               self._neg_cosine_simililarity(p1, z0) / 2

        return loss
    