import torch

@torch.no_grad()
def std_of_l2_normalized(z: torch.Tensor):
    """Calculates the mean of the standard deviation of z along each dimension.

    This measure was used by [0] to determine the level of collapse of the
    learned representations. If the returned number is 0., the outputs z have
    collapsed to a constant vector. "If the output z has a zero-mean isotropic
    Gaussian distribution" [0], the returned number should be close to 1/sqrt(d)
    where d is the dimensionality of the output.

    [0]: https://arxiv.org/abs/2011.10566

    Args:
        z:
            A torch tensor of shape batch_size x dimension.

    Returns:
        The mean of the standard deviation of the l2 normalized tensor z along
        each dimension.
    
    """

    if len(z.shape) != 2:
        raise ValueError(
            f'Input tensor must have two dimensions but has {len(z.shape)}!'
        )

    _, d = z.shape

    z_norm = torch.nn.functional.normalize(z, dim=1)
    return torch.std(z_norm, dim=0).mean()
