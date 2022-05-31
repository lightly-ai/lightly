import torch

@torch.no_grad()
def std_of_l2_normalized(z: torch.Tensor):
    """Calculates the mean of the standard deviation of z along each dimension.

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
