"""SplitBatchNorm values, gradients, and statistics for strided inputs."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from lightly.models.batchnorm import SplitBatchNorm


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_splits", [1, 2, 4])
@pytest.mark.parametrize(
    "layout", ["contiguous", "channels_last", "batch_transpose", "spatial_transpose"]
)
def test_split_batchnorm_layout(
    dtype: torch.dtype, num_splits: int, layout: str
) -> None:
    """Match independent BatchNorm layers assigned interleaved batch samples."""
    torch.manual_seed(42)
    if layout == "batch_transpose":
        inputs = torch.randn(3, 8, 4, 5, dtype=dtype).transpose(0, 1)
    else:
        inputs = torch.randn(8, 3, 4, 5, dtype=dtype)
        if layout == "channels_last":
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        elif layout == "spatial_transpose":
            inputs = inputs.transpose(2, 3)
    inputs = inputs.detach().requires_grad_()
    reference_inputs = inputs.detach().contiguous().requires_grad_()
    original = inputs.detach().clone()
    layer = SplitBatchNorm(num_features=3, num_splits=num_splits).to(dtype=dtype)
    assert layer.weight is not None
    assert layer.bias is not None
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([0.5, 1.0, 1.5], dtype=dtype))
        layer.bias.copy_(torch.tensor([-0.2, 0.1, 0.3], dtype=dtype))
    references: list[nn.BatchNorm2d] = []
    expected = torch.empty_like(reference_inputs)
    for split in range(num_splits):
        reference = nn.BatchNorm2d(num_features=3).to(dtype=dtype)
        assert reference.weight is not None
        assert reference.bias is not None
        with torch.no_grad():
            reference.weight.copy_(layer.weight)
            reference.bias.copy_(layer.bias)
        expected[split::num_splits] = reference(reference_inputs[split::num_splits])
        references.append(reference)

    actual = layer(inputs)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(inputs, original)
    running_means: list[Tensor] = []
    running_variances: list[Tensor] = []
    for reference in references:
        assert reference.running_mean is not None
        assert reference.running_var is not None
        running_means.append(reference.running_mean)
        running_variances.append(reference.running_var)
    torch.testing.assert_close(layer.running_mean, torch.cat(running_means))
    torch.testing.assert_close(layer.running_var, torch.cat(running_variances))
    weights = torch.randn_like(actual)
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    weight_gradients: list[Tensor] = []
    bias_gradients: list[Tensor] = []
    for reference in references:
        assert reference.weight is not None and reference.weight.grad is not None
        assert reference.bias is not None and reference.bias.grad is not None
        weight_gradients.append(reference.weight.grad)
        bias_gradients.append(reference.bias.grad)
    torch.testing.assert_close(
        layer.weight.grad, torch.stack(weight_gradients).sum(dim=0)
    )
    torch.testing.assert_close(layer.bias.grad, torch.stack(bias_gradients).sum(dim=0))
    layer.eval()
    expected_eval = nn.functional.batch_norm(
        input=inputs,
        running_mean=torch.stack(running_means).mean(dim=0),
        running_var=torch.stack(running_variances).mean(dim=0),
        weight=layer.weight,
        bias=layer.bias,
        training=False,
    )
    torch.testing.assert_close(layer(inputs), expected_eval)
