"""SplitBatchNorm supports absent affine parameters like BatchNorm2d."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from lightly.models.batchnorm import SplitBatchNorm


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_splits", [1, 2, 4])
@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("affine", [False, True])
def test_split_batchnorm_optional_affine(
    dtype: torch.dtype, num_splits: int, training: bool, affine: bool
) -> None:
    """Match functional BatchNorm values, gradients, and running statistics."""
    torch.manual_seed(42)
    layer = SplitBatchNorm(num_features=3, num_splits=num_splits, affine=affine).to(
        dtype=dtype
    )
    means = torch.linspace(-0.5, 0.5, 3 * num_splits, dtype=dtype).reshape(
        num_splits, 3
    )
    variances = torch.linspace(1.0, 2.0, 3 * num_splits, dtype=dtype).reshape(
        num_splits, 3
    )
    assert layer.running_mean is not None
    assert layer.running_var is not None
    layer.running_mean.copy_(means.flatten())
    layer.running_var.copy_(variances.flatten())
    layer.train(training)
    parameters = dict(layer.named_parameters())
    weight_parameter = parameters.get("weight")
    offset_parameter = parameters.get("bias")
    weight = (
        weight_parameter.detach().clone().requires_grad_()
        if weight_parameter is not None
        else None
    )
    offset = (
        offset_parameter.detach().clone().requires_grad_()
        if offset_parameter is not None
        else None
    )
    inputs = torch.randn(8, 3, 4, 5, dtype=dtype, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    expected = torch.empty_like(reference_inputs)
    if training:
        for split in range(num_splits):
            expected[split::num_splits] = nn.functional.batch_norm(
                input=reference_inputs[split::num_splits],
                running_mean=means[split],
                running_var=variances[split],
                weight=weight,
                bias=offset,
                training=True,
                momentum=0.1,
                eps=layer.eps,
            )
        expected_mean = means.flatten()
        expected_variance = variances.flatten()
    else:
        expected = nn.functional.batch_norm(
            input=reference_inputs,
            running_mean=means.mean(dim=0),
            running_var=variances.mean(dim=0),
            weight=weight,
            bias=offset,
            training=False,
            eps=layer.eps,
        )
        expected_mean = means.mean(dim=0).repeat(num_splits)
        expected_variance = variances.mean(dim=0).repeat(num_splits)

    actual = layer(inputs)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(layer.running_mean, expected_mean)
    torch.testing.assert_close(layer.running_var, expected_variance)
    gradients = torch.randn_like(actual)
    (actual * gradients).sum().backward()
    (expected * gradients).sum().backward()
    torch.testing.assert_close(inputs.grad, reference_inputs.grad)
    if weight_parameter is None:
        assert weight is None
    else:
        assert weight is not None
        torch.testing.assert_close(weight_parameter.grad, weight.grad)
    if offset_parameter is None:
        assert offset is None
    else:
        assert offset is not None
        torch.testing.assert_close(offset_parameter.grad, offset.grad)
