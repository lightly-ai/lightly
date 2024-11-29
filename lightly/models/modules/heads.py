""" Projection and Prediction Heads for Self-supervised Learning """

# Copyright (c) 2021. Lightly AG and its affiliates.
# All Rights Reserved

from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor

from lightly.models import utils


class ProjectionHead(nn.Module):
    """Base class for all projection and prediction heads.

    Args:
        blocks:
            List of tuples, each denoting one block of the projection head MLP.
            Each tuple reads (in_features, out_features, batch_norm_layer,
            non_linearity_layer, use_bias (optional)).

    Examples:
        >>> # the following projection head has two blocks
        >>> # the first block uses batch norm an a ReLU non-linearity
        >>> # the second block is a simple linear layer
        >>> projection_head = ProjectionHead([
        >>>     (256, 256, nn.BatchNorm1d(256), nn.ReLU()),
        >>>     (256, 128, None, None)
        >>> ])
    """

    def __init__(
        self,
        blocks: Sequence[
            Union[
                Tuple[int, int, Optional[nn.Module], Optional[nn.Module]],
                Tuple[int, int, Optional[nn.Module], Optional[nn.Module], bool],
            ],
        ],
    ) -> None:
        """Initializes the ProjectionHead module with the specified blocks."""
        super().__init__()

        layers: List[nn.Module] = []
        for block in blocks:
            input_dim, output_dim, batch_norm, non_linearity, *bias = block
            use_bias = bias[0] if bias else not bool(batch_norm)
            layers.append(nn.Linear(input_dim, output_dim, bias=use_bias))
            if batch_norm:
                layers.append(batch_norm)
            if non_linearity:
                layers.append(non_linearity)
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Computes one forward pass through the projection head.

        Args:
            x:
                Input of shape bsz x num_ftrs.
        """
        projection: Tensor = self.layers(x)
        return projection


class BarlowTwinsProjectionHead(ProjectionHead):
    """Projection head used for Barlow Twins.

    "The projector network has three linear layers, each with 8192 output
    units. The first two layers of the projector are followed by a batch
    normalization layer and rectified linear units." [0]

    - [0]: 2021, Barlow Twins, https://arxiv.org/abs/2103.03230
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 8192, output_dim: int = 8192
    ):
        """Initializes the BarlowTwinsProjectionHead with the specified dimensions.

        Args:
            input_dim:
                Dimensionality of the input features.
            hidden_dim:
                Dimensionality of the hidden layers.
            output_dim:
                Dimensionality of the output features.
        """
        super(BarlowTwinsProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class BYOLProjectionHead(ProjectionHead):
    """Projection head used for BYOL.

    "This MLP consists in a linear layer with output size 4096 followed by
    batch normalization, rectified linear units (ReLU), and a final
    linear layer with output dimension 256." [0]

    - [0]: BYOL, 2020, https://arxiv.org/abs/2006.07733
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 4096, output_dim: int = 256
    ):
        """Initializes the BYOLProjectionHead with the specified dimensions."""
        super(BYOLProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class BYOLPredictionHead(ProjectionHead):
    """Prediction head used for BYOL.

    "This MLP consists in a linear layer with output size 4096 followed by
    batch normalization, rectified linear units (ReLU), and a final
    linear layer with output dimension 256." [0]

    - [0]: BYOL, 2020, https://arxiv.org/abs/2006.07733
    """

    def __init__(
        self, input_dim: int = 256, hidden_dim: int = 4096, output_dim: int = 256
    ):
        super(BYOLPredictionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class MoCoProjectionHead(ProjectionHead):
    """Projection head used for MoCo.

    "(...) we replace the fc head in MoCo with a 2-layer MLP head (hidden layer
    2048-d, with ReLU)" [1]

    "The projection head is a 3-layer MLP. The prediction head is a 2-layer MLP. The
    hidden layers of both MLPs are 4096-d and are with ReLU; the output layers of both
    MLPs are 256-d, without ReLU. In MoCo v3, all layers in both MLPs have BN" [2]

    - [0]: MoCo v1, 2020, https://arxiv.org/abs/1911.05722
    - [1]: MoCo v2, 2020, https://arxiv.org/abs/2003.04297
    - [2]: MoCo v3, 2021, https://arxiv.org/abs/2104.02057
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 2,
        batch_norm: bool = False,
    ):
        """Initialize a new MoCoProjectionHead instance.

        Args:
            input_dim:
                Number of input dimensions.
            hidden_dim:
                Number of hidden dimensions (2048 for v2, 4096 for v3).
            output_dim:
                Number of output dimensions (128 for v2, 256 for v3).
            num_layers:
                Number of hidden layers (2 for v2, 3 for v3).
            batch_norm:
                Whether or not to use batch norms. (False for v2, True for v3).
        """
        layers: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]] = []
        layers.append(
            (
                input_dim,
                hidden_dim,
                nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                nn.ReLU(),
            )
        )
        for _ in range(2, num_layers):
            layers.append(
                (
                    hidden_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                    nn.ReLU(),
                )
            )
        layers.append(
            (
                hidden_dim,
                output_dim,
                nn.BatchNorm1d(output_dim) if batch_norm else None,
                None,
            )
        )
        super().__init__(layers)


class NNCLRProjectionHead(ProjectionHead):
    """Projection head used for NNCLR.

    "The architectureof the projection MLP is 3 fully connected layers of sizes
    [2048,2048,d] where d is the embedding size used to apply the loss. We use
    d = 256 in the experiments unless otherwise stated. All fully-connected
    layers are followed by batch-normalization [36]. All the batch-norm layers
    except the last layer are followed by ReLU activation." [0]

    - [0]: NNCLR, 2021, https://arxiv.org/abs/2104.14548
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 256
    ):
        """Initializes the NNCLRProjectionHead with the specified dimensions.

        Args:
            input_dim:
                Dimensionality of the input features.
            hidden_dim:
                Dimensionality of the hidden layers.
            output_dim:
                Dimensionality of the output features.
        """
        super(NNCLRProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, nn.BatchNorm1d(output_dim), None),
            ]
        )


class NNCLRPredictionHead(ProjectionHead):
    """Prediction head used for NNCLR.

    "The architecture of the prediction MLP g is 2 fully-connected layers
    of size [4096,d]. The hidden layer of the prediction MLP is followed by
    batch-norm and ReLU. The last layer has no batch-norm or activation." [0]

    - [0]: NNCLR, 2021, https://arxiv.org/abs/2104.14548
    """

    def __init__(
        self, input_dim: int = 256, hidden_dim: int = 4096, output_dim: int = 256
    ):
        super(NNCLRPredictionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class SimCLRProjectionHead(ProjectionHead):
    """Projection head used for SimCLR.

    "We use a MLP with one hidden layer to obtain zi = g(h) = W_2 * σ(W_1 * h)
    where σ is a ReLU non-linearity." [0]

    "We use a 3-layer MLP projection head on top of a ResNet encoder." [1]

    - [0] SimCLR v1, 2020, https://arxiv.org/abs/2002.05709
    - [1] SimCLR v2, 2020, https://arxiv.org/abs/2006.10029
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 2,
        batch_norm: bool = True,
    ):
        """Initialize a new SimCLRProjectionHead instance.

        Args:
            input_dim:
                Number of input dimensions.
            hidden_dim:
                Number of hidden dimensions.
            output_dim:
                Number of output dimensions.
            num_layers:
                Number of hidden layers (2 for v1, 3+ for v2).
            batch_norm:
                Whether or not to use batch norms.
        """
        layers: List[Tuple[int, int, Optional[nn.Module], Optional[nn.Module]]] = []
        layers.append(
            (
                input_dim,
                hidden_dim,
                nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                nn.ReLU(),
            )
        )
        for _ in range(2, num_layers):
            layers.append(
                (
                    hidden_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                    nn.ReLU(),
                )
            )
        layers.append(
            (
                hidden_dim,
                output_dim,
                nn.BatchNorm1d(output_dim) if batch_norm else None,
                None,
            )
        )
        super().__init__(layers)


class SimSiamProjectionHead(ProjectionHead):
    """Projection head used for SimSiam.

    "The projection MLP (in f) has BN applied to each fully-connected (fc)
    layer, including its output fc. Its output fc has no ReLU. The hidden fc is
    2048-d. This MLP has 3 layers." [0]

    - [0]: SimSiam, 2020, https://arxiv.org/abs/2011.10566
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 2048
    ):
        super(SimSiamProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (
                    hidden_dim,
                    output_dim,
                    nn.BatchNorm1d(output_dim, affine=False),
                    None,
                ),
            ]
        )


class SMoGPrototypes(nn.Module):
    """SMoG prototypes module for synchronous momentum grouping.

    Args:
        group_features:
            Tensor containing the group features.
        beta:
            Beta parameter for momentum updating.
    """

    def __init__(
        self,
        group_features: Tensor,
        beta: float,
    ):
        """Initializes the SMoGPrototypes module with the specified parameter."""
        super(SMoGPrototypes, self).__init__()
        self.group_features = nn.Parameter(group_features, requires_grad=False)
        self.beta = beta

    def forward(
        self, x: Tensor, group_features: Tensor, temperature: float = 0.1
    ) -> Tensor:
        """Computes the logits for given model outputs and group features.

        Args:
            x:
                Tensor of shape bsz x dim.
            group_features:
                Momentum updated group features of shape n_groups x dim.
            temperature:
                Temperature parameter for calculating the logits.

        Returns:
            The computed logits.
        """
        x = torch.nn.functional.normalize(x, dim=1)
        group_features = torch.nn.functional.normalize(group_features, dim=1)
        logits = torch.mm(x, group_features.t())
        return logits / temperature

    def get_updated_group_features(self, x: Tensor) -> Tensor:
        """Performs the synchronous momentum update of the group vectors.

        Args:
            x:
                Tensor of shape bsz x dim.

        Returns:
            The updated group features.
        """
        assignments = self.assign_groups(x)
        group_features = torch.clone(self.group_features.data)
        for assigned_class in torch.unique(assignments):
            mask = assignments == assigned_class
            group_features[assigned_class] = self.beta * self.group_features[
                assigned_class
            ] + (1 - self.beta) * x[mask].mean(dim=0)

        return group_features

    def set_group_features(self, x: Tensor) -> None:
        """Sets the group features and asserts they don't require gradient."""
        self.group_features.data = x.to(self.group_features.device)

    @torch.no_grad()
    def assign_groups(self, x: Tensor) -> Tensor:
        """Assigns each representation in x to a group based on cosine similarity.

        Args:
            x:
                Tensor of shape (bsz, dim).

        Returns:
            Tensor of shape (bsz,) indicating group assignments.
        """
        return torch.argmax(self.forward(x, self.group_features), dim=-1)


class SMoGProjectionHead(ProjectionHead):
    """Projection head used for SMoG.

    "The two kinds of head are both a two-layer MLP and their hidden layer is
    followed by a BatchNorm [28] and an activation function. (...) The output
    layer of projection head also has BN" [0]

    - [0]: SMoG, 2022, https://arxiv.org/pdf/2207.06167.pdf
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128
    ):
        """Initializes the SMoGProjectionHead with the specified dimensions.

        Args:
            input_dim:
                Dimensionality of the input features.
            hidden_dim:
                Dimensionality of the hidden layers.
            output_dim:
                Dimensionality of the output features.
        """
        super(SMoGProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (
                    hidden_dim,
                    output_dim,
                    nn.BatchNorm1d(output_dim, affine=False),
                    None,
                ),
            ]
        )


class SMoGPredictionHead(ProjectionHead):
    """Prediction head used for SMoG.

    "The two kinds of head are both a two-layer MLP and their hidden layer is
    followed by a BatchNorm [28] and an activation function. (...) The output
    layer of projection head also has BN" [0]

    - [0]: SMoG, 2022, https://arxiv.org/pdf/2207.06167.pdf
    """

    def __init__(
        self, input_dim: int = 128, hidden_dim: int = 2048, output_dim: int = 128
    ):
        """Initializes the SMoGPredictionHead with the specified dimensions.

        Args:
            input_dim:
                Dimensionality of the input features.
            hidden_dim:
                Dimensionality of the hidden layers.
            output_dim:
                Dimensionality of the output features.
        """

        super(SMoGPredictionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class SimSiamPredictionHead(ProjectionHead):
    """Prediction head used for SimSiam.

    "The prediction MLP (h) has BN applied to its hidden fc layers. Its output
    fc does not have BN (...) or ReLU. This MLP has 2 layers." [0]

    - [0]: SimSiam, 2020, https://arxiv.org/abs/2011.10566
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 512, output_dim: int = 2048
    ):
        """Initializes the SimSiamPredictionHead with the specified dimensions.

        Args:
            input_dim:
                Dimensionality of the input features.
            hidden_dim:
                Dimensionality of the hidden layers.
            output_dim:
                Dimensionality of the output features.
        """
        super(SimSiamPredictionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class SwaVProjectionHead(ProjectionHead):
    """Projection head used for SwaV.

    - [0]: SwAV, 2020, https://arxiv.org/abs/2006.09882
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128
    ):
        """Initializes the SwaVProjectionHead with the specified dimensions."""
        super(SwaVProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class SwaVPrototypes(nn.Module):
    """Multihead Prototypes used for SwaV.

    Each output feature is assigned to a prototype, SwaV solves the swapped
    prediction problem where the features of one augmentation are used to
    predict the assigned prototypes of the other augmentation.

    Attributes:
        input_dim:
            The input dimension of the head.
        n_prototypes:
            Number of prototypes.
        n_steps_frozen_prototypes:
            Number of steps during which we keep the prototypes fixed.

    Examples:
        >>> # use features with 128 dimensions and 512 prototypes
        >>> prototypes = SwaVPrototypes(128, 512)
        >>>
        >>> # pass batch through backbone and projection head.
        >>> features = model(x)
        >>> features = nn.functional.normalize(features, dim=1, p=2)
        >>>
        >>> # logits has shape bsz x 512
        >>> logits = prototypes(features)
    """

    def __init__(
        self,
        input_dim: int = 128,
        n_prototypes: Union[List[int], int] = 3000,
        n_steps_frozen_prototypes: int = 0,
    ):
        """Intializes the SwaVPrototypes module with the specified parameters"""
        super(SwaVPrototypes, self).__init__()

        # Default to a list of 1 if n_prototypes is an int.
        self.n_prototypes = (
            n_prototypes if isinstance(n_prototypes, list) else [n_prototypes]
        )
        self._is_single_prototype = True if isinstance(n_prototypes, int) else False
        self.heads = nn.ModuleList(
            [nn.Linear(input_dim, prototypes) for prototypes in self.n_prototypes]
        )
        self.n_steps_frozen_prototypes = n_steps_frozen_prototypes

    def forward(
        self, x: Tensor, step: Optional[int] = None
    ) -> Union[Tensor, List[Tensor]]:
        """Forward pass of the SwaVPrototypes module.

        Args:
            x:
                Input tensor.
            step:
                Current training step.

        Returns:
            The logits after passing through the prototype heads. Returns a single tensor
            if there's one prototype head, otherwise returns a list of tensors.
        """
        self._freeze_prototypes_if_required(step)
        out = []
        for layer in self.heads:
            out.append(layer(x))
        return out[0] if self._is_single_prototype else out

    def normalize(self) -> None:
        """Normalizes the prototypes so that they are on the unit sphere."""
        for layer in self.heads:
            utils.normalize_weight(layer.weight)

    def _freeze_prototypes_if_required(self, step: Optional[int] = None) -> None:
        """Freezes the prototypes if the specified number of steps has been reached."""
        if self.n_steps_frozen_prototypes > 0:
            if step is None:
                raise ValueError(
                    "`n_steps_frozen_prototypes` is greater than 0, please"
                    " provide the `step` argument to the `forward()` method."
                )
            self.requires_grad_(step >= self.n_steps_frozen_prototypes)


class DINOProjectionHead(ProjectionHead):
    """Projection head used in DINO.

    "The projection head consists of a 3-layer multi-layer perceptron (MLP)
    with hidden dimension 2048 followed by l2 normalization and a weight
    normalized fully connected layer with K dimensions, which is similar to the
    design from SwAV [1]." [0]

    - [0]: DINO, 2021, https://arxiv.org/abs/2104.14294
    - [1]: SwAV, 2020, https://arxiv.org/abs/2006.09882

    Attributes:
        input_dim:
            The input dimension of the head.
        hidden_dim:
            The hidden dimension.
        bottleneck_dim:
            Dimension of the bottleneck in the last layer of the head.
        output_dim:
            The output dimension of the head.
        batch_norm:
            Whether to use batch norm or not. Should be set to False when using
            a vision transformer backbone.
        freeze_last_layer:
            Number of epochs during which we keep the output layer fixed.
            Typically doing so during the first epoch helps training. Try
            increasing this value if the loss does not decrease.
        norm_last_layer:
            Whether or not to weight normalize the last layer of the DINO head.
            Not normalizing leads to better performance but can make the
            training unstable.
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        batch_norm: bool = False,
        freeze_last_layer: int = -1,
        norm_last_layer: bool = True,
    ):
        """Initializes the DINOProjectionHead with the specified dimensions."""
        super().__init__(
            [
                (
                    input_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                    nn.GELU(),
                ),
                (
                    hidden_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                    nn.GELU(),
                ),
                (hidden_dim, bottleneck_dim, None, None),
            ]
        )
        self.apply(self._init_weights)
        self.freeze_last_layer = freeze_last_layer
        self.last_layer = nn.Linear(bottleneck_dim, output_dim, bias=False)
        self.last_layer = nn.utils.weight_norm(self.last_layer)
        # Tell mypy this is ok because fill_ is overloaded.
        self.last_layer.weight_g.data.fill_(1)  # type: ignore

        # Option to normalize last layer.
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def cancel_last_layer_gradients(self, current_epoch: int) -> None:
        """Cancel last layer gradients to stabilize the training."""
        if current_epoch >= self.freeze_last_layer:
            return
        for param in self.last_layer.parameters():
            param.grad = None

    def forward(self, x: Tensor) -> Tensor:
        """Computes one forward pass through the head."""
        x = self.layers(x)
        # l2 normalization
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

    def _init_weights(self, module: nn.Module) -> None:
        """Initializes layers with a truncated normal distribution."""
        if isinstance(module, nn.Linear):
            utils._no_grad_trunc_normal(
                module.weight,
                mean=0,
                std=0.2,
                a=-2,
                b=2,
            )
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


class MMCRProjectionHead(ProjectionHead):
    """Projection head used for MMCR.

    "Following Chen et al. (14), we append a small perceptron to the output
    of the average pooling layer of the ResNet so that zi = g(h(xi)), where
    h is the ResNet and g is the MLP." [0]

    - [0]: MMCR, 2023, https://arxiv.org/abs/2303.03307
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 8192,
        output_dim: int = 512,
        num_layers: int = 2,
        batch_norm: bool = True,
        use_bias: bool = False,
    ):
        """Initialize a new MMCRProjectionHead instance.

        Args:
            input_dim:
                Number of input dimensions.
            hidden_dim:
                Number of hidden dimensions.
            output_dim:
                Number of output dimensions.
            num_layers:
                Number of hidden layers.
            batch_norm:
                Whether or not to use batch norms.
            use_bias:
                Whether or not to use bias in the linear layers.
        """
        layers: List[
            Tuple[int, int, Optional[nn.Module], Optional[nn.Module], bool]
        ] = []

        # Add the first layer
        layers.append(
            (
                input_dim,
                hidden_dim,
                nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                nn.ReLU(),
                use_bias,
            )
        )

        # Add the hidden layers
        for _ in range(num_layers - 1):
            layers.append(
                (
                    hidden_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                    nn.ReLU(),
                    use_bias,
                )
            )

        # Add the output layer
        layers.append((hidden_dim, output_dim, None, None, use_bias))
        super().__init__(layers)


class MSNProjectionHead(ProjectionHead):
    """Projection head for MSN [0].

    "We train with a 3-layer projection head with output dimension 256 and
    batch-normalization at the input and hidden layers.." [0]

    Code inspired by [1].

    - [0]: Masked Siamese Networks, 2022, https://arxiv.org/abs/2204.07141
    - [1]: https://github.com/facebookresearch/msn

    Attributes:
        input_dim:
            Input dimension, default value 768 is for a ViT base model.
        hidden_dim:
            Hidden dimension.
        output_dim:
            Output dimension.
    """

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 2048,
        output_dim: int = 256,
    ):
        """Initializes the MSNProjectionHead with the specified dimensions."""
        super().__init__(
            blocks=[
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.GELU()),
                (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.GELU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class TiCoProjectionHead(ProjectionHead):
    """Projection head used for TiCo.

    "This MLP consists in a linear layer with output size 4096 followed by
    batch normalization, rectified linear units (ReLU), and a final
    linear layer with output dimension 256." [0]

    - [0]: TiCo, 2022, https://arxiv.org/pdf/2206.10698.pdf
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 4096, output_dim: int = 256
    ):
        """Initializes the TiCoProjectionHead with the specified dimensions."""
        super(TiCoProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class VICRegProjectionHead(ProjectionHead):
    """Projection head used for VICReg.

    "The projector network has three linear layers, each with 8192 output
    units. The first two layers of the projector are followed by a batch
    normalization layer and rectified linear units." [0]

    - [0]: 2022, VICReg, https://arxiv.org/pdf/2105.04906.pdf
    """

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 8192,
        output_dim: int = 8192,
        num_layers: int = 3,
    ):
        """Initializes the VICRegProjectionHead with the specified dimensions.

        Args:
            input_dim:
                Dimensionality of the input features.
            hidden_dim:
                Dimensionality of the hidden layers.
            output_dim:
                Dimensionality of the output features.
            num_layers:
                Number of layers in the projection head.
        """
        hidden_layers = [
            (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU())
            for _ in range(num_layers - 2)  # Exclude first and last layer.
        ]
        super(VICRegProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                *hidden_layers,
                (hidden_dim, output_dim, None, None),
            ]
        )


class VicRegLLocalProjectionHead(ProjectionHead):
    """Projection head used for the local head of VICRegL.

    "The projector network has three linear layers. The first two layers of the projector
    are followed by a batch normalization layer and rectified linear units." [0]

    - [0]: 2022, VICRegL, https://arxiv.org/abs/2210.01571
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 8192, output_dim: int = 8192
    ):
        """Initializes the VicRegLLocalProjectionHead with the specified dimensions."""
        super(VicRegLLocalProjectionHead, self).__init__(
            [
                (input_dim, hidden_dim, nn.LayerNorm(hidden_dim), nn.ReLU()),
                (hidden_dim, hidden_dim, nn.LayerNorm(hidden_dim), nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )


class DenseCLProjectionHead(ProjectionHead):
    """Projection head for DenseCL [0].

    The projection head consists of a 2-layer MLP. It can be used for global and local
    features.

    - [0]: 2021, DenseCL: https://arxiv.org/abs/2011.09157
    """

    def __init__(
        self, input_dim: int = 2048, hidden_dim: int = 2048, output_dim: int = 128
    ):
        """Initializes the DenseCLProjectionHead with the specified dimensions."""
        super().__init__(
            [
                (input_dim, hidden_dim, None, nn.ReLU()),
                (hidden_dim, output_dim, None, None),
            ]
        )
