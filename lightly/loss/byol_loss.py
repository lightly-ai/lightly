import singleton as singleton
import torch
from torch import nn

import torch.nn.functional as F

from models.mlp import MLP


class ExperimentParams:
    """
    Encapsulate hyper-parameters for performing and logging experiments.
    """

    def __init(self, use_momentum, beta, augment_functions, net, projection_size, projection_hidden_size, layer):
        # TODO: implement momentum
        self.use_momentum = use_momentum
        self.beta = beta
        self.augment_functions = augment_functions
        self.net = net
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size
        self.layer = layer

    def __str__(self):
        return "Running experiment with the following hyper-parameters.\n" \
               "Momentum: {}\nBeta: {}\nAugment Function: {}\nNet: {}\nProjection Size: {}\n" \
               "Projection Hidden Size: {}\n" "Layer: {}".format(self.use_momentum, self.beta, self.augment_functions,
                                                                 self.net, self.projection_size,
                                                                 self.projection_hidden_size, self.layer)


class NetWrapper(nn.Module):
    """
    Utilities for accessing hidden layers.
    """
    def __init__(self, net, projection_size, projection_hidden_size, layer=-2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.projector = None
        self.projection_size = projection_size
        self.projection_hidden_size = projection_hidden_size

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = self.flatten(output)

    def _register_hook(self):
        layer = self._find_layer()
        assert layer, f'hidden layer ({self.layer}) not found'
        layer.register_forward_hook(self._hook)
        self.hook_registered = True

    @singleton('projector')
    def get_projector(self, hidden):
        _, dim = hidden.shape
        projector = MLP(dim, self.projection_size, self.projection_hidden_size)
        return projector.to(hidden)

    def get_representation(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

    @staticmethod
    def default(val, def_val):
        return val if val else def_val

    @staticmethod
    def flatten(t):
        return t.reshape(t.shape[0], -1)


class OnlineModel:
    """
    Online student model, encapsulates an encoder and a predictor.
    """

    def __init__(self, net, projection_size, projection_hidden_size, hidden_layer):
        self.encoder = NetWrapper(net, projection_size, projection_hidden_size, layer=hidden_layer)
        self.predictor = MLP(projection_size, projection_size, projection_hidden_size)

    def make_prediction(self, image):
        return self.predictor(self.encoder(image))


class TargetModel:
    """
    Mean Teacher Model that takes an exponential moving average weighed by beta of past observed weights.
    """

    def __init__(self, beta, encoder, ema):
        self.beta = beta
        self.encoder = encoder
        self.ema = ema

    def update_with_online_encoder(self, online_encoder):
        for current_params, ma_params in zip(online_encoder.parameters(), self.encoder.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self._update_average(old_weight, up_weight)

    def _update_average(self, old, new):
        if self.ema:
            return old * self.beta + (1 - self.beta) * new
        else:
            return new


class BYOLLoss:
    """
    Calculates consistency loss over MSE loss of the student model and the latent representation of the
    mean teacher (target model).
    """

    def __init__(self, params):
        super().__init__()

        self.decay = params.decay

        self.augment_1, self.augment_2 = params.augment_functions

        self.use_momentum = params.use_momentum

        self.online = OnlineModel(params.net, params.projection_size, params.projection_hidden_size, params.layer)
        self.target = TargetModel(params.beta, self.online.encoder)

    def forward(self, x):
        """

        Args:
            x: image
        1. use two different augmentation function on same image
        2. pass image through online encoder and predictor
        3. extract projection from 'mean teacher' network
        4. cross-wise consistency loss, sum and mean

        Returns: the average consistency loss between student and
        teacher networks for two perturbations of the same image

        """
        image_one, image_two = self.augment_1(x), self.augment_2(x)

        online_pred_one = self.online.make_prediction(image_one)
        online_pred_two = self.online.make_prediction(image_two)

        with torch.no_grad():
            target_encoder = self.target.encoder
            target_proj_one = target_encoder(image_one).detach()
            target_proj_two = target_encoder(image_two).detach()

        loss_one = self._loss_fn(online_pred_one, target_proj_two.detach())
        loss_two = self._loss_fn(online_pred_two, target_proj_one.detach())

        loss = loss_one + loss_two
        return loss.mean()

    @staticmethod
    def _loss_fn(x, y):
        """
        Args:
            x: loss from online prediction and target projector (augmentation one and two respectively)
            y: loss from online prediction and target projector (augmentation two and one respectively)

        Returns: consistency loss for the discriminative task
        """
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)
