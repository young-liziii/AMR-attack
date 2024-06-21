import numpy as np

import torch
from torch.nn.functional import cross_entropy

from typing import Tuple, Union

# Internal Includes
import rfml.nn.F as F
from rfml.nn.model import Model
from rfml.ptradio import Slicer

from .utils import _convert_or_throw, _infer_input_size, _dither, _normalize
from .utils import _compute_multiplier


def fgsm(
    x: torch.Tensor,
    y: Union[torch.Tensor, int],
    net: Model,
    spr: float,
    input_size: int = None,
    sps: int = 8,
) -> torch.Tensor:

    x, y = _convert_or_throw(x=x, y=y)
    input_size = _infer_input_size(x=x, input_size=input_size)
    p = compute_signed_gradient(x=x, y=y, input_size=input_size, sps=sps, net=net)
    p = scale_perturbation(sg=p, spr=spr, sps=sps)

    return x + p


def compute_signed_gradient(
    x: torch.Tensor,
    y: Union[torch.Tensor, int],
    net: Model,
    input_size: int = None,
    sps: int = 8,
) -> torch.Tensor:

    x, y = _convert_or_throw(x=x, y=y)
    input_size = _infer_input_size(x=x, input_size=input_size)
    slicer = Slicer(width=input_size)

    # Ensure that the gradient is tracked at the input, add some noise to avoid any
    # actual zeros in the signal (dithering), and then ensure its the proper shape
    x.requires_grad = True
    _x = _normalize(x=x, sps=sps)
    _x = _dither(_x)
    _x = slicer(_x)

    # Ensure the model is in eval mode so that batch norm/dropout etc. doesn't take
    # effect -- in order to be transparent to the caller we need to restore the state
    # at the end.
    set_training = net.training
    if set_training:
        net.eval()

    # Perform forward/backward pass to get the gradient at the input
    _y = net(_x)
    loss = cross_entropy(_y, y)
    loss.backward()
    # loss_value = loss.item()  # 获取损失值作为标量

    ret = torch.sign(x.grad.data)

    # Restore the network state so the caller never notices the change
    if set_training:
        net.train()

    return ret


def scale_perturbation(sg: torch.Tensor, spr: float, sps: int = 8) -> torch.Tensor:
    if spr == np.inf:
        return sg * 0
    # multiplier = _compute_multiplier(spr=spr, sps=sps)
    multiplier = 1/512
    return sg * multiplier



