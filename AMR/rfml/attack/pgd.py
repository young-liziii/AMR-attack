import numpy as np

import torch
from torch.autograd import Variable
from torch.nn.functional import cross_entropy

from typing import Union

from rfml.nn.model import Model
from rfml.ptradio import Slicer

from .utils import _convert_or_throw, _infer_input_size, _normalize
from .utils import _random_uniform_start, _compute_multiplier


def pgd(
    x: torch.Tensor,
    y: Union[torch.Tensor, int],
    net: Model,
    k: int,
    input_size: int = None,
) -> torch.Tensor:
    # x (torch.Tensor): 输入信号的连续表示，形状为 (BxCxIQxN)，其中 B 是批处理大小，C 是通道数，IQ 是信号维度，N 是样本数量。
    # y (Union[torch.Tensor, int]): 输入信号的分类标签。可以是单个整数，表示所有输入的标签，或者是一个长度为 B 的张量，表示每个输入的标签。
    # 对输入的信号 x 和标签 y 进行类型检查和转换
    x, y = _convert_or_throw(x=x, y=y)
    # 从输入信号x中推断样本数量或验证用户提供input_size是否与x相匹配
    input_size = _infer_input_size(x=x, input_size=input_size)
    x = Slicer(width=input_size)(x)
    _x = x

    eps = 1/128
    # 步长取值参考了 Madry 等人的论文，他们提出了 step_size = 2/255 * eps 作为在 MNIST 数据集上的一个有效选择
    step = eps / float(64)

    set_training = net.training
    if set_training:
        net.eval()

    for _k in range(k):
        _x.requires_grad = True
        _y = net(_x)
        loss = cross_entropy(_y, y)
        loss.backward()
        # 计算梯度符号
        ret = torch.sign(_x.grad.data)
        _x = _x + step * ret

        # 保证不越界
        compare_result1 = _x < (x - eps)
        _x = torch.where(compare_result1, x-eps, _x)
        compare_result2 = _x > (x + eps)
        _x = torch.where(compare_result2, x+eps, _x)

        _x = Variable(_x)

    if set_training:
        net.train()

    return _x
