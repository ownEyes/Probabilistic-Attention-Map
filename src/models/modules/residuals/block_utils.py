import torch.nn as nn
from abc import ABC, abstractmethod

# nn.Conv2d default values:

# self,
# in_channels: int,
# out_channels: int,
# kernel_size: _size_2_t,
# stride: _size_2_t = 1,
# padding: Union[str, _size_2_t] = 0,
# dilation: _size_2_t = 1,
# groups: int = 1,
# bias: bool = True,
# padding_mode: str = 'zeros',  # TODO: refine this type
# device=None,
# dtype=None


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Block(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass
