'''
The implementation of ResNets on CIFAR10/100 as described in the original paper[1].

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
[3] https://microsoft.github.io/archai/_modules/archai/supergraph/models/resnet_paper.html#ResNet
'''
import torch.nn as nn
import torch
from torch import Tensor
from typing import Any, Callable, List, Optional, Type, Union
import torch.nn.functional as F

from ..modules.residuals import Block, BasicBlock, Bottleneck, conv3x3

__all__ = [
    'ResNet_CIFAR',
    'resnet20_cifar',
    'resnet32_cifar',
    'resnet44_cifar',
    'resnet56_cifar',
    'resnet110_cifar',
    'resnet1202_cifar'
]


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResNet_CIFAR(nn.Module):
    def __init__(
        self,
        block: Type[Block],
        layers: List[int],
        num_classes: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        # _log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16    # different from 64 in torchvision
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = conv3x3(3, self.inplanes)  # different from torchvision
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # no maxpooling

        # number of planes are different from torchvision
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(
            block, 32, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(
            block, 64, layers[2], stride=2, dilate=replace_stride_with_dilation[1])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # if zero_init_residual:
        #     for m in self.modules():
        #         if isinstance(m, Bottleneck) and m.bn3.weight is not None:
        #             # type: ignore[arg-type]
        #             nn.init.constant_(m.bn3.weight, 0)
        #         elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
        #             # type: ignore[arg-type]
        #             nn.init.constant_(m.bn2.weight, 0)
        if zero_init_residual:
            for m in self.modules():
                # Assuming convention where the last BN layer in any residual block ends with 'bn3' or 'bn2'
                if isinstance(m, Block):
                    last_bn = getattr(m, 'bn3', None) or getattr(
                        m, 'bn2', None)
                    if last_bn is not None and hasattr(last_bn, 'weight'):
                        nn.init.constant_(last_bn.weight, 0)

    def _make_layer(
        self,
        block: Type[Block],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # downsample = nn.Sequential(
            #     conv1x1(self.inplanes, planes * block.expansion, stride),
            #     norm_layer(planes * block.expansion),
            # )

            # option A downsample
            downsample = LambdaLayer(lambda x:
                                     F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # no maxpooling
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet_cifar(
    block: Type[Block],
    layers: List[int],
    **kwargs: Any,
) -> ResNet_CIFAR:

    model = ResNet_CIFAR(block, layers, **kwargs)

    return model


def resnet20_cifar(block_class: Type[Block] = BasicBlock, **kwargs: Any) -> ResNet_CIFAR:
    return _resnet_cifar(block_class, [3, 3, 3], **kwargs)


def resnet32_cifar(block_class: Type[Block] = BasicBlock, **kwargs: Any) -> ResNet_CIFAR:
    return _resnet_cifar(block_class, [5, 5, 5], **kwargs)


def resnet44_cifar(block_class: Type[Block] = BasicBlock, **kwargs: Any) -> ResNet_CIFAR:
    return _resnet_cifar(block_class, [7, 7, 7], **kwargs)


def resnet56_cifar(block_class: Type[Block] = BasicBlock, **kwargs: Any) -> ResNet_CIFAR:
    return _resnet_cifar(block_class, [9, 9, 9], **kwargs)


def resnet110_cifar(block_class: Type[Block] = BasicBlock, **kwargs: Any) -> ResNet_CIFAR:
    return _resnet_cifar(block_class, [18, 18, 18], **kwargs)


def resnet1202_cifar(block_class: Type[Block] = BasicBlock, **kwargs: Any) -> ResNet_CIFAR:
    return _resnet_cifar(block_class, [200, 200, 200], **kwargs)


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(
        lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


# cd src/
# python -m models.backbones.resnet_cifar
if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()

'''
output:

resnet20_cifar
Total number of params 269722
Total layers 20

resnet32_cifar
Total number of params 464154
Total layers 32

resnet44_cifar
Total number of params 658586
Total layers 44

resnet56_cifar
Total number of params 853018
Total layers 56

resnet110_cifar
Total number of params 1727962
Total layers 110

resnet1202_cifar
Total number of params 19421274
Total layers 1202


from original paper[1]:
name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4M
'''
