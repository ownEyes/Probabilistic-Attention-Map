from .resnet_cifar import *
from .resnet import *
from .mobilenetv2 import mobilenet_v2_cifar

model_dict = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,

    "resnet20": resnet20_cifar,
    "resnet32": resnet32_cifar,
    "resnet44": resnet44_cifar,
    "resnet56": resnet56_cifar,
    "resnet110": resnet110_cifar,
    "resnet1202": resnet1202_cifar,

    "mobilenet_v2": mobilenet_v2_cifar,
}
