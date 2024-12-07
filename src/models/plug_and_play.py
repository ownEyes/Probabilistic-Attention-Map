import functools
from typing import Callable, Optional, List
import torch.nn as nn
from torch import Tensor

from .modules import BasicBlock, Bottleneck, InvertedResidual, Conv2dNormActivation
from .modules import get_attention_module
from .backbones import model_dict

resnet_arch_block_dict = {
    "resnet18": BasicBlock,
    "resnet34": BasicBlock,
    "resnet50": Bottleneck,
    "resnet101": Bottleneck,
    "resnet152": Bottleneck,

    "resnet20": BasicBlock,
    "resnet32": BasicBlock,
    "resnet44": BasicBlock,
    "resnet56": BasicBlock,
    "resnet110": BasicBlock,
    "resnet1202": BasicBlock,

    "mobilenet_v2": InvertedResidual,
}

resnet_attention_basicblock_dict = {
    "cbam": 'bn2',
    "se": 'bn2',
    "simam": 'conv2',
    "pdfam_gmm": 'conv2',
    "pdfam_gau": 'conv2',

    # "lap_spatial_only": 'after_add',
    # "lap_spatial_param": 'after_add',
    "lap_spatial_only": 'after_relu',
    "lap_spatial_param": 'after_relu',

    # ablation on resnet-20
    "lap_ablation_none": 'after_relu',
    "lap_ablation_no_adjust": 'after_relu',
    "lap_ablation_no_scale": 'after_relu',
    "lap_ablation_no_width": 'after_relu',
    "lap_ablation_only_adjust": 'after_relu',
    "lap_ablation_only_scale": 'after_relu',
    "lap_ablation_only_width": 'after_relu',
}

resnet_attention_bottlenect_dict = {
    "cbam": 'bn3',
    "se": 'bn3',
    "simam": 'conv2',
    "pdfam_gmm": 'conv2',
    "pdfam_gau": 'conv2',

    "lap_spatial_only": 'after_add',
    "lap_spatial_param": 'after_add',
}

mobilenetv2_attention_invertedresidual_dict = {
    "cbam": 'bn3',
    "se": 'bn3',
    "simam": 'conv2',
    "pdfam_gmm": 'conv2',
    "pdfam_gau": 'conv2',

    "lap_spatial_only": 'after_relu',
    "lap_spatial_param": 'after_relu',
}


def create_net(args):
    net = None

    attention_module = get_attention_module(args.attention_type)

    # srm does not have any input parameters
    if args.attention_type == "se" or args.attention_type == "cbam":
        attention_module = functools.partial(
            attention_module, reduction=args.attention_param)
    elif args.attention_type == "simam":
        attention_module = functools.partial(
            attention_module, e_lambda=args.attention_param)
    # elif args.attention_type == "pdfam_gau":
    #     attention_module = functools.partial(attention_module)
    elif args.attention_type == "pdfam_gmm":
        attention_module = functools.partial(
            attention_module, num_T=args.attention_param, num_K=args.attention_param2)

    insertion_point = get_insertion_point(args)

    BlockAttention = plug_attention(args, attention_module, insertion_point)

    kwargs = {}
    kwargs["num_classes"] = args.num_class

    net = model_dict[args.arch.lower()](BlockAttention, **kwargs)

    return net


def plug_attention(args, attention_module_class=None, insertion_point=None):
    # Define valid insertion points
    valid_insertion_points = ['conv2', 'bn2', 'bn3', 'after_add', 'after_relu']

    # Validate the insertion_point
    if insertion_point != None and insertion_point not in valid_insertion_points:
        raise ValueError(
            f"Invalid insertion_point '{insertion_point}'. Valid options are: {valid_insertion_points}")

    block_class = resnet_arch_block_dict[args.arch.lower()]

    if attention_module_class != None:
        if block_class != InvertedResidual:
            class BlockAttention(block_class):
                def __init__(self,
                             inplanes: int,
                             planes: int,
                             stride: int = 1,
                             downsample: Optional[nn.Module] = None,
                             groups: int = 1,
                             base_width: int = 64,
                             dilation: int = 1,
                             norm_layer: Optional[Callable[..., nn.Module]] = None,):
                    super().__init__(inplanes, planes, stride, downsample,
                                     groups, base_width, dilation, norm_layer)
                    if type(attention_module_class) == functools.partial:
                        module_name = attention_module_class.func.get_module_name()
                    else:
                        module_name = attention_module_class.get_module_name()
                    # Initialize the attention module, if applicable
                    if module_name == 'lap_spatial_only':
                        self.attention = attention_module_class()

                    else:
                        self.attention = attention_module_class(
                            planes * self.expansion)

                def forward(self, x):
                    identity = x

                    out = self.conv1(x)
                    out = self.bn1(out)
                    out = self.relu(out)

                    if insertion_point == 'after_relu':
                        out = self.attention(out)

                    out = self.conv2(out)

                    if insertion_point == 'conv2':
                        out = self.attention(out)

                    out = self.bn2(out)

                    if insertion_point == 'bn2':
                        out = self.attention(out)

                # Additional layers in Bottleneck
                    if hasattr(self, 'conv3'):
                        out = self.relu(out)
                        out = self.conv3(out)

                        out = self.bn3(out)
                        if insertion_point == 'bn3':
                            out = self.attention(out)

                    if self.downsample is not None:
                        identity = self.downsample(x)

                    out += identity
                    out = self.relu(out)

                    if insertion_point == 'after_add':
                        out = self.attention(out)

                    return out

            return BlockAttention

        else:
            class InvertedBlockAttention(block_class):
                def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int, norm_layer: Optional[Callable[..., nn.Module]] = None):
                    super().__init__(inp, oup, stride, expand_ratio, norm_layer)

                    hidden_dim = int(round(inp * expand_ratio))
                    if stride not in [1, 2]:
                        raise ValueError(
                            f"stride should be 1 or 2 instead of {stride}")

                    if norm_layer is None:
                        norm_layer = nn.BatchNorm2d

                    if type(attention_module_class) == functools.partial:
                        module_name = attention_module_class.func.get_module_name()
                    else:
                        module_name = attention_module_class.get_module_name()
                    # Initialize the attention module, if applicable
                    if module_name == 'lap_spatial_only':
                        self.attention = attention_module_class()
                    elif insertion_point == 'bn3':
                        self.attention = attention_module_class(oup)
                    else:
                        self.attention = attention_module_class(hidden_dim)

                    layers: List[nn.Module] = []
                    if expand_ratio != 1:
                        # pw
                        layers.append(
                            Conv2dNormActivation(
                                inp, hidden_dim, kernel_size=1, norm_layer=norm_layer, activation_layer=nn.ReLU6)
                        )
                    layers.extend(
                        ([
                            # dw
                            Conv2dNormActivation(
                                hidden_dim,
                                hidden_dim,
                                stride=stride,
                                groups=hidden_dim,
                                norm_layer=norm_layer,
                                activation_layer=nn.ReLU6,
                                attention_layer=self.attention,
                            ),
                        ] if insertion_point == 'conv2' else
                            [
                            # dw
                            Conv2dNormActivation(
                                hidden_dim,
                                hidden_dim,
                                stride=stride,
                                groups=hidden_dim,
                                norm_layer=norm_layer,
                                activation_layer=nn.ReLU6,
                            ),
                        ]) +
                        ([self.attention] if insertion_point == 'after_relu' else []) +
                        [
                            # pw-linear
                            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                            norm_layer(oup),
                        ]
                        # + ([self.attention] if insertion_point == 'bn3' else [])
                    )

                    self.conv = nn.Sequential(*layers)

                def forward(self, x: Tensor) -> Tensor:
                    out = self.conv(x)
                    if insertion_point == 'bn3':
                        out = self.attention(out)

                    if self.use_res_connect:
                        return x + out
                    else:
                        return out

            return InvertedBlockAttention

    else:
        return block_class


def get_insertion_point(args):
    if args.attention_type != 'none':
        block_class = resnet_arch_block_dict[args.arch.lower()]
        if block_class == BasicBlock:
            insertion_point = resnet_attention_basicblock_dict[args.attention_type.lower(
            )]
        elif block_class == Bottleneck:
            insertion_point = resnet_attention_bottlenect_dict[args.attention_type.lower(
            )]
        elif block_class == InvertedResidual:
            insertion_point = mobilenetv2_attention_invertedresidual_dict[args.attention_type.lower(
            )]
        return insertion_point
    else:
        return None
