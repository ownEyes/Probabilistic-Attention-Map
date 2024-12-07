from torch import nn
import torch.nn.functional as F


class se_module(nn.Module):
    def __init__(self, channel, reduction=16):
        super(se_module, self).__init__()
        # NOTE: change to nn.functional.adaptive_avg_pool2d without the need for creating a layer object,
        # which can help avoid the issue with thop hooks.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(channel // reduction), channel, bias=False),
            nn.Sigmoid()
        )

    @staticmethod
    def get_module_name():
        return "se"

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        # y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
