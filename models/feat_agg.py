#-*-coding:utf-8-*-
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models

from .basic import weight_init


class FeatAgg(torch.nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels,**kwargs):
        super(FeatAgg, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )

    def forward(self, x_ori, x_p, x_a):
        x = x_ori + x_p + x_a
        x = self.conv1(x)
        return x
