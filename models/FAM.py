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


class FAM(torch.nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, size,size1,size2,size3,**kwargs):
        super(FAM, self).__init__()
        self.size = size
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3

        self.out = torch.nn.Sequential(
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,padding=1,bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU())

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU()
        )
        self.pool1 = nn.AdaptiveAvgPool2d(size1)
        self.pool2 = nn.AdaptiveAvgPool2d(size2)
        self.pool3 = nn.AdaptiveAvgPool2d(size3)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        x_1 = self.upsample(self.conv1(self.pool1(x)), self.size)
        x_2 = self.upsample(self.conv2(self.pool2(x)), self.size)
        x_3 = self.upsample(self.conv3(self.pool3(x)), self.size)
        x = x_1 + x_2 + x_3
        x = self.out(x)
        return x
