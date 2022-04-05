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

class PyramidPooling(torch.nn.Module):
    """Pyramid pooling module"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)  # 这里N=4与原文一致
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1,bias=False),
            torch.nn.BatchNorm2d(inter_channels),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1,bias=False),
            torch.nn.BatchNorm2d(inter_channels),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1,bias=False),
            torch.nn.BatchNorm2d(inter_channels),
            torch.nn.ReLU()
        )
        if in_channels% 2 != 0:
            self.conv4 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, inter_channels + 1, kernel_size=1, stride=1,bias=False),
                torch.nn.BatchNorm2d(inter_channels+1),
                torch.nn.ReLU()
            )

        else:

            self.conv4 = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1,bias=False),
                torch.nn.BatchNorm2d(inter_channels),
                torch.nn.ReLU()
            )

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)  # 自适应的平均池化，目标size分别为1x1,2x2,3x3,6x6
        return avgpool(x)

    def upsample(self, x, size):  # 上采样使用双线性插值
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)

        x = torch.cat([feat1, feat2, feat3, feat4], dim=1)  # concat 四个池化的结果

        return x

