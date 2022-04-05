import sys
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models
from .PyramidPooling import PyramidPooling
from .FAM import FAM
from .feat_agg import FeatAgg

from .basic import weight_init


class LFGAAVGG19(torch.nn.Module):
    def __init__(self, k):
        super(LFGAAVGG19, self).__init__()
        self.k = k
        vgg19 = models.vgg19_bn(pretrained=True)

        features_list = list(vgg19.features.children())
        self.conv2_2 = torch.nn.Sequential(*features_list[:13])      # 1 x 128 x 112x112
        self.conv3_4 = torch.nn.Sequential(*features_list[13:26])     # 1 x 256 x 56 x 56
        self.conv4_4 = torch.nn.Sequential(*features_list[26: 39])  # 1 x 512 x 28 x 28
        self.conv5_4 = torch.nn.Sequential(*features_list[39:-1])   # 1 x 512 x 14 x 14

        self.conv_6 = torch.nn.Sequential(
            torch.nn.Conv2d(512, self.k, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.k),
            torch.nn.ReLU()
        )

        self.conv_7 = torch.nn.Sequential(
            torch.nn.Conv2d(512, self.k, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.k),
            torch.nn.ReLU()
        )

        self.gap_layer_1 = torch.nn.AdaptiveAvgPool2d(1)
        self.gap_layer_2 = torch.nn.AdaptiveAvgPool2d(1)

        self.bn1 = torch.nn.BatchNorm1d(k)
        self.bn2 = torch.nn.BatchNorm1d(k)

        self.ppm = PyramidPooling(k, 512)

        self.ppm_out1 = torch.nn.Sequential(
            torch.nn.Conv2d(k, 512, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU())

        self.ppm_out2 = torch.nn.Sequential(
            torch.nn.Conv2d(k, 256, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU())

        self.ppm_out3 = torch.nn.Sequential(
            torch.nn.Conv2d(k, 128, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU())

        self.fam1 = FAM(512, 512, [28,28], [14,14], [8,8], [4,4])
        self.feat_agg1 = FeatAgg(512, 512)

        self.fam2 = FAM(512, 256, [56, 56], [28, 28], [14, 14], [8, 8])
        self.feat_agg2 = FeatAgg(256, 256)

        self.fam3 = FAM(256, 128, [112, 112], [56, 56], [28, 28], [14, 14])
        self.feat_agg3 = FeatAgg(128, 128)

        self.fam4 = FAM(128, 128, [112, 112], [56, 56], [28, 28], [14, 14])
        self.feat_agg4 = FeatAgg(128, 128)

        self.fam5 = FAM(128, 256, [56, 56], [28, 28], [14, 14], [8, 8])
        self.feat_agg5 = FeatAgg(256, 256)

        self.fam6 = FAM(256, 512, [28, 28], [14, 14], [8, 8], [4, 4])
        self.feat_agg6 = FeatAgg(512, 512)

        self.gap_layer_p0 = torch.nn.AdaptiveAvgPool2d(1)
        self.gap_layer_p1 = torch.nn.AdaptiveAvgPool2d(1)
        self.gap_layer_p2 = torch.nn.AdaptiveAvgPool2d(1)

        self.conv_p0 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=8, stride=8),
            torch.nn.Conv2d(128, self.k, kernel_size=1, stride=1)
        )
        self.conv_p1 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=4, stride=4),
            torch.nn.Conv2d(256, self.k, kernel_size=1, stride=1)
        )
        self.conv_p2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(512, self.k, kernel_size=1, stride=1)
        )

    def upsample(self, x, size):  # 上采样使用双线性插值
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        conv2_2 = self.conv2_2(x)
        conv3_4 = self.conv3_4(conv2_2)
        conv4_4 = self.conv4_4(conv3_4)
        conv5_4 = self.conv5_4(conv4_4)

        conv_6 = self.conv_6(conv5_4)
        conv_7 = self.conv_7(conv5_4)

        gap = F.relu(torch.squeeze(self.gap_layer_1(conv_6)))

        gap1 = F.relu(torch.squeeze(self.gap_layer_2(conv_7)))

        attr = self.bn1(gap)

        latent = self.bn2(gap1)

        ppm_feat = conv_7
        ppm_feat = self.ppm(ppm_feat)

        ppm_feat1 = self.ppm_out1(self.upsample(ppm_feat,[28,28]))
        ppm_feat2 = self.ppm_out2(self.upsample(ppm_feat, [56, 56]))
        ppm_feat3 = self.ppm_out3(self.upsample(ppm_feat, [112, 112]))

        fam1 = self.fam1(conv5_4)
        feat_agg1 = self.feat_agg1(conv4_4, ppm_feat1, fam1)

        fam2 = self.fam2(feat_agg1)
        feat_agg2 = self.feat_agg2(conv3_4, ppm_feat2, fam2)

        fam3 = self.fam3(feat_agg2)
        feat_agg3 = self.feat_agg3(conv2_2, ppm_feat3, fam3)

        fam4 = self.fam4(feat_agg3)
        feat_agg4 = self.feat_agg4(conv2_2, ppm_feat3, fam4)

        fam5 = self.fam5(feat_agg4)
        feat_agg5 = self.feat_agg5(conv3_4, ppm_feat2, fam5)

        fam6 = self.fam6(feat_agg5)
        feat_agg6 = self.feat_agg6(conv4_4, ppm_feat1, fam6)

        feat2 = self.conv_p2(feat_agg6)
        feat1 = self.conv_p1(feat_agg5)
        feat0 = self.conv_p0(feat_agg4)

        p_0_feat = torch.squeeze(self.gap_layer_p0(feat0))
        p_1_feat = torch.squeeze(self.gap_layer_p1(feat1))
        p_2_feat = torch.squeeze(self.gap_layer_p2(feat2))

        p_0 = F.softmax(p_0_feat, dim=1)
        p_1 = F.softmax(p_1_feat, dim=1)
        p_2 = F.softmax(p_2_feat, dim=1)
        f_0 = torch.sigmoid(p_0_feat)
        f_1 = torch.sigmoid(p_1_feat)
        f_2 = torch.sigmoid(p_2_feat)

        p = p_0 + p_1 + p_2
        f = f_0 + f_1 + f_2

        return attr * p, latent, conv_6[:, 0: self.k, :, :], attr, f, p



