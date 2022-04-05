import sys
import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models as th_models
from .PyramidPooling import PyramidPooling
from .FAM import FAM
from .feat_agg import FeatAgg
from .basic import weight_init

model_urls = {
    'inception_v3_google': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
}

class LFGAAGoogleNet(torch.nn.Module):
    def __init__(self, k):
        super(LFGAAGoogleNet, self).__init__()
        self.k = k
    
        inception_v3 = th_models.Inception3(aux_logits=False, transform_input=False)
        state_dict = model_zoo.load_url(model_urls['inception_v3_google'])
        state_dict_rm_aux = {k: v for k, v in state_dict.items() if 'AuxLogits' not in k}
        inception_v3.load_state_dict(state_dict_rm_aux)

        layers = list(inception_v3.children())[:-1]
        layers.insert(3, torch.nn.MaxPool2d(3, 2))
        layers.insert(6, torch.nn.MaxPool2d(3, 2))
        layers.append(torch.nn.AvgPool2d(8))

        self.layers_1 = torch.nn.Sequential(*layers[:3]) # 64 x 147 x 147
        self.layers_2 = torch.nn.Sequential(*layers[3:6]) # 192 x 71 x 71
        self.layers_3 = torch.nn.Sequential(*layers[6:10]) # 288 x 35 x 35
        self.layers_4 = torch.nn.Sequential(*layers[10:15]) # 768 x 17 x 17
        self.tail_layers   = torch.nn.Sequential(*layers[15:])

        self.conv_6 = torch.nn.Sequential(
            torch.nn.Conv2d(768, self.k, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.k),
            torch.nn.ReLU()
        )

        self.conv_7 = torch.nn.Sequential(
            torch.nn.Conv2d(768, self.k, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(self.k),
            torch.nn.ReLU()
        )

        self.gap_layer_1 = torch.nn.AdaptiveAvgPool2d(1)
        self.gap_layer_2 = torch.nn.AdaptiveAvgPool2d(1)

        self.bn1 = torch.nn.BatchNorm1d(k)
        self.bn2 = torch.nn.BatchNorm1d(k)

        self.ppm = PyramidPooling(k, 512)

        self.ppm_out1 = torch.nn.Sequential(
            torch.nn.Conv2d(k, 288, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(288),
            torch.nn.ReLU())

        self.ppm_out2 = torch.nn.Sequential(
            torch.nn.Conv2d(k, 192, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(192),
            torch.nn.ReLU())

        self.ppm_out3 = torch.nn.Sequential(
            torch.nn.Conv2d(k, 64, kernel_size=1, stride=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU())

        self.fam1 = FAM(768, 288, [35, 35], [14, 14], [8, 8], [4, 4])
        self.feat_agg1 = FeatAgg(288, 288)

        self.fam2 = FAM(288, 192, [71, 71], [28, 28], [14, 14], [8, 8])
        self.feat_agg2 = FeatAgg(192, 192)

        self.fam3 = FAM(192, 64, [147, 147], [56, 56], [28, 28], [14, 14])
        self.feat_agg3 = FeatAgg(64, 64)

        self.fam4 = FAM(64, 64, [147, 147], [56, 56], [28, 28], [14, 14])
        self.feat_agg4 = FeatAgg(64, 64)

        self.fam5 = FAM(64, 192, [71, 71], [28, 28], [14, 14], [8, 8])
        self.feat_agg5 = FeatAgg(192, 192)

        self.fam6 = FAM(192, 288, [35, 35], [14, 14], [8, 8], [4, 4])
        self.feat_agg6 = FeatAgg(288, 288)

        self.gap_layer_p0 = torch.nn.AdaptiveAvgPool2d(1)
        self.gap_layer_p1 = torch.nn.AdaptiveAvgPool2d(1)
        self.gap_layer_p2 = torch.nn.AdaptiveAvgPool2d(1)

        self.conv_p0 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=8, stride=8),
            torch.nn.Conv2d(64, self.k, kernel_size=1, stride=1)
        )
        self.conv_p1 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=4, stride=4),
            torch.nn.Conv2d(192, self.k, kernel_size=1, stride=1)
        )
        self.conv_p2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(288, self.k, kernel_size=1, stride=1)
        )

    def attention_sublayers(self, feats, embedding_layers, latent):
        feats = feats.view((feats.size(0), self.k, -1))
        feats = feats.transpose(dim0=1, dim1=2)
        feats = feats + latent.unsqueeze(1)
        feats = feats.transpose(dim0=1, dim1=2)

        feats = embedding_layers(feats).squeeze(-1)
        p = F.softmax(feats, dim=1)
        return p

    def upsample(self, x, size):  # 上采样使用双线性插值
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        feats_1 = self.layers_1(x)
        feats_2 = self.layers_2(feats_1)
        feats_3 = self.layers_3(feats_2)
        feats_4 = self.layers_4(feats_3)
        conv_6 = self.conv_6(feats_4)
        conv_7 = self.conv_7(feats_4)

        gap = F.relu(torch.squeeze(self.gap_layer_1(conv_6)))

        gap1 = F.relu(torch.squeeze(self.gap_layer_2(conv_7)))

        attr = self.bn1(gap)

        latent = self.bn2(gap1)

        # attr_cos = torch.sigmoid(attr)
        '''
        x = F.relu(self.fc4(self.fc_layers(self.tail_layer(conv5_4).view(-1, 25088))))
        attr = self.bn1(x[:, :self.k])
        latent = self.bn2(x[:, self.k:])
        '''
        ppm_feat = conv_7
        ppm_feat = self.ppm(ppm_feat)

        ppm_feat1 = self.ppm_out1(self.upsample(ppm_feat, [35, 35]))
        ppm_feat2 = self.ppm_out2(self.upsample(ppm_feat, [71, 71]))
        ppm_feat3 = self.ppm_out3(self.upsample(ppm_feat, [147, 147]))

        fam1 = self.fam1(feats_4)
        feat_agg1 = self.feat_agg1(feats_3, ppm_feat1, fam1)

        fam2 = self.fam2(feat_agg1)
        feat_agg2 = self.feat_agg2(feats_2, ppm_feat2, fam2)

        fam3 = self.fam3(feat_agg2)
        feat_agg3 = self.feat_agg3(feats_1, ppm_feat3, fam3)

        fam4 = self.fam4(feat_agg3)
        feat_agg4 = self.feat_agg4(feats_1, ppm_feat3, fam4)

        fam5 = self.fam5(feat_agg4)
        feat_agg5 = self.feat_agg5(feats_2, ppm_feat2, fam5)

        fam6 = self.fam6(feat_agg5)
        feat_agg6 = self.feat_agg6(feats_3, ppm_feat1, fam6)

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
