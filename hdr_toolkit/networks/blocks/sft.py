import torch.nn as nn
import torch.nn.functional as F
from torch import cat


# reference: https://github.com/xinntao/SFTGAN/blob/master/pytorch_test/architectures.py
class SFTLayer(nn.Module):
    def __init__(self, fea_channels=64, cond_channels=64, mid_channels=64):
        super(SFTLayer, self).__init__()
        self.scale_conv0 = nn.Conv2d(cond_channels, mid_channels, 1)
        self.scale_conv1 = nn.Conv2d(mid_channels, fea_channels, 1)
        self.shift_conv0 = nn.Conv2d(cond_channels, mid_channels, 1)
        self.shift_conv1 = nn.Conv2d(mid_channels, fea_channels, 1)

    def forward(self, x, condition):
        # x: feature, condition: condition
        scale = self.scale_conv1(F.leaky_relu(self.scale_conv0(condition), 0.1, inplace=True))
        shift = self.shift_conv1(F.leaky_relu(self.shift_conv0(condition), 0.1, inplace=True))
        return x * (scale + 1) + shift


class SFTBlock(nn.Module):
    def __init__(self, fea_channels, cond_channels, mid_channels, learn_residual=False):
        super(SFTBlock, self).__init__()
        self.learn_residual = learn_residual
        self.sft0 = SFTLayer(fea_channels, cond_channels, mid_channels)
        self.sft1 = SFTLayer(fea_channels, cond_channels, mid_channels)
        self.conv0 = nn.Conv2d(fea_channels, fea_channels, 3, 1, 1)
        self.conv1 = nn.Conv2d(fea_channels, fea_channels, 3, 1, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, condition):
        fea = self.sft0(x, condition)
        fea = self.leaky_relu(self.conv0(fea))
        fea = self.sft1(fea, condition)
        fea = self.conv1(fea)

        if self.learn_residual:
            return x + fea  # not reasonable for modulate non-reference features
        else:
            return fea


class NaivePyramidSFT(nn.Module):
    def __init__(self, n_channels=64, n_levels=3, simple_sft=False):
        super(NaivePyramidSFT, self).__init__()
        self.n_levels = n_levels
        self.cond_convs = nn.ModuleList()
        self.sft = nn.ModuleList()
        self.feat_cat_conv = nn.ModuleList()
        for i in range(n_levels):
            self.cond_convs.append(_create_condition_convs(n_channels))
            if i < n_levels - 1:
                self.feat_cat_conv.append(nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1))
            if simple_sft:
                self.sft.append(SFTLayer(n_channels, n_channels, n_channels))
            else:
                self.sft.append(SFTBlock(n_channels, n_channels, n_channels))
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, non_ref_feat, ref_feat):
        cond_li = []
        for i in range(self.n_levels):
            cond_li = self.cond_convs[i](cat((non_ref_feat[i], ref_feat[i]), dim=1))

        sft_last_out = self.sft[-1](non_ref_feat[-1], cond_li[-1])
        for i in reversed(range(self.n_levels - 1)):
            sft_curr_out = self.sft[i](non_ref_feat[i], cond_li[i])
            sft_last_out = self.up_sample(sft_last_out)
            sft_curr_out = self.leaky_relu(self.feat_cat_conv(cat((sft_curr_out, sft_last_out), dim=1)))
            sft_last_out = sft_curr_out

        return sft_last_out


class PyramidSFT(nn.Module):

    def __init__(self, n_channels=64, n_levels=3, simple_sft=False):
        super(PyramidSFT, self).__init__()
        self.n_levels = n_levels
        self.cond_conv_first = nn.ModuleList()
        self.cond_conv_concat = nn.ModuleList()
        self.cond_conv_last = nn.ModuleList()
        for i in range(n_levels):
            if simple_sft:
                self.sft.append(SFTLayer(n_channels, n_channels, n_channels))
            else:
                self.sft.append(SFTBlock(n_channels, n_channels, n_channels))
                self.cond_convs.append(nn.Conv2d())
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, non_ref_feat, ref_feat):
        sft_li, cond_li = [], []
        for i in range(self.n_levels):
            non_ref_feat_li, ref_feat_li = non_ref_feat[i], ref_feat[i]
            cond_li = self.cond_convs[i](cat((non_ref_feat_li, ref_feat_li), dim=1))
            sft_li.append(self.sft[i](non_ref_feat_li, cond_li))

        for i in reversed(range(self.n_levels)):
            non_ref_feat_li, ref_feat_li = non_ref_feat[i], ref_feat[i]
            cond_li.append(self.cond_convs[i](cat((non_ref_feat_li, ref_feat_li), dim=1)))
            if not self.pyramid_cond:
                sft_li.append(self.sft[i](non_ref_feat_li, cond_li))
            else:
                if i != self.n_levels - 1:
                    # if not the highest level
                    up_sampled_last_cond = self.up_sample(cond_li[-1])
                    cond_li.append(self.cond_convs[i]())


def _create_condition_convs(n_channels):
    return nn.Sequential(
        nn.Conv2d(n_channels * 2, n_channels * 2, kernel_size=3, padding=1, bias=True),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(n_channels * 2, n_channels, kernel_size=3, padding=1, bias=True),
        nn.LeakyReLU(inplace=True)
    )
