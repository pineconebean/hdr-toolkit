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


class ResSFTPack(nn.Module):
    """
    Residual Spatial Feature Transform Pack

    The ResSFTPack accepts a feature and an extra feature as inputs, and learn residuals of
    the extra feature using spatial feature transform.
    """

    def __init__(self, feat_channels, mid_channels, n_sft=1, only_sft=True, simple_cond=True):
        """
        Args:
            feat_channels: channels of features.
            mid_channels: channels of the middle layer in SFT which is used to provide fine-grained control
            n_sft: the number of sft layers
            only_sft: if True, there will be no convolution layers after applying SFT to current features
            simple_cond: only use one convolution layer to learn the condition map.
                This is used to be same as the attention module used in AHDRNet
        """
        super(ResSFTPack, self).__init__()
        self.n_sft = n_sft
        self.only_sft = only_sft
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if simple_cond:
            self.cond = nn.Sequential(nn.Conv2d(feat_channels * 2, feat_channels * 2, 3, 1, 1), self.leaky_relu)
            cond_channels = feat_channels * 2
        else:
            self.cond = _create_condition_convs(feat_channels)
            cond_channels = feat_channels

        self.sft_layers = nn.ModuleList()
        self.conv_after_sft = nn.ModuleList()
        for _ in range(n_sft):
            self.sft_layers.append(SFTLayer(feat_channels, cond_channels, mid_channels))
            if not only_sft:
                self.conv_after_sft.append(nn.Conv2d(feat_channels, feat_channels, 3, 1, 1))

    def forward(self, x, extra_feat):
        cond_maps = self.cond(cat((x, extra_feat), dim=1))
        feat = x

        if self.only_sft:
            feat = self.sft_layers[0](feat, cond_maps)
        else:
            for i in range(self.n_sft):
                feat = self.sft_layers[i](feat, cond_maps)
                feat = self.leaky_relu(self.conv_after_sft[i](feat))
        return feat + extra_feat


class NaivePyramidSFT(nn.Module):
    """PyramidSFT which generates the condition maps at each level without condition maps from higher level"""

    def __init__(self, n_channels, n_levels=3, simple_sft=False, sft_learn_residual=False):
        """
        Args:
            n_channels (int): the number of feature channels.
            n_levels (int): the number of pyramid levels. Default: 3
            simple_sft (bool): If True, SFTLayer will be used to refine the non-reference feature.
                Otherwise, SFTBlock will be used. Default: False
            sft_learn_residual (bool): Whether the SFT learns residuals. Only works when simple_sft is True.
                Default: False
        """
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
                self.sft.append(SFTBlock(n_channels, n_channels, n_channels, sft_learn_residual))

        # cascaded sft block
        self.cas_cond_convs = _create_condition_convs(n_channels)
        if simple_sft:
            self.cas_sft = SFTLayer(n_channels, n_channels, n_channels)
        else:
            self.cas_sft = SFTBlock(n_channels, n_channels, n_channels, sft_learn_residual)

        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, non_ref_feat, ref_feat):
        # generate condition maps for each level
        cond_maps = []
        for i in range(self.n_levels):
            cond_maps.append(self.cond_convs[i](cat((non_ref_feat[i], ref_feat[i]), dim=1)))
        # do SFT for each level
        sft_last_out = self.sft[-1](non_ref_feat[-1], cond_maps[-1])
        for i in reversed(range(self.n_levels - 1)):
            sft_curr_out = self.sft[i](non_ref_feat[i], cond_maps[i])
            sft_last_out = self.up_sample(sft_last_out)
            sft_curr_out = self.leaky_relu(self.feat_cat_conv[i](cat((sft_curr_out, sft_last_out), dim=1)))
            sft_last_out = sft_curr_out

        cas_cond_map = self.cas_cond_convs(cat((sft_last_out, ref_feat[0]), dim=1))
        return self.cas_sft(sft_last_out, cas_cond_map)


class PyramidSFT(nn.Module):

    def __init__(self, n_channels=64, n_levels=3, simple_sft=False, sft_learn_residual=False):
        super(PyramidSFT, self).__init__()
        self.n_levels = n_levels
        self.cond_conv_first = nn.ModuleList()
        self.cond_conv_concat = nn.ModuleList()
        self.cond_conv_last = nn.ModuleList()
        self.sft = nn.ModuleList()
        self.feat_cat_conv = nn.ModuleList()
        for i in range(n_levels):
            self.cond_conv_first.append(nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1))
            self.cond_conv_last.append(nn.Conv2d(n_channels, n_channels, 3, 1, 1))
            if i != n_levels - 1:
                self.cond_conv_concat.append(nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1))
                self.feat_cat_conv.append(nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1))
            if simple_sft:
                self.sft.append(SFTLayer(n_channels, n_channels, n_channels))
            else:
                self.sft.append(SFTBlock(n_channels, n_channels, n_channels, sft_learn_residual))

        # cascaded sft block
        self.cas_cond_convs = _create_condition_convs(n_channels)
        if simple_sft:
            self.cas_sft = SFTLayer(n_channels, n_channels, n_channels)
        else:
            self.cas_sft = SFTBlock(n_channels, n_channels, n_channels, sft_learn_residual)

        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, non_ref_feat, ref_feat):
        # generate condition maps and do the SFT for the highest level
        cond_prev_out = self.leaky_relu(self.cond_conv_first[-1](cat((non_ref_feat[-1], ref_feat[-1]), dim=1)))
        cond_prev_out = self.leaky_relu(self.cond_conv_last[-1](cond_prev_out))
        sft_prev_out = self.sft[-1](non_ref_feat[-1], cond_prev_out)
        for i in reversed(range(self.n_levels - 1)):
            # generate condition maps for current level
            cond_curr_out = self.leaky_relu(self.cond_conv_first[i](cat((non_ref_feat[i], ref_feat[i]), dim=1)))
            cond_prev_out = self.up_sample(cond_prev_out)
            cond_curr_out = self.leaky_relu(self.cond_conv_concat[i](cat((cond_curr_out, cond_prev_out), dim=1)))
            cond_curr_out = self.leaky_relu(self.cond_conv_last[i](cond_curr_out))

            # do spatial feature transformation for current level
            sft_curr_out = self.sft[i](non_ref_feat[i], cond_curr_out)
            sft_prev_out = self.up_sample(sft_prev_out)
            sft_curr_out = self.leaky_relu(self.feat_cat_conv[i](cat((sft_curr_out, sft_prev_out), dim=1)))
            sft_prev_out = sft_curr_out

        cas_cond_map = self.cas_cond_convs(cat((sft_prev_out, ref_feat[0]), dim=1))
        return self.cas_sft(sft_prev_out, cas_cond_map)


def _create_condition_convs(n_channels):
    return nn.Sequential(
        nn.Conv2d(n_channels * 2, n_channels * 2, 3, 1, 1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )
