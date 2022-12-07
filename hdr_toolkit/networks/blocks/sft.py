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
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, non_ref_feat, ref_feat):
        # generate condition maps for each level
        cond_li = []
        for i in range(self.n_levels):
            cond_li.append(self.cond_convs[i](cat((non_ref_feat[i], ref_feat[i]), dim=1)))
        # do SFT for each level
        print(non_ref_feat[-1].shape)
        print(cond_li[-1].shape)
        sft_last_out = self.sft[-1](non_ref_feat[-1], cond_li[-1])
        for i in reversed(range(self.n_levels - 1)):
            sft_curr_out = self.sft[i](non_ref_feat[i], cond_li[i])
            sft_last_out = self.up_sample(sft_last_out)
            sft_curr_out = self.leaky_relu(self.feat_cat_conv[i](cat((sft_curr_out, sft_last_out), dim=1)))
            sft_last_out = sft_curr_out

        return sft_last_out


class PyramidSFT(nn.Module):
    """EDVR network structure for video super-resolution.

    Now only support X4 upsampling factor.
    Paper:
    EDVR: Video Restoration with Enhanced Deformable Convolutional Networks.

    Params:
        n_channels (int): Channel number of inputs.
        out_channels (int): Channel number of outputs.
        mid_channels (int): Channel number of intermediate features.
            Default: 64.
        num_frames (int): Number of input frames. Default: 5.
        deform_groups (int): Deformable groups. Defaults: 8.
        num_blocks_extraction (int): Number of blocks for feature extraction.
            Default: 5.
        num_blocks_reconstruction (int): Number of blocks for reconstruction.
            Default: 10.
        center_frame_idx (int): The index of center frame. Frame counting from
            0. Default: 2.
        with_tsa (bool): Whether to use TSA module. Default: True.
    """
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

        return sft_prev_out


def _create_condition_convs(n_channels):
    return nn.Sequential(
        nn.Conv2d(n_channels * 2, n_channels * 2, 3, 1, 1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )
