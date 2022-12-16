import torch.nn as nn
import torch
from hdr_toolkit.networks.blocks import AHDRMergingNet, SpatialAttention, ResSFTPack, VanillaDA


class ResRefAHDR(nn.Module):

    def __init__(self, n_channels, out_activation='sigmoid', conv_after_res='same'):
        super(ResRefAHDR, self).__init__()
        self.option_conv_after_res = conv_after_res
        self.extract_feature = nn.Sequential(
            nn.Conv2d(6, n_channels, 3, padding='same'),
            nn.LeakyReLU(inplace=True)
        )

        self.att_short_mid = SpatialAttention(n_channels)
        self.att_long_mid = SpatialAttention(n_channels)

        self.conv_after_res_sm = nn.Conv2d(n_channels, n_channels, 3, 1, 1)
        self.conv_after_res_lm = nn.Conv2d(n_channels, n_channels, 3, 1, 1)

        self.merging = AHDRMergingNet(n_channels * 3, n_channels, out_activation)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, short, mid, long):
        z_short = self.extract_feature(short)
        z_mid = self.extract_feature(mid)
        z_long = self.extract_feature(long)

        # let z_short learn the residual
        z_short = self.att_short_mid(z_short, z_mid) + z_mid
        z_long = self.att_long_mid(z_long, z_mid) + z_mid

        if self.option_conv_after_res == 'same':
            z_short = self.leaky_relu(self.conv_after_res_sm(z_short))
            z_long = self.leaky_relu(self.conv_after_res_sm(z_long))
        elif self.option_conv_after_res == 'diff':
            z_short = self.leaky_relu(self.conv_after_res_sm(z_short))
            z_long = self.leaky_relu(self.conv_after_res_lm(z_long))
        elif self.option_conv_after_res == 'none':
            return self.merging(torch.cat((z_short, z_mid, z_long), dim=1), z_mid)

        return self.merging(torch.cat((z_short, z_mid, z_long), dim=1), z_mid)


class ResRefSFTNet(nn.Module):

    def __init__(self, n_channels, out_activation='sigmoid', **kwargs):
        super(ResRefSFTNet, self).__init__()
        self.extract_feat = nn.Sequential(
            nn.Conv2d(6, n_channels, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.sft_pack_sm = ResSFTPack(n_channels, n_channels, **kwargs)
        self.sft_pack_lm = ResSFTPack(n_channels, n_channels, **kwargs)

        self.merging = AHDRMergingNet(n_channels * 3, n_channels, out_activation)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, short, mid, long):
        feat_s = self.extract_feat(short)
        feat_m = self.extract_feat(mid)
        feat_l = self.extract_feat(long)

        feat_s = self.sft_pack_sm(feat_s, feat_m)
        feat_l = self.sft_pack_lm(feat_l, feat_m)

        return self.merging(torch.cat((feat_s, feat_m, feat_l), dim=1), feat_m)

    @classmethod
    def create(cls, sft_res_type, n_channels, out_activation='sigmoid'):
        """Create ResRefSFTNet with pre-defined settings

        Args:
            sft_res_type:
                `two-sft`: two sft layers and two convolution layers will be used
                `default`: one sft layer and no convolution layers after the sft layer
                `one-sft`: one sft layer and one convolution layer will be used
            n_channels: number of feature channels
            out_activation: the activation function used for last layer of the whole network
        """
        if sft_res_type == 'two-sft':
            return cls(n_channels, out_activation, n_sft=2, only_sft=False, simple_cond=False)
        elif sft_res_type == 'default':
            return cls(n_channels, out_activation)
        elif sft_res_type == 'one-sft':
            return cls(n_channels, out_activation, n_sft=1, only_sft=False, simple_cond=False)
        else:
            raise KeyError(f'Pre-defined type {sft_res_type} is not valid')


class ResRefDANet(nn.Module):

    def __init__(self, n_channels, out_activation='sigmoid', groups=8, learn_res=True, **kwargs):
        super(ResRefDANet, self).__init__()
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.learn_res = learn_res

        self.extract_feat = nn.Sequential(
            nn.Conv2d(6, n_channels, 3, 1, 1),
            self.leaky_relu
        )

        self.da_sm = VanillaDA(n_channels, groups, **kwargs)
        self.da_lm = VanillaDA(n_channels, groups, **kwargs)

        self.merging = AHDRMergingNet(n_channels * 3, n_channels, out_activation)

    def forward(self, short, mid, long):
        feat_s = self.extract_feat(short)
        feat_m = self.extract_feat(mid)
        feat_l = self.extract_feat(long)

        feat_s = self.da_sm(feat_s, feat_m)
        feat_l = self.da_lm(feat_l, feat_m)
        if self.learn_res:
            feat_s = feat_s + feat_m
            feat_l = feat_l + feat_m

        return self.merging(torch.cat((feat_s, feat_m, feat_l), dim=1), feat_m)

    @classmethod
    def create(cls, pre_defined, n_channels, out_activation):
        if pre_defined == 'default':
            return cls(n_channels, out_activation)
        elif pre_defined == 'no_res':
            return cls(n_channels, out_activation, learn_res=False)
        elif pre_defined == 'no_act':
            return cls(n_channels, out_activation, with_act=False)
        elif pre_defined == 'groups16':
            return cls(n_channels, out_activation, groups=16)
        else:
            raise KeyError(f'Pre-defined type {pre_defined} is not valid')
