import torch.nn as nn
import torch

from hdr_toolkit.networks.blocks import PCDAlign, AHDRMergingNet, SpatialAttention, PyramidFeature


# ref: https://github.com/liuzhen03/ADNet/blob/main/DCNv2/dcn_v2.py
class ADNet(nn.Module):

    def __init__(self, n_channels, n_dense_layers, growth_rate, out_activation='relu'):
        super(ADNet, self).__init__()
        self.n_dense_layer = n_dense_layers
        self.n_channels = n_channels
        self.growth_rate = growth_rate

        # PCD align module
        self.pyramid_feats = PyramidFeature(in_channels=3, n_channels=n_channels)
        self.align_module = PCDAlign(n_channels)

        # Spatial attention module
        self.att_short_mid = SpatialAttention(n_channels)
        self.att_long_mid = SpatialAttention(n_channels)

        # feature extraction
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, n_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        self.merging = AHDRMergingNet(n_channels * 6, n_channels, out_activation=out_activation)

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x1, x2, x3):
        # *_t means LDR domain image, *_l means HDR domain (or exposure aligned) image
        x1_t, x1_l = x1[:, 0:3, ...], x1[:, 3:, ...]
        x2_t, x2_l = x2[:, 0:3, ...], x2[:, 3:, ...]
        x3_t, x3_l = x3[:, 0:3, ...], x3[:, 3:, ...]

        # pyramid features of linear domain
        f1_l = self.pyramid_feats(x1_l)
        f2_l = self.pyramid_feats(x2_l)
        f3_l = self.pyramid_feats(x3_l)
        f2_ = f2_l[0]

        # PCD alignment
        f1_aligned_l = self.align_module(f1_l, f2_l)
        f3_aligned_l = self.align_module(f3_l, f2_l)

        # Spatial attention module
        f1_t = self.feat_extract(x1_t)
        f2_t = self.feat_extract(x2_t)
        f3_t = self.feat_extract(x3_t)
        f1_t_ = self.att_short_mid(f1_t, f2_t)
        f3_t_ = self.att_long_mid(f3_t, f2_t)

        # Merge features to generate an HDR result
        return self.merging(torch.cat((f1_aligned_l, f1_t_, f2_, f2_t, f3_aligned_l, f3_t_), dim=1), f2_)
