import torch.nn as nn
import torch
from hdr_toolkit.networks.blocks import AHDRMergingNet, SpatialAttention


class ResRefAHDR(nn.Module):

    def __init__(self, n_channels, out_activation='sigmoid'):
        super(ResRefAHDR, self).__init__()
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

        z_short = self.leaky_relu(self.conv_after_res_sm(z_short))
        z_long = self.leaky_relu(self.conv_after_res_sm(z_long))

        return self.merging(torch.cat((z_short, z_mid, z_long), dim=1), z_mid)


class ResRefADNet(nn.Module):

    def __init__(self):
        super(ResRefADNet, self).__init__()

    def forward(self, short, mid, long):
        pass
