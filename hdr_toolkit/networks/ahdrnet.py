import torch
import torch.nn as nn

from hdr_toolkit.networks.blocks import AHDRMergingNet, SpatialAttention


class AHDRNet(nn.Module):

    def __init__(self, n_channels=64, out_activation='relu'):
        super(AHDRNet, self).__init__()
        self.extract_feature = nn.Sequential(
            nn.Conv2d(6, n_channels, 3, padding='same'),
            nn.LeakyReLU(inplace=True)
        )

        self.att_short_mid = SpatialAttention(n_channels)
        self.att_long_mid = SpatialAttention(n_channels)

        self.merging = AHDRMergingNet(n_channels * 3, n_channels, out_activation)

    def forward(self, short, mid, long):
        z_short = self.extract_feature(short)
        z_mid = self.extract_feature(mid)
        z_long = self.extract_feature(long)

        z_short = self.att_short_mid(z_short, z_mid)
        z_long = self.att_long_mid(z_long, z_mid)

        return self.merging(torch.cat((z_short, z_mid, z_long), dim=1), z_mid)
