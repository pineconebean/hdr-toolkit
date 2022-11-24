import torch
import torch.nn as nn

from hdr_toolkit.networks.blocks.drdb import DRDB


class AHDRNet(nn.Module):

    def __init__(self, n_channels=64, out_activation='relu'):
        super(AHDRNet, self).__init__()
        self.extract_feature = nn.Sequential(
            nn.Conv2d(6, n_channels, 3, padding='same'),
            nn.LeakyReLU(inplace=True)
        )

        self.att_sr = AttentionModule()
        self.att_lr = AttentionModule()

        self.merging = MergingNet(n_channels * 3, n_channels, out_activation)

    def forward(self, low, ref, high):
        z_low = self.extract_feature(low)
        z_ref = self.extract_feature(ref)
        z_high = self.extract_feature(high)

        att_lr = self.att_sr(z_low, z_ref)
        att_hr = self.att_lr(z_high, z_ref)

        z_low = z_low * att_lr
        z_high = z_high * att_hr

        return self.merging(torch.cat((z_low, z_ref, z_high), dim=1), z_ref)


class AttentionModule(nn.Module):

    def __init__(self, n_channels=64):
        super(AttentionModule, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(n_channels * 2, n_channels, 3, padding='same'),
            nn.LeakyReLU(inplace=False),
            nn.Conv2d(n_channels, n_channels, 3, padding='same'),
            nn.Sigmoid()
        )

    def forward(self, z_i, z_ref):
        cat_feature = torch.cat((z_i, z_ref), dim=1)
        return self.network(cat_feature)


class MergingNet(nn.Module):

    def __init__(self, in_channels, n_channels, out_activation='relu'):
        super(MergingNet, self).__init__()
        if out_activation == 'relu':
            self.out_activation = nn.ReLU(inplace=True)
        elif out_activation == 'sigmoid':
            self.out_activation = nn.Sigmoid()
        else:
            raise ValueError('Invalid activation for the output layer')

        self.conv1 = nn.Conv2d(in_channels, n_channels, 3, padding=1)

        self.drdb0 = DRDB(n_channels, 32, 3)
        self.drdb1 = DRDB(n_channels, 32, 3)
        self.drdb2 = DRDB(n_channels, 32, 3)

        self.gff1x1 = nn.Conv2d(n_channels * 3, n_channels, kernel_size=1, padding=0)
        self.gff3x3 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding='same')
        self.conv_up = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding='same')
        self.out_conv = nn.Conv2d(n_channels, 3, kernel_size=3, padding='same')

    def forward(self, x, ref):
        x = self.conv1(x)

        x1 = self.drdb0(x)
        x2 = self.drdb1(x1)
        x3 = self.drdb2(x2)

        drdb_outs = (x1, x2, x3)

        x = torch.cat(drdb_outs, dim=1)
        x = self.gff1x1(x)
        x = self.gff3x3(x)
        x = x + ref
        x = self.conv_up(x)
        x = self.out_conv(x)

        return self.out_activation(x)
