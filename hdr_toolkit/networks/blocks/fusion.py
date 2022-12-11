from torch import nn
import torch

from hdr_toolkit.networks.blocks import DRDB, BAEnhancedDRDB, BAPack


def _get_activation(activation_name):
    if activation_name == 'relu':
        return nn.ReLU(inplace=True)
    elif activation_name == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise ValueError('Invalid activation for the output layer')


class AHDRMergingNet(nn.Module):

    def __init__(self, in_channels, n_channels, out_activation='relu'):
        super(AHDRMergingNet, self).__init__()
        self.out_activation = _get_activation(out_activation)

        self.conv1 = nn.Conv2d(in_channels, n_channels, 3, padding=1)

        self.drdb_layers = nn.ModuleList()
        for _ in range(3):
            self.drdb_layers.append(DRDB(n_channels, growth_rate=32, n_dense_layers=3))

        self.gff1x1 = nn.Conv2d(n_channels * 3, n_channels, kernel_size=1, padding=0)
        self.gff3x3 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding='same')
        self.conv_up = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding='same')
        self.out_conv = nn.Conv2d(n_channels, 3, kernel_size=3, padding='same')
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, features, ref_feature):
        x = self.conv1(features)

        x1 = self.drdb_layers[0](x)
        x2 = self.drdb_layers[1](x1)
        x3 = self.drdb_layers[2](x2)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.gff1x1(x)
        x = self.gff3x3(x)
        x = x + ref_feature
        x = self.relu(self.conv_up(x))
        x = self.out_conv(x)

        return self.out_activation(x)


class BAMergingNet(nn.Module):

    def __init__(self, in_channels, n_channels, ba_type='default', out_activation='sigmoid'):
        super(BAMergingNet, self).__init__()
        self.out_activation = _get_activation(out_activation)
        self.conv1 = nn.Conv2d(in_channels, n_channels, 3, padding=1)

        self.drdb_layers = nn.ModuleList()
        for _ in range(3):
            self.drdb_layers.append(BAEnhancedDRDB(n_channels, 32, 3, ba_type))

        self.gff1x1 = nn.Conv2d(n_channels * 3, n_channels, kernel_size=1, padding=0)
        self.gff3x3 = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding='same')
        self.ba_pack = BAPack.create(n_channels, ba_type)
        self.conv_up = nn.Conv2d(n_channels, n_channels, kernel_size=3, padding='same')
        self.out_conv = nn.Conv2d(n_channels, 3, kernel_size=3, padding='same')
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, feat, ref_feat):
        x = self.conv1(feat)

        x1 = self.drdb_layers[0](x, ref_feat)
        x2 = self.drdb_layers[1](x1, ref_feat)
        x3 = self.drdb_layers[2](x2, ref_feat)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.gff1x1(x)
        x = self.gff3x3(x)
        x = self.ba_pack(x, ref_feat)
        x = x + ref_feat
        x = self.relu(self.conv_up(x))
        x = self.out_conv(x)

        return self.out_activation(x)
