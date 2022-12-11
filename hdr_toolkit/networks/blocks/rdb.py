import torch
import torch.nn as nn
import torch.nn.functional as F
from hdr_toolkit.networks.blocks.attention import BAPack


# reference: https://github.com/qingsenyangit/AHDRNet/blob/master/model.py
class DRDB(nn.Module):

    def __init__(self, n_channels, growth_rate, n_dense_layers):
        super(DRDB, self).__init__()
        curr_in_channels = n_channels
        drdb_conv_layers = []
        for _ in range(n_dense_layers):
            drdb_conv_layers.append(DRDBConv(curr_in_channels, growth_rate))
            curr_in_channels += growth_rate
        self.drdb_conv_layers = nn.Sequential(*drdb_conv_layers)
        self.conv_1x1 = nn.Conv2d(curr_in_channels, n_channels, kernel_size=1, bias=True)

    def forward(self, x):
        out = self.drdb_conv_layers(x)
        out = self.conv_1x1(out)
        return out + x


class BAEnhancedDRDB(DRDB):

    def __init__(self, n_channels, growth_rate, n_dense_layers, ba_type='default'):
        super(BAEnhancedDRDB, self).__init__(n_channels, growth_rate, n_dense_layers)
        self.ba_pack = BAPack.create(n_channels, ba_type)

    def forward(self, x, extra_feat):
        out = self.drdb_conv_layers(x)
        out = self.conv_1x1(out)
        out = self.ba_pack(out, extra_feat)
        return out + x


class DeformRDB(nn.Module):

    def __init__(self):
        super(DeformRDB, self).__init__()

    def forward(self, x):
        pass


class DRDBConv(nn.Module):

    def __init__(self, in_channels, growth_rate, kernel_size=3):
        super(DRDBConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size, padding='same', dilation=2, bias=True)

    def forward(self, x):
        out = F.relu(self.conv(x))
        return torch.cat((x, out), dim=1)
