import torch.nn as nn
import torch.nn.functional as F


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


class SFTResBlock(nn.Module):
    def __init__(self, fea_channels, cond_channels, mid_channels):
        super(SFTResBlock, self).__init__()
        self.sft0 = SFTLayer(fea_channels, cond_channels, mid_channels)
        self.sft1 = SFTLayer(fea_channels, cond_channels, mid_channels)
        self.conv0 = nn.Conv2d(fea_channels, fea_channels, 3, 1, 1)
        self.conv1 = nn.Conv2d(fea_channels, fea_channels, 3, 1, 1)

    def forward(self, x, condition):
        fea = self.sft0(x, condition)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1(fea, condition)
        fea = self.conv1(fea)
        return x + fea


class ResBlock(nn.Module):
    def __init__(self, n_channels):
        super(ResBlock, self).__init__()
        self.conv0 = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)
        self.conv1 = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)

    def forward(self, x):
        fea = F.relu(self.conv0(x), True)
        fea = self.conv1(fea)
        return x + fea


class PyramidSFT(nn.Module):

    def __init__(self):
        super(PyramidSFT, self).__init__()

    def forward(self, x):
        pass
