from torch import nn


# Ref: ADNet (Legacy module)
class PyramidFeature(nn.Module):
    def __init__(self, in_channels=6, n_channels=64):
        super(PyramidFeature, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, n_channels, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.feature_extraction = ResBlock()

        self.down_sample1 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.down_sample2 = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        x_in = self.conv1(x)
        x1 = self.feature_extraction(x_in)
        x2 = self.down_sample1(x1)
        x3 = self.down_sample2(x2)
        return [x1, x2, x3]


class HomoPyramidFeature(nn.Module):
    """Generate pyramid features using the same convolution layers.

    Args:
        n_channels (int, optional): the number of feature channels.
            Default: 64
    """

    def __init__(self, n_channels=64, n_levels=3):
        super(HomoPyramidFeature, self).__init__()
        self.n_levels = n_levels
        self.feature_extract = ResBlock(n_channels)
        self.down_sample = _create_down_sample_conv(n_channels)

    def forward(self, x):
        feat_li = [self.feature_extract(x)]
        for i in range(self.n_levels - 1):
            feat_li.append(self.down_sample(feat_li[-1]))
        return feat_li


class HeteroPyramidFeature(nn.Module):
    """Generate pyramid features using different convolution layers for each level.

    Args:
        n_channels (int, optional): the number of feature channels.
            Default: 64
    """

    def __init__(self, n_channels=64, n_levels=3):
        super(HeteroPyramidFeature, self).__init__()
        self.n_levels = n_levels
        self.feature_extraction = ResBlock(n_channels)
        self.down_sample = nn.ModuleList()
        for i in range(n_levels - 1):
            self.down_sample.append(_create_down_sample_conv(n_channels))

    def forward(self, x):
        feat_li = [self.feature_extraction(x)]
        for i in range(self.n_levels - 1):
            feat_li.append(self.down_sample[i](feat_li[-1]))
        return feat_li


class ResidualPyramidFeature(nn.Module):
    """
    todo: Finish this (consider more about how to perform the down-sample)
    """

    def __init__(self, n_channels=64, n_levels=3):
        super(ResidualPyramidFeature, self).__init__()
        self.n_levels = n_levels
        self.feature_extraction = ResBlock(n_channels)
        self.down_sample = _create_down_sample_conv(n_channels)

    def forward(self, x):
        feat_l1 = self.feature_extraction(x)
        residual_l1 = x - feat_l1
        feat_l2 = self.down_sample


# Ref: ADNet
class ResBlock(nn.Module):

    def __init__(self, n_channels=64, res_scale=1, act='relu'):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        res = self.conv2(self.act(self.conv1(x)))
        return x + res * self.res_scale


def _create_down_sample_conv(n_channels):
    return nn.Sequential(
        nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True),
        nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(negative_slope=0.1, inplace=True)
    )
