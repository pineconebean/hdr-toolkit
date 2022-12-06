import torch.nn as nn

from hdr_toolkit.networks.blocks import HomoPyramidFeature, AHDRMergingNet, PyramidSFT, PCDAlign, HeteroPyramidFeature
from hdr_toolkit.util.registry import NETWORK_REGISTRY


@NETWORK_REGISTRY.register('psftd_net')
class PSFTDNet:

    def __init__(self, in_channels=3, n_channels=64, out_activation='relu', same_conv_for_pyramid=True):
        self.first_conv = nn.Conv2d(in_channels, n_channels, 3, padding=1)

        if same_conv_for_pyramid:
            self.pyramid_feature_extract = HomoPyramidFeature(n_channels)
        else:
            self.pyramid_feature_extract = HeteroPyramidFeature(n_channels)

        self.align_module = PCDAlign(n_channels)
        self.psft = PyramidSFT()
        self.merging = AHDRMergingNet(n_channels * 6, n_channels, out_activation)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, short, mid, long):
        pass
