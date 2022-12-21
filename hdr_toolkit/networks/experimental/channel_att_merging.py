import torch.nn as nn

from hdr_toolkit.networks import AHDRNet
from hdr_toolkit.networks.blocks import BAMergingNet


class BAHDRNet(nn.Module):

    def __init__(self, n_channels, out_activation='relu', ba_type='default'):
        super(BAHDRNet, self).__init__()
        self.ahdr = AHDRNet(n_channels, out_activation)
        # reference about replace a layer:
        #   https://discuss.pytorch.org/t/how-to-replace-a-layer-with-own-custom-variant/43586/13
        setattr(self.ahdr, 'merging', BAMergingNet(n_channels * 3, n_channels, 'default', out_activation))

    def forward(self, short, mid, long):
        return self.ahdr(short, mid, long)
