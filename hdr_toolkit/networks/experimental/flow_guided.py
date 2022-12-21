import torch.nn as nn
import torch
from hdr_toolkit.networks.blocks import FlowGuidedDA, SpyNet
from torchvision.ops.deform_conv import DeformConv2d


class FDANet(nn.Module):

    def __init__(self, n_channels, flow_net_load_path, align_together=False, out_activation='sigmoid'):
        super(FDANet, self).__init__()
        self.flow_net = SpyNet(flow_net_load_path)
        self.align_together = align_together
        self.fda = FlowGuidedDA(n_channels, double_non_ref=align_together)

    def forward(self, short, mid, long):
        # compute flows
        flows = [self.flow_net(mid, short), self.flow_net(mid, long)]
        #
