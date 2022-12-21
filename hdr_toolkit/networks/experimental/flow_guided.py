import torch
import torch.nn as nn

from hdr_toolkit.networks.blocks import FlowGuidedDA, SpyNet, SpatialAttention, AHDRMergingNet, ResBlock


class FDANet(nn.Module):

    def __init__(self, n_channels, flow_net_load_path, align_together=False, out_activation='sigmoid'):
        super(FDANet, self).__init__()
        self.flow_net = SpyNet(flow_net_load_path)
        self.align_together = align_together
        self.fda = FlowGuidedDA(n_channels, double_non_ref=align_together)
        self.fda.requires_grad_(False)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # Spatial attention module
        self.att_short_mid = SpatialAttention(n_channels)
        self.att_long_mid = SpatialAttention(n_channels)

        self.feat_extract = nn.ModuleList()
        self.ldr_feat_extract = nn.Sequential(
            nn.Conv2d(3, n_channels, 3, 1, 1),
            self.leaky_relu
        )
        self.ea_feat_extract = nn.Sequential(
            nn.Conv2d(3, n_channels, 3, 1, 1),
            self.leaky_relu,
            ResBlock(n_channels)
        )

        merging_in_channels = n_channels * 5 if align_together else n_channels * 6
        self.merging = AHDRMergingNet(merging_in_channels, n_channels, out_activation=out_activation)

    def forward(self, short, mid, long):
        ldr_short, ea_short = short[:, :3, ...], short[:, 3:6, ...]
        ldr_mid, ea_mid = mid[:, :3, ...], mid[:, 3:6, ...]
        ldr_long, ea_long = long[:, :3, ...], long[:, 3:6, ...]

        # Align module
        flow_s, flow_l = self.flow_net(ea_mid, ea_short), self.flow_net(ea_mid, ea_long)
        ea_feat_s = self.ea_feat_extract(ea_short)
        ea_feat_m = self.ea_feat_extract(ea_mid)
        ea_feat_l = self.ea_feat_extract(ea_long)

        if self.align_together:
            aligned_feat = self.fda((ea_feat_s, ea_feat_l), ea_feat_m, (flow_s, flow_l))
            merged_feat = torch.cat((aligned_feat, ea_feat_m), dim=1)
        else:
            aligned_feat_s = self.fda(ea_feat_s, ea_feat_m, flow_s)
            aligned_feat_l = self.fda(ea_feat_l, ea_feat_m, flow_l)
            merged_feat = torch.cat((aligned_feat_s, ea_feat_m, aligned_feat_l), dim=1)

        # Attention module
        ldr_feat_s = self.ldr_feat_extract(ldr_short)
        ldr_feat_m = self.ldr_feat_extract(ldr_mid)
        ldr_feat_l = self.ldr_feat_extract(ldr_long)

        att_feat_s = self.att_short_mid(ldr_feat_s, ldr_feat_m)
        att_feat_l = self.att_short_mid(ldr_feat_l, ldr_feat_m)

        merged_feat = torch.cat((merged_feat, att_feat_s, ldr_feat_m, att_feat_l), dim=1)
        return self.merging(merged_feat, ea_feat_m)
