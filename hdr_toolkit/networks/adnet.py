import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

from hdr_toolkit.networks.blocks.drdb import DRDB


# ref: https://github.com/liuzhen03/ADNet/blob/main/DCNv2/dcn_v2.py
class PCDDeformConv2d(nn.Module):
    """Use other features to generate offsets and masks"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1,
                 groups=1):
        super(PCDDeformConv2d, self).__init__()

        channels_ = groups * 3 * kernel_size * kernel_size
        self.conv_offset_mask = nn.Conv2d(in_channels, channels_, kernel_size=kernel_size,
                                          stride=stride, padding=padding, bias=True)
        self.init_offset()
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups)

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        self.conv_offset_mask.bias.data.zero_()

    def forward(self, in_fea, offset_fea):
        """
        input: input features for deformable conv
        fea: other features used for generating offsets and mask
        """
        out = self.conv_offset_mask(offset_fea)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)

        offset_mean = torch.mean(torch.abs(offset))
        if offset_mean > 100:
            print('Offset mean is {}, larger than 100.'.format(offset_mean))

        mask = torch.sigmoid(mask)
        return self.deform_conv(in_fea, offset=offset, mask=mask)


class PCDAlign(nn.Module):

    def __init__(self, nf=64, groups=8):
        super(PCDAlign, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.l3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.l3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.l3_deform_conv = PCDDeformConv2d(nf, nf, 3, stride=1, padding=1, dilation=1, groups=groups)
        # extra_offset_mask=True)
        # L2: level 2, 1/2 spatial size
        self.l2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.l2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.l2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.l2_deform_conv = PCDDeformConv2d(nf, nf, 3, stride=1, padding=1, dilation=1, groups=groups)
        # extra_offset_mask=True)
        self.l2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.l1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.l1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
        self.l1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.l1_deform_conv = PCDDeformConv2d(nf, nf, 3, stride=1, padding=1, dilation=1, groups=groups)
        # extra_offset_mask=True)
        self.l1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        self.cas_deform_conv = PCDDeformConv2d(nf, nf, 3, stride=1, padding=1, dilation=1, groups=groups)
        # extra_offset_mask=True)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        """
        align other neighboring frames to the reference frame in the feature level
        nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
        """
        # L3
        l3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
        l3_offset = self.leaky_relu(self.l3_offset_conv1(l3_offset))
        l3_offset = self.leaky_relu(self.l3_offset_conv2(l3_offset))
        l3_fea = self.leaky_relu(self.l3_deform_conv(nbr_fea_l[2], l3_offset))
        # L2
        l2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
        l2_offset = self.leaky_relu(self.l2_offset_conv1(l2_offset))
        l3_offset = F.interpolate(l3_offset, scale_factor=2, mode='bilinear', align_corners=False)
        l2_offset = self.leaky_relu(self.l2_offset_conv2(torch.cat([l2_offset, l3_offset * 2], dim=1)))
        l2_offset = self.leaky_relu(self.l2_offset_conv3(l2_offset))
        l2_fea = self.l2_deform_conv(nbr_fea_l[1], l2_offset)
        l3_fea = F.interpolate(l3_fea, scale_factor=2, mode='bilinear', align_corners=False)
        l2_fea = self.leaky_relu(self.l2_fea_conv(torch.cat([l2_fea, l3_fea], dim=1)))
        # L1
        l1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
        l1_offset = self.leaky_relu(self.l1_offset_conv1(l1_offset))
        l2_offset = F.interpolate(l2_offset, scale_factor=2, mode='bilinear', align_corners=False)
        l1_offset = self.leaky_relu(self.l1_offset_conv2(torch.cat([l1_offset, l2_offset * 2], dim=1)))
        l1_offset = self.leaky_relu(self.l1_offset_conv3(l1_offset))
        l1_fea = self.l1_deform_conv(nbr_fea_l[0], l1_offset)
        l2_fea = F.interpolate(l2_fea, scale_factor=2, mode='bilinear', align_corners=False)
        l1_fea = self.l1_fea_conv(torch.cat([l1_fea, l2_fea], dim=1))
        # Cascading
        offset = torch.cat([l1_fea, ref_fea_l[0]], dim=1)
        offset = self.leaky_relu(self.cas_offset_conv1(offset))
        offset = self.leaky_relu(self.cas_offset_conv2(offset))
        l1_fea = self.leaky_relu(self.cas_deform_conv(l1_fea, offset))

        return l1_fea


class ResidualBlockNoBN(nn.Module):

    def __init__(self, num_feat=64, res_scale=1):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Pyramid(nn.Module):
    def __init__(self, in_channels=6, n_feats=64):
        super(Pyramid, self).__init__()
        self.in_channels = in_channels
        self.n_feats = n_feats
        num_feat_extra = 1

        self.conv1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.n_feats, kernel_size=1, stride=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        layers = []
        for _ in range(num_feat_extra):
            layers.append(ResidualBlockNoBN())
        self.feature_extraction = nn.Sequential(*layers)

        self.down_sample1 = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.down_sample2 = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        x_in = self.conv1(x)
        x1 = self.feature_extraction(x_in)
        x2 = self.down_sample1(x1)
        x3 = self.down_sample2(x2)
        return [x1, x2, x3]


class SpatialAttentionModule(nn.Module):

    def __init__(self, n_feats):
        super(SpatialAttentionModule, self).__init__()
        self.att1 = nn.Conv2d(n_feats * 2, n_feats * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(n_feats * 2, n_feats, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return att_map


class ADNet(nn.Module):

    def __init__(self, n_channel, n_dense_layer, n_feat, growth_rate, align_version='v0', out_act='relu'):
        super(ADNet, self).__init__()
        self.n_channel = n_channel
        self.n_dense_layer = n_dense_layer
        self.n_feats = n_feat
        self.growth_rate = growth_rate
        self.align_version = align_version

        # PCD align module
        self.pyramid_feats = Pyramid(3)
        self.align_module = PCDAlign()

        # Spatial attention module
        self.att_module_l = SpatialAttentionModule(self.n_feats)
        self.att_module_h = SpatialAttentionModule(self.n_feats)

        # feature extraction
        self.feat_extract = nn.Sequential(
            nn.Conv2d(3, n_feat, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )

        # conv1
        self.conv1 = nn.Conv2d(self.n_feats * 6, self.n_feats, kernel_size=3, padding=1, bias=True)
        # 3 x DRDBs
        self.RDB1 = DRDB(self.n_feats, self.growth_rate, self.n_dense_layer)
        self.RDB2 = DRDB(self.n_feats, self.growth_rate, self.n_dense_layer)
        self.RDB3 = DRDB(self.n_feats, self.growth_rate, self.n_dense_layer)
        # conv2
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.n_feats * 3, self.n_feats, kernel_size=1, padding=0, bias=True),
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, padding=1, bias=True)
        )
        # post conv
        last_act = nn.ReLU(inplace=True) if out_act == 'relu' else nn.Sigmoid()
        self.post_conv = nn.Sequential(
            nn.Conv2d(self.n_feats, self.n_feats, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(self.n_feats, 3, kernel_size=3, padding=1, bias=True),
            last_act
        )
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x1, x2, x3):
        x1_t, x1_l = x1[:, 0:3, ...], x1[:, 3:, ...]
        x2_t, x2_l = x2[:, 0:3, ...], x2[:, 3:, ...]
        x3_t, x3_l = x3[:, 0:3, ...], x3[:, 3:, ...]
        # pyramid features of linear domain
        f1_l = self.pyramid_feats(x1_l)
        f2_l = self.pyramid_feats(x2_l)
        f3_l = self.pyramid_feats(x3_l)
        f2_ = f2_l[0]
        # PCD alignment
        f1_aligned_l = self.align_module(f1_l, f2_l)
        f3_aligned_l = self.align_module(f3_l, f2_l)
        # Spatial attention module
        f1_t = self.feat_extract(x1_t)
        f2_t = self.feat_extract(x2_t)
        f3_t = self.feat_extract(x3_t)
        f1_t_att = self.att_module_l(f1_t, f2_t)
        f1_t_ = f1_t * f1_t_att
        f3_t_att = self.att_module_h(f3_t, f2_t)
        f3_t_ = f3_t * f3_t_att

        # fusion subnet
        f_ = torch.cat((f1_aligned_l, f1_t_, f2_, f2_t, f3_aligned_l, f3_t_), 1)
        f_0 = self.conv1(f_)
        f_1 = self.RDB1(f_0)
        f_2 = self.RDB2(f_1)
        f_3 = self.RDB3(f_2)
        ff = torch.cat((f_1, f_2, f_3), 1)
        ff = self.conv2(ff)
        ff = ff + f2_
        res = self.post_conv(ff)
        return res
