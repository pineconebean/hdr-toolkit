import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d


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

    def __init__(self, n_channels=64, groups=8):
        super(PCDAlign, self).__init__()
        # L3: level 3, 1/4 spatial size
        self.l3_offset_conv1 = nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1, bias=True)  # concat for diff
        self.l3_offset_conv2 = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)
        self.l3_deform_conv = PCDDeformConv2d(n_channels, n_channels, 3, stride=1, padding=1, dilation=1, groups=groups)

        # L2: level 2, 1/2 spatial size
        self.l2_offset_conv1 = nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1, bias=True)  # concat for diff
        self.l2_offset_conv2 = nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1, bias=True)  # concat for offset
        self.l2_offset_conv3 = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)
        self.l2_deform_conv = PCDDeformConv2d(n_channels, n_channels, 3, stride=1, padding=1, dilation=1, groups=groups)

        self.l2_fea_conv = nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1, bias=True)  # concat for fea
        # L1: level 1, original spatial size
        self.l1_offset_conv1 = nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1, bias=True)  # concat for diff
        self.l1_offset_conv2 = nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1, bias=True)  # concat for offset
        self.l1_offset_conv3 = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)
        self.l1_deform_conv = PCDDeformConv2d(n_channels, n_channels, 3, stride=1, padding=1, dilation=1, groups=groups)

        self.l1_fea_conv = nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1, bias=True)  # concat for fea
        # Cascading DCN
        self.cas_offset_conv1 = nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1, bias=True)  # concat for diff
        self.cas_offset_conv2 = nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True)

        self.cas_deform_conv = PCDDeformConv2d(n_channels, n_channels, 3, stride=1, padding=1, dilation=1,
                                               groups=groups)

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


class SharedOffsetsPCD(nn.Module):

    def __init__(self, n_channels):
        super(SharedOffsetsPCD, self).__init__()

    def forward(self, non_ref_feat, ref_feat):
        pass
