import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import DeformConv2d


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

    def __init__(self, n_channels, groups=8, extra_mask=True):
        super(SharedOffsetsPCD, self).__init__()
        self.n_levels = 3
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.offset_conv_first = nn.ModuleList()
        self.offset_conv_last = nn.ModuleList()
        self.offset_conv_concat = nn.ModuleList()
        self.so_deform_conv = nn.ModuleList()
        self.feat_concat_conv = nn.ModuleList()
        for i in range(self.n_levels):
            self.offset_conv_first.append(nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1))
            self.offset_conv_last.append(nn.Conv2d(n_channels, n_channels, 3, 1, 1))
            if i != self.n_levels - 1:
                self.offset_conv_concat.append(nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1))
                self.feat_concat_conv.append(nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1))
            self.so_deform_conv.append(
                SharedOffsetsDeformConv2d(n_channels, n_channels, 3, 1, 1, 1, groups, extra_mask))

        # cascaded deformable convolution layer
        self.cas_offset_conv = nn.Sequential(
            nn.Conv2d(n_channels * 2, n_channels, 3, 1, 1, bias=True),
            self.leaky_relu,
            nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True),
            self.leaky_relu
        )
        self.cas_deform_conv = PCDDeformConv2d(n_channels, n_channels, 3, 1, 1, 1, groups)
        self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, non_ref_feat, ref_feat, share_with_feat):
        deform_share_pyramid = []
        deform_non_ref_prev, offset_feat_prev = None, None
        for i in reversed(range(self.n_levels)):
            # generate offset features for current level
            offset_feat_curr = self.leaky_relu(self.offset_conv_first[i](torch.cat((non_ref_feat[i], ref_feat[i]), dim=1)))
            if i < self.n_levels - 1:
                offset_feat_curr = torch.cat((offset_feat_curr, self.up_sample(offset_feat_prev)), dim=1)
                offset_feat_curr = self.leaky_relu(self.offset_conv_concat[i](offset_feat_curr))
            offset_feat_curr = self.leaky_relu(self.offset_conv_last[i](offset_feat_curr))

            # perform the deformable convolution
            deform_non_ref_curr, deform_share = self.so_deform_conv[i](non_ref_feat[i], offset_feat_curr,
                                                                       share_with_feat[i])
            deform_share_pyramid.append(deform_share)

            if i < self.n_levels - 1:
                # collapse one level of the pyramid
                deform_non_ref_curr = torch.cat((deform_non_ref_curr, self.up_sample(deform_non_ref_prev)), dim=1)
                deform_non_ref_curr = self.feat_concat_conv[i](deform_non_ref_curr)

            if i > 0:
                # the first level does not apply leaky_relu for the output according to EDVR and ADNet
                deform_non_ref_curr = self.leaky_relu(deform_non_ref_curr)

            deform_non_ref_prev = deform_non_ref_curr
            offset_feat_prev = offset_feat_curr

        # cascaded deformable convolution after collapsing the whole pyramid
        cas_offset_feat = self.cas_offset_conv(torch.cat((deform_non_ref_prev, ref_feat[0]), dim=1))
        cas_deform_non_ref = self.leaky_relu(self.cas_deform_conv(deform_non_ref_prev, cas_offset_feat))
        deform_share_pyramid.reverse()
        return cas_deform_non_ref, deform_share_pyramid


class SharedOffsetsDeformConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1,
                 groups=1, extra_mask=True):
        super(SharedOffsetsDeformConv2d, self).__init__()

        self.extra_mask = extra_mask
        mask_channels = groups * kernel_size * kernel_size
        offset_channels = mask_channels * 2

        self.conv_offset_mask = nn.Conv2d(in_channels, mask_channels + offset_channels, 3, 1, 1)
        if extra_mask:
            self.conv_extra_mask1 = nn.Conv2d(in_channels * 2, in_channels, 3, 1, 1)
            self.conv_extra_mask2 = nn.Conv2d(in_channels, mask_channels, 3, 1, 1)
        self.init_offset()

        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, share_with_feat, offset_feat):
        """
        Args:
            x: input features from which the offsets are learned
            share_with_feat: the features want to use the offsets to do deformable convolution directly
            offset_feat: the features used to generate offsets and masks
        """
        o1, o2, mask_x = torch.chunk(self.conv_offset_mask(offset_feat), 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask_x = torch.sigmoid(mask_x)
        x_deform = self.deform_conv(x, offset=offset, mask=mask_x)

        if self.extra_mask:
            extra_mask = self.leaky_relu(self.conv_extra_mask1(torch.cat((share_with_feat, offset_feat), dim=1)))
            extra_mask = torch.sigmoid(self.conv_extra_mask2(extra_mask))
            return x_deform, self.deform_conv(share_with_feat, offset=offset, mask=extra_mask)
        else:
            return x_deform, self.deform_conv(share_with_feat, offset=offset, mask=mask_x)

    def init_offset(self):
        def _init_to_zero(x):
            x.weight.data.zero_()
            x.bias.data.zero_()

        _init_to_zero(self.conv_offset_mask)
        _init_to_zero(self.conv_extra_mask2)


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
