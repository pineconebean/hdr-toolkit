import torch
from torch import cat
from torch import nn

from hdr_toolkit.networks.blocks import SELayer, ECALayer, PCDAlign, SpatialAttention, PyramidFeature, \
    AHDRMergingNet


class SEBlock(nn.Module):

    def __init__(self, n_feats=64, deform_groups=8):
        super(SEBlock, self).__init__()
        self.deform_groups = deform_groups  # not used now
        self.trans_conv = nn.ModuleList()
        self.select_fea = SELayer(n_feats * 6)
        for _ in range(6):
            self.trans_conv.append(nn.Conv2d(n_feats, n_feats, 1))

    def forward(self, att_fea, aligned_fea):
        # transform
        att1 = self.trans_conv[0](att_fea[0])
        att2 = self.trans_conv[1](att_fea[1])
        att3 = self.trans_conv[2](att_fea[2])
        aligned1 = self.trans_conv[3](aligned_fea[0])
        aligned2 = self.trans_conv[4](aligned_fea[1])
        aligned3 = self.trans_conv[5](aligned_fea[2])

        # squeeze and excitation
        f = cat((att1, att2, att3, aligned1, aligned2, aligned3), dim=1)
        return self.select_fea(f)


class ECADNet(nn.Module):

    def __init__(self, n_channels, trans_conv_groups=6, out_activation='relu'):
        super(ECADNet, self).__init__()
        self.trans_conv_groups = trans_conv_groups
        # PCD alignment module
        self.pyramid_feats = PyramidFeature(in_channels=3, n_channels=n_channels)
        self.align_module = PCDAlign(n_channels)

        # convolution layer for extracting features from images
        self.extract_feature = nn.Conv2d(6, n_channels, 3, padding='same')

        # Spatial attention module
        self.att_short_mid = SpatialAttention(n_channels)
        self.att_long_mid = SpatialAttention(n_channels)

        # ECA layer for select features
        self.trans_conv = nn.Conv2d(n_channels * 6, n_channels * 6, 3, padding='same', groups=trans_conv_groups)
        self.eca = ECALayer(k_size=5)  # k_size = |log2(64 * 6) / 2 + 0.5|_odd = 5

        self.merging = AHDRMergingNet(n_channels * 6, n_channels, out_activation)
        self.relu = nn.LeakyReLU(inplace=True)

    def _channel_shuffle(self, x: torch.Tensor):
        b, c, h, w = x.shape
        x = x.view(b, self.trans_conv_groups, c // self.trans_conv_groups, h, w)
        x = torch.transpose(x, 1, 2)
        x = x.view(b, -1, h, w)
        return x

    def forward(self, short, mid, long):
        # ldr image 0:3; hdr domain image 3:6; exposure aligned image 6: 9
        h_short = short[:, 3:6, ...]
        h_mid = mid[:, 3:6, ...]
        h_long = long[:, 3:6, ...]

        # extract pyramid features
        py_feat_s = self.pyramid_feats(h_short)
        py_feat_m = self.pyramid_feats(h_mid)
        py_feat_l = self.pyramid_feats(h_long)
        feat_mid = py_feat_m[0]

        # PCD alignment
        aligned_s = self.align_module(py_feat_s, py_feat_m)
        aligned_l = self.align_module(py_feat_l, py_feat_m)

        # Spatial attention
        att_feat_s = self.relu(self.extract_feature(short))
        att_feat_m = self.relu(self.extract_feature(mid))
        att_feat_l = self.relu(self.extract_feature(long))
        att_refined_s = self.att_short_mid(att_feat_s, att_feat_m)
        att_refined_l = self.att_short_mid(att_feat_l, att_feat_m)

        # Channel attention with ECA
        x = self.trans_conv(cat((aligned_s, feat_mid, aligned_l, att_refined_s, att_feat_m, att_refined_l), dim=1))
        if self.trans_conv_groups > 1:
            x = self._channel_shuffle(x)
        x = self.eca(x)
        return self.merging(x, att_feat_m)
