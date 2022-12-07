import torch
import torch.nn as nn
from torch import cat
from hdr_toolkit.networks.blocks import (HomoPyramidFeature, AHDRMergingNet, PyramidSFT, PCDAlign,
                                         HeteroPyramidFeature, NaivePyramidSFT, SharedOffsetsPCD)


class PSFTDNet(nn.Module):

    def __init__(self,
                 in_channels=3,
                 n_channels=64,
                 out_activation='sigmoid',
                 share_offsets=False,
                 extract_same_feat=False,
                 same_conv_for_pyramid=True,
                 naive_pyramid=True,
                 simple_sft=True):
        super(PSFTDNet, self).__init__()

        self.extract_same_feat = extract_same_feat
        self.share_offsets = share_offsets

        # first convolution layer for extracting initial features
        self.first_conv = nn.Conv2d(in_channels, n_channels, 3, 1, 1)
        if not extract_same_feat:
            self.first_conv = nn.ModuleList((self.first_conv, nn.Conv2d(in_channels, n_channels, 3, 1, 1)))

        # modules for extracting pyramid features
        self.pyramid_feat_extract = nn.ModuleList()
        if same_conv_for_pyramid:
            self.pyramid_feat_extract = HomoPyramidFeature(n_channels)
            if not extract_same_feat:
                self.pyramid_feat_extract = nn.ModuleList([self.pyramid_feat_extract, HomoPyramidFeature(n_channels)])
        else:
            self.pyramid_feat_extract = HeteroPyramidFeature(n_channels)
            if not extract_same_feat:
                self.pyramid_feat_extract = nn.ModuleList([self.pyramid_feat_extract, HeteroPyramidFeature(n_channels)])

        # align module
        if share_offsets:
            self.align_module = SharedOffsetsPCD(n_channels)
        else:
            self.align_module = PCDAlign(n_channels)

        # pyramid SFT module
        if naive_pyramid:
            self.psft = NaivePyramidSFT(n_channels, simple_sft=simple_sft)
        else:
            self.psft = PyramidSFT(n_channels, simple_sft=simple_sft)

        # merging network
        self.merging = AHDRMergingNet(n_channels * 6, n_channels, out_activation)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, short, mid, long):
        features = self._generate_feature(short, mid, long)
        feat_to_sft_s, feat_to_align_s = features[0]
        feat_to_sft_m, feat_to_align_m = features[1]
        feat_to_sft_l, feat_to_align_l = features[2]

        # align the features with PCD
        # todo: add shared offsets here
        aligned_feat_s = self.align_module(feat_to_align_s, feat_to_align_m)
        aligned_feat_l = self.align_module(feat_to_align_l, feat_to_align_m)
        aligned_feat_m = feat_to_sft_m[0]

        # refine the features with SFT
        sft_feat_s = self.psft(feat_to_sft_s, feat_to_sft_m)
        sft_feat_l = self.psft(feat_to_sft_l, feat_to_sft_m)
        sft_feat_m = feat_to_sft_m[0]

        cat_feat = cat((aligned_feat_s, aligned_feat_m, aligned_feat_l,
                        sft_feat_s, sft_feat_m, sft_feat_l), dim=1)
        self.merging(cat_feat, aligned_feat_m)

    def _generate_feature(self, short, mid, long):
        def _generate_same(x):
            x1 = self.leaky_relu(self.first_conv(x[:, :3, ...]))
            x2 = self.leaky_relu(self.first_conv(x[:, 3:6, ...]))
            return self.pyramid_feat_extract(x1), self.pyramid_feat_extract(x2)

        def _generate_different(x):
            x1 = self.leaky_relu(self.first_conv[0](x[:, :3, ...]))
            x2 = self.leaky_relu(self.first_conv[1](x[:, 3:6, ...]))
            return self.pyramid_feat_extract[0](x1), self.pyramid_feat_extract[1](x2)

        generate = _generate_same if self.extract_same_feat else _generate_different
        return generate(short), generate(mid), generate(long)
