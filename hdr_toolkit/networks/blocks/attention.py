import torch
from torch import nn
import torch.nn.functional as F
import math


# Used in ADNet and AHDRNet, from https://github.com/liuzhen03/ADNet/blob/main/graphs/adnet.py
class SpatialAttention(nn.Module):

    def __init__(self, n_channels=64):
        super(SpatialAttention, self).__init__()
        self.att1 = nn.Conv2d(n_channels * 2, n_channels * 2, kernel_size=3, padding=1, bias=True)
        self.att2 = nn.Conv2d(n_channels * 2, n_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        f_cat = torch.cat((x1, x2), 1)
        att_map = torch.sigmoid(self.att2(self.relu(self.att1(f_cat))))
        return x1 * att_map


# from https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# reference https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
class ECALayer(nn.Module):
    """Constructs an ECA module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class BAPack(nn.Module):

    def __init__(self, n_channels, fca=False, bias=False, reduction=8, dct_size=56):
        super(BAPack, self).__init__()
        self.fca = fca
        self.dct_size = dct_size
        if fca:
            # according to the setting of FCANet c2wh = dict([(64,56), (128,28), (256,14), (512,7)])
            freq_indices_x, freq_indices_y = get_freq_indices('top16')
            freq_indices_x = [ix * (dct_size // 7) for ix in freq_indices_x]
            freq_indices_y = [iy * (dct_size // 7) for iy in freq_indices_y]
            self.feat_compress = MultiSpectralDCTLayer(dct_size, dct_size, freq_indices_x, freq_indices_y, n_channels)
        else:
            self.feat_compress = nn.AdaptiveAvgPool2d(1)
        reduced_channels = n_channels // reduction
        self.fc_x = nn.Sequential(
            nn.Linear(n_channels, reduced_channels, bias=bias),
            nn.BatchNorm1d(reduced_channels)
        )
        self.fc_extra_feat = nn.Sequential(
            nn.Linear(n_channels, reduced_channels, bias=bias),
            nn.BatchNorm1d(reduced_channels)
        )

        self.generation = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, n_channels, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x, extra_feat):
        n, c, _, _ = x.shape
        y = self.fc_x(self._compress_feat(x))
        y_extra_feat = self.fc_extra_feat(self._compress_feat(extra_feat))
        att = self.generation(y + y_extra_feat).view(n, c, 1, 1)
        return x * att

    def _compress_feat(self, x):
        n, c, h, w = x.shape
        if self.fca:
            # DCT
            x_pooled = x
            if h != self.dct_size or w != self.dct_size:
                x_pooled = F.adaptive_avg_pool2d(x, (self.dct_size, self.dct_size))
            return self.feat_compress(x_pooled)
        else:
            # GAP
            return self.feat_compress(x).view(n, c)

    @classmethod
    def create(cls, n_channels, ba_type):
        if ba_type == 'fca':
            return cls(n_channels, fca=True)
        elif ba_type == 'default':
            return cls(n_channels)
        elif ba_type == 'use_bias':
            return cls(n_channels, bias=True)
        elif ba_type == 'no_reduction':
            return cls(n_channels, reduction=1)
        else:
            raise KeyError(f'{ba_type} is invalid for Bridge Attention')


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2,
                             4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2, 6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2,
                             6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0, 5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0,
                             1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5,
                             4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5,
                             6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5, 3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1,
                             4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3, 3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


# From official code of FCA https://github.com/cfzd/FcaNet/blob/master/model/layer.py
class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = F.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    # noinspection PyMethodMayBeStatic, PyPep8Naming
    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = \
                        self.build_filter(t_x, u_x, tile_size_x) * self.build_filter(t_y, v_y, tile_size_y)

        return dct_filter
