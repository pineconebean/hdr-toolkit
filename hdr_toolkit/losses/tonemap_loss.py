import numpy as np
import torch.nn.functional as F

from hdr_toolkit.hdr_ops.tonemap import tanh_norm_mu_tonemap


# reference: https://github.com/liuzhen03/ADNet/blob/main/graphs/loss/muloss.py
class NtireMuLoss:

    def __init__(self, loss_func='mse', gamma=2.24, percentile=99):
        super(NtireMuLoss, self).__init__()
        self.gamma = gamma
        self.percentile = percentile
        if loss_func == 'mse':
            self.loss_func = F.mse_loss
        elif loss_func == 'l1':
            self.loss_func = F.l1_loss
        else:
            raise ValueError('Unexpected loss function')

    def __call__(self, pred, gt):
        # the output of the model (i.e. pred) and the ground truth HDR image are non-linear images,
        # so they need to be linearized to perform tone-map
        hdr_linear_pred = pred ** self.gamma
        hdr_linear_gt = gt ** self.gamma
        norm_value = np.percentile(hdr_linear_gt.data.cpu().numpy().astype(np.float32), self.percentile)

        mu_pred = tanh_norm_mu_tonemap(hdr_linear_pred, norm_value)
        mu_gt = tanh_norm_mu_tonemap(hdr_linear_gt, norm_value)
        if hdr_linear_pred.isnan().any():
            raise ValueError(f'linear pred nan')
        elif mu_pred.isnan().any():
            raise ValueError(f'mu pred nan, norm value is {norm_value}')
        return self.loss_func(mu_pred, mu_gt)
