import numpy as np
import torch

from hdr_toolkit.hdr_ops.tonemap import tanh_norm_mu_tonemap


def psnr(im0, im1, norm=1):
    return -10 * torch.log10(torch.mean(torch.pow(im0 / norm - im1 / norm, 2)))


def normalized_psnr(im0, im1, norm):
    return psnr(im0 / norm, im1 / norm)


def psnr_tanh_norm_mu_tonemap(hdr_nonlinear_ref, hdr_nonlinear_res, percentile=99, gamma=2.24):
    """ This function computes Peak Signal to Noise Ratio (PSNR) between the mu-law computed images from two non-linear
    HDR images.

            Args:
                hdr_nonlinear_ref (np.ndarray): HDR Reference Image after gamma correction, used for the percentile norm
                hdr_nonlinear_res (np.ndarray: HDR Estimated Image after gamma correction
                percentile (float): Percentile use for normalization
                gamma (float): Value used to linearize the non-linear images

            Returns:
                np.ndarray (): Returns the mean mu-law PSNR value for the complete image.

            """
    hdr_linear_ref = hdr_nonlinear_ref ** gamma
    hdr_linear_res = hdr_nonlinear_res ** gamma
    norm_perc = np.percentile(hdr_linear_ref, percentile)
    return psnr(tanh_norm_mu_tonemap(hdr_linear_ref, norm_perc), tanh_norm_mu_tonemap(hdr_linear_res, norm_perc))
