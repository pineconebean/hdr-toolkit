import numpy as np
import torch


def tonemap(img, mu=5000., dataset='kalantari', gamma=2.24, percentile=99, backend='torch'):
    if dataset == 'kalantari':
        if backend == 'torch':
            return torch.log(mu * img + 1) / np.log(1. + mu)
        elif backend == 'np':
            return np.log(mu * img + 1) / np.log(1. + mu)
        else:
            raise ValueError(f'Invalid backend {backend}')
    elif dataset == 'ntire':
        linear_img = img ** gamma
        norm_value = np.percentile(linear_img.data.cpu().numpy().astype(np.float32), percentile)
        result = tanh_norm_mu_tonemap(linear_img, norm_value)
        if result.isnan().any():
            raise ValueError(f'Nan norm value: {norm_value}')
            # print(f'Nan norm value: {norm_value}')
            # print(f'Non zero value in img: {np.count_nonzero(cpu_img)} / {np.size(cpu_img)}')
            # cpu_img = linear_img.data.cpu().numpy().astype(np.float32)
        return result
    else:
        raise ValueError('Invalid dataset type')


def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    """ This function normalizes the input HDR linear image by the specified norm_value, afterwards bounds the
    HDR image values by applying a tanh function and afterwards computes the mu-law tonemapped image.

        the mu-law tonemapped image.
        Args:
            hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
            norm_value (float): Value for the normalization (i.e. hdr_image/norm_value)
            mu (float): Parameter controlling the compression performed during tone mapping.

        Returns:
            np.ndarray (): Returns the mu-law tonemapped image.

        """
    bounded_hdr = torch.tanh(hdr_image / norm_value)
    return tonemap(bounded_hdr, mu)


def norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    return tonemap(hdr_image / norm_value, mu)
