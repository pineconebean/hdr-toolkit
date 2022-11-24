import os
import pathlib

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

import hdr_toolkit.data.data_io as io

BICUBIC = transforms.InterpolationMode.BICUBIC


class NTIREDataset(Dataset):

    def __init__(self, read_path, with_gt=True, two_level_dir=False):
        super(NTIREDataset, self).__init__()
        self.root = pathlib.Path(read_path)
        self.width = 256
        self.height = 256
        self.with_gt = with_gt
        self.suffix = 'gt.png' if with_gt else 'long.png'
        if two_level_dir:
            glob_pattern = f'*/*{self.suffix}'
        else:
            glob_pattern = f'*{self.suffix}'

        self.img_with_suffix = list(map(str, self.root.glob(glob_pattern)))
        if len(self.img_with_suffix) == 0:
            raise ValueError('Invalid suffix (0 data read)')

    def __len__(self):
        return len(self.img_with_suffix)

    def __getitem__(self, index):
        path_img_with_suffix = self.img_with_suffix[index]
        img_dir = os.path.dirname(path_img_with_suffix)

        img_id = os.path.basename(path_img_with_suffix).replace(f'_{self.suffix}', '').split('_')[0]
        exposures = np.load(os.path.join(img_dir, '{}_exposures.npy'.format(img_id)))

        img_short = read_ntire_ldr(path_img_with_suffix.replace(f'_{self.suffix}', '_short.png'), exposures[0])
        img_medium = read_ntire_ldr(path_img_with_suffix.replace(f'_{self.suffix}', '_medium.png'), exposures[1])
        img_long = read_ntire_ldr(path_img_with_suffix.replace(f'_{self.suffix}', '_long.png'), exposures[2])

        result = {
            'ldr_images': [img_short, img_medium, img_long],
            'img_id': int(img_id),
            'exposures': exposures
        }

        if self.with_gt:
            align_ratio_path = os.path.join(img_dir, '{}_alignratio.npy'.format(img_id))
            path_gt = path_img_with_suffix.replace(f'_{self.suffix}', '_gt.png')
            result['gt'] = F.to_tensor(io.imread_uint16_png(path_gt, align_ratio_path))
            # gt is non-linearized

        return result


def _gamma_correction(img, gamma, exposure):
    return (img ** gamma) * 2.0 ** (-1 * exposure)


def read_ntire_ldr(path, exposure, device='cuda:0'):
    img = (cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / 255.0).astype(np.float32)
    img = F.to_tensor(img).to(device)
    img_corrected = _gamma_correction(img, 2.24, exposure)
    return torch.cat((img, img_corrected), dim=0)


def read_exposures(file_path):
    with open(file_path) as f:
        return [float(line.strip()) for line in f.readlines()]
