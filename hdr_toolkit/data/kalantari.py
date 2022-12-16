from pathlib import Path

import cv2
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torch import cat

from hdr_toolkit.data.data_io import read_ldr, gamma_correction, ev_align


class KalantariDataset(Dataset):

    def __init__(self, img_dir, hdr_domain=True, exposure_aligned=False, suffix='gt.hdr', **_):
        super(KalantariDataset, self).__init__()
        self.suffix = suffix
        self.gt_images = list(map(str, Path(img_dir).glob(f'*{suffix}')))
        self.exposure_aligned = exposure_aligned
        self.hdr_domain = hdr_domain

    def __len__(self):
        return len(self.gt_images)

    def __getitem__(self, idx):
        example_path = Path(self.gt_images[idx])
        img_id = example_path.stem.split('_')[0]

        exposures = np.loadtxt(example_path.parent.joinpath(f'{img_id}_exposure.txt'))

        img_short = read_ldr(str(example_path).replace(self.suffix, 'short.tif'), 16)
        img_medium = read_ldr(str(example_path).replace(self.suffix, 'medium.tif'), 16)
        img_long = read_ldr(str(example_path).replace(self.suffix, 'long.tif'), 16)

        gt = F.to_tensor(cv2.cvtColor(cv2.imread(str(example_path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))

        if self.hdr_domain:
            img_short = cat((img_short, gamma_correction(img_short, exposures[0], 2.2)), dim=0)
            img_medium = cat((img_medium, gamma_correction(img_medium, exposures[1], 2.2)), dim=0)
            img_long = cat((img_long, gamma_correction(img_long, exposures[2], 2.2)), dim=0)
        if self.exposure_aligned:
            img_short = cat((img_short, ev_align(img_short, exposures[0], 2.2)), dim=0)
            img_medium = cat((img_medium, ev_align(img_medium, exposures[1], 2.2)), dim=0)
            img_long = cat((img_long, ev_align(img_long, exposures[2], 2.2)), dim=0)

        return {
            'ldr_images': [img_short, img_medium, img_long],
            'gt': gt,
            'img_id': img_id  # used by writer to write the result when testing
        }

