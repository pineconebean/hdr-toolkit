from pathlib import Path

import cv2
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import Dataset

from hdr_toolkit.data.data_io import read_ldr


class KalantariDataset(Dataset):

    def __init__(self, img_dir, suffix='gt.hdr', **_):
        super(KalantariDataset, self).__init__()
        self.suffix = suffix
        self.ldr_images = list(map(str, Path(img_dir).glob(f'*{suffix}')))

    def __len__(self):
        return len(self.ldr_images)

    def __getitem__(self, idx):
        example_path = Path(self.ldr_images[idx])
        img_id = example_path.stem.split('_')[0]

        exposures = np.loadtxt(example_path.parent.joinpath(f'{img_id}_exposure.txt'))

        img_short = read_ldr(str(example_path).replace(self.suffix, 'short.tif'), exposures[0], 16)
        img_medium = read_ldr(str(example_path).replace(self.suffix, 'medium.tif'), exposures[1], 16)
        img_long = read_ldr(str(example_path).replace(self.suffix, 'long.tif'), exposures[2], 16)

        gt = F.to_tensor(cv2.cvtColor(cv2.imread(str(example_path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))

        return {
            'ldr_images': [img_short, img_medium, img_long],
            'gt': gt,
            'img_id': img_id  # used by writer to write the result when testing
        }
