import abc
import pathlib
from abc import ABC

import cv2
import numpy as np

from hdr_toolkit.data.data_io import imwrite_uint16_png


class Writer(ABC):

    def __init__(self, out_dir: pathlib.Path):
        super(Writer, self).__init__()
        self.out_dir = out_dir
        self.tone_map_dir = out_dir.joinpath('tonemapped')
        self.tone_map_dir.mkdir(parents=True, exist_ok=True)
        self.gt_dir = out_dir.joinpath('gt')
        self.gt_dir.mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def write_hdr(self, image, image_id):
        """Write linear domain hdr image"""

    @abc.abstractmethod
    def write_tonemap(self, image, image_id):
        """Write tonemapped domain image"""

    @abc.abstractmethod
    def write_mu_gt(self, image, image_id):
        """Write tonemapped domain ground truth image"""


class NTIREWriter(Writer):

    def write_hdr(self, image, image_id):
        image_path = str(self.out_dir.joinpath('{:04d}.png'.format(image_id)))
        alignratio_path = str(self.out_dir.joinpath('{:04d}_alignratio.npy'.format(image_id)))

        imwrite_uint16_png(image_path, image, alignratio_path)

    def write_tonemap(self, image, image_id):
        self._write_tonemap(image, str(self.tone_map_dir.joinpath(f'{image_id:04d}.png')))

    def write_mu_gt(self, image, image_id):
        self._write_tonemap(image, str(self.gt_dir.joinpath(f'{image_id:04d}_gt.png')))

    @staticmethod
    def _write_tonemap(image, output_path):
        if np.max(image) <= 1.:
            image = (image * 255).round().astype(np.uint8)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


class KalantariWriter(Writer):

    def write_hdr(self, image, image_id):
        img_path = str(self.out_dir.joinpath(f'{image_id}.hdr'))
        cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def write_tonemap(self, image, image_id):
        KalantariWriter._write_tonemap(image, str(self.tone_map_dir.joinpath(f'{image_id}.tif')))

    def write_mu_gt(self, image, image_id):
        KalantariWriter._write_tonemap(image, str(self.gt_dir.joinpath(f'{image_id}_gt.tif')))

    @staticmethod
    def _write_tonemap(image, output_path):
        image = (image * 65535).round().astype(np.uint16)
        cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
