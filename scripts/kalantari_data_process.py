import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def reorganize_test_files(src, dst):
    r"""Rename the files of Kalantari test dataset and copy them into the assigned directory."""
    target_dir = Path(dst)
    target_dir.mkdir(exist_ok=True)
    for example in Path(src).glob('*/*'):
        assert example.is_dir()
        ldr, hdr, exposure = _read_example_files_path(example)
        example_id = example.stem.lower()
        target_dir.joinpath(example_id)
        shutil.copyfile(hdr, target_dir.joinpath(f'{example_id}_gt.hdr'))
        shutil.copyfile(exposure, target_dir.joinpath(f'{example_id}_exposure.txt'))
        shutil.copyfile(ldr[0], target_dir.joinpath(f'{example_id}_short.tif'))
        shutil.copyfile(ldr[1], target_dir.joinpath(f'{example_id}_medium.tif'))
        shutil.copyfile(ldr[2], target_dir.joinpath(f'{example_id}_long.tif'))


def crop_training_files(src, dst):
    r"""Crop the training examples and save them into the assigned directory."""
    target_dir = Path(dst)
    target_dir.mkdir(exist_ok=True)
    for e_cnt, example in enumerate(Path(src).glob('*')):
        # then this is a README.txt
        if not example.is_dir():
            continue

        example_id = example.stem
        ldr_path, hdr_path, exposure_path = _read_example_files_path(example)
        # read and write exposure file
        shutil.copyfile(exposure_path, target_dir.joinpath(f'{example_id}_exposure.txt'))
        # augment and crop the example, then save them
        ldr_data, hdr_data = _read_example_data(ldr_path, hdr_path)
        ldr_patches, hdr_patches = _augment_and_crop(ldr_data, hdr_data)
        for i in range(len(hdr_patches)):
            for j in range(len(hdr_patches[i])):
                count = i * len(hdr_patches[0]) + j
                cv2.imwrite(str(target_dir.joinpath(f'{example_id}_{count:04d}_gt.hdr')), hdr_patches[i][j])
                cv2.imwrite(str(target_dir.joinpath(f'{example_id}_{count:04d}_short.tif')), ldr_patches[i][0][j])
                cv2.imwrite(str(target_dir.joinpath(f'{example_id}_{count:04d}_medium.tif')), ldr_patches[i][1][j])
                cv2.imwrite(str(target_dir.joinpath(f'{example_id}_{count:04d}_long.tif')), ldr_patches[i][2][j])
        print(f'Progress: {len(hdr_patches) * len(hdr_patches[0]) * (e_cnt + 1)} / {61568}')


def prepare_validation_data(src, dst, size, include):
    r"""Randomly copy or link data from source directory as validation dataset (only for Kalantari dataset)"""
    target_dir = Path(dst)
    src_dir = Path(src)
    target_dir.mkdir(exist_ok=True)

    examples = _collect_file_names(src_dir)
    examples = examples.difference(set(include))
    chosen_examples = np.random.choice(list(examples), size, replace=False).tolist() + include
    print(chosen_examples)
    for e in chosen_examples:
        # copy exposure files
        exposure_path = src_dir.joinpath(f'{e}_exposure.txt')
        shutil.copyfile(exposure_path, target_dir.joinpath(exposure_path.name))

        # read and crop images to small size (so that not occupy too much space)
        ldr_path = [src_dir.joinpath(f'{e}_short.tif'),
                    src_dir.joinpath(f'{e}_medium.tif'),
                    src_dir.joinpath(f'{e}_long.tif')]
        hdr_path = src_dir.joinpath(f'{e}_gt.hdr')
        ldr_data, hdr_data = _read_example_data(ldr_path, hdr_path)

        def _crop_val_image(image):
            return _crop_an_image(image, crop_size=256, step=250, threshold=0)

        hdr_patches = _crop_val_image(hdr_data)
        ldr_patches = []
        for ldr_i in ldr_data:
            ldr_patches.append(_crop_val_image(ldr_i))

        for i in range(len(hdr_patches)):
            cv2.imwrite(str(target_dir.joinpath(f'{e}_{i:02d}_gt.hdr')), hdr_patches[i])
            cv2.imwrite(str(target_dir.joinpath(f'{e}_{i:02d}_short.tif')), ldr_patches[0][i])
            cv2.imwrite(str(target_dir.joinpath(f'{e}_{i:02d}_medium.tif')), ldr_patches[1][i])
            cv2.imwrite(str(target_dir.joinpath(f'{e}_{i:02d}_long.tif')), ldr_patches[2][i])


def _collect_file_names(src_dir: Path, suffix='_gt.hdr') -> set:
    all_files = src_dir.glob(f'*{suffix}')
    result = set()

    for file in all_files:
        result.add(file.name[:-len(suffix)])

    return result


def _augment_and_crop(ldr, hdr):
    # augment
    aug_ldr_s = _augment(ldr[0])
    aug_ldr_m = _augment(ldr[1])
    aug_ldr_l = _augment(ldr[2])
    aug_hdr = _augment(hdr)
    # crop
    ldr_patches = []
    hdr_patches = []
    for i in range(len(aug_hdr)):
        hdr_patches.append(_crop_an_image(aug_hdr[i]))
        ldr_patches.append([_crop_an_image(aug_ldr_s[i]), _crop_an_image(aug_ldr_m[i]), _crop_an_image(aug_ldr_l[i])])
    return ldr_patches, hdr_patches


def _crop_an_image(image, crop_size=256, step=105, threshold=40):
    h, w, _ = image.shape
    h_space = np.arange(0, h - crop_size + 1, step)
    w_space = np.arange(0, w - crop_size + 1, step)

    def _check_threshold(total_len, last_pos):
        return total_len - (last_pos + crop_size) > threshold

    if _check_threshold(h, h_space[-1]):
        h_space = np.append(h_space, h - crop_size)
    if _check_threshold(w, w_space[-1]):
        w_space = np.append(w_space, w - crop_size)

    result = []
    for y in h_space:
        for x in w_space:
            result.append(np.ascontiguousarray(image[y:y + crop_size, x:x + crop_size, :]))
    return result


def _augment(image):
    result = []
    for i in range(2):
        curr = cv2.transpose(image) if i == 1 else image
        result.append(curr)
        for j in range(3):
            result.append(cv2.flip(curr, j - 1))
    assert len(result) == 8
    return result


def _read_example_files_path(example_path):
    hdr = example_path.joinpath('HDRImg.hdr')
    exposure = example_path.joinpath('exposure.txt')
    ldr = sorted(list(example_path.glob('*.tif')))
    return ldr, hdr, exposure


def _read_example_data(ldr_path, hdr_path):
    ldr_s = cv2.imread(str(ldr_path[0]), cv2.IMREAD_UNCHANGED)
    ldr_m = cv2.imread(str(ldr_path[1]), cv2.IMREAD_UNCHANGED)
    ldr_l = cv2.imread(str(ldr_path[2]), cv2.IMREAD_UNCHANGED)
    hdr = cv2.imread(str(hdr_path), cv2.IMREAD_UNCHANGED)
    return [ldr_s, ldr_m, ldr_l], hdr


# used for testing the correctness of _augment
def _write_aug_result(aug_image, dst):
    for i in range(len(aug_image)):
        cv2.imwrite(str(dst.joinpath(f'{i:02d}.tif')), aug_image[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', dest='read_dir', required=True)
    parser.add_argument('--write', dest='write_dir', required=True)
    parser.add_argument('--op', dest='op', required=True, choices=['reorganize', 'crop', 'prepare_val'])
    parser.add_argument('--val-size', dest='val_size', default=6, type=int)
    parser.add_argument('--include', nargs='+', type=str)
    args = parser.parse_args()
    if args.op == 'crop':
        crop_training_files(args.read_dir, args.write_dir)
    elif args.op == 'prepare_val':
        prepare_validation_data(args.read_dir, args.write_dir, args.val_size, args.include)
    else:
        reorganize_test_files(args.read_dir, args.write_dir)
