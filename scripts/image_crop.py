import cv2
from pathlib import Path
import argparse


def crop_image(input_dir, output_dir, x, y, size):
    images = Path(input_dir).glob('*.tif')
    for curr in images:
        img = cv2.imread(str(curr), cv2.IMREAD_UNCHANGED)
        print(str(curr))
        cropped = img[x:x + size, y:y + size, :]
        cv2.imwrite(str(Path(output_dir).joinpath(curr.name)), cropped)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--read-path", dest='read_path')
    parser.add_argument("--write-path", dest='write_path')
    parser.add_argument('--x', type=int)
    parser.add_argument('--y', type=int)
    parser.add_argument('--size', type=int)

    args = parser.parse_args()

    crop_image(args.read_path, args.write_path, args.x, args.y, args.size)
