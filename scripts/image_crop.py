import cv2
from pathlib import Path
import argparse


# Kalantari 009 y:1092 x:448 size: 75
# Kalantari 009 y:1019 x:466 size: 75
# Kalantari 010 y:779 x:535 size: 75
# Kalantari 010 y:473 x:649 size: 75
def crop_image(input_dir, output_dir, x, y, size):
    images = Path(input_dir).glob('*')
    for curr in images:
        if curr.suffix in ['.hdr', '.tif']:
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
