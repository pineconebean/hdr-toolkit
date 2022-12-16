import cv2
from pathlib import Path

import torch
import torchvision.transforms.functional as F
from hdr_toolkit.metrics.psnr import psnr, normalized_psnr, psnr_tanh_norm_mu_tonemap

hdr_file_path = r'/Users/jundaliao/Documents/HDR/Kalantari/Test2/009_gt.hdr'
target_dir = Path(r'/Users/jundaliao/Documents/HDR/Kalantari/test_cv2')
target_dir.mkdir(exist_ok=True)
target_file_path = target_dir.joinpath('result.hdr')

# test whether cv2 read and write will change the data
read_img = cv2.imread(hdr_file_path, cv2.IMREAD_UNCHANGED)
cv2.imwrite(str(target_file_path), read_img)
re_read_img = cv2.imread(str(target_file_path), cv2.IMREAD_UNCHANGED)
# noinspection PyUnresolvedReferences
print(f'Test 1: {(read_img == re_read_img).all()}')

# test whether torch to_tensor will change the data
tensor_read_img = F.to_tensor(read_img)
tensor_file_path = target_dir.joinpath('tensor_result.hdr')
cv2.imwrite(str(tensor_file_path), tensor_read_img.permute(1, 2, 0).detach().cpu().numpy())
re_tensor_img = cv2.imread(str(tensor_file_path), cv2.IMREAD_UNCHANGED)
# noinspection PyUnresolvedReferences
print(f'Test 2: {(read_img == re_tensor_img).all()}')

# test whether change color channels will change the data
tensor_rgb_read_img = F.to_tensor(cv2.cvtColor(read_img, cv2.COLOR_BGR2RGB))
tensor_from_numpy = torch.from_numpy(read_img)
tensor_rgb_file_path = target_dir.joinpath('tensor_bgr_result.hdr')
print(tensor_rgb_read_img.shape)
cv2.imwrite(str(tensor_rgb_file_path),
            cv2.cvtColor(tensor_rgb_read_img.permute(1, 2, 0).cpu().numpy(), cv2.COLOR_RGB2BGR))
re_tensor_rgb_img = cv2.imread(str(tensor_rgb_file_path), cv2.IMREAD_UNCHANGED)
# noinspection PyUnresolvedReferences
print(f'Test 3: {(read_img == re_tensor_rgb_img).all()}')
print(f'Test 4: {(read_img == tensor_from_numpy.numpy()).all()}')

# test psnr result computed for (3, 1000, 1500) and (1000, 1500, 3)
ref = r'/Users/jundaliao/Documents/HDR/Kalantari/Test2/009_gt.hdr'
res = r'/Users/jundaliao/Documents/HDR/Kalantari/result/ahdr/009.hdr'
ref_hdr_image = torch.from_numpy(cv2.cvtColor(cv2.imread(ref, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))
res_hdr_image = torch.from_numpy(cv2.cvtColor(cv2.imread(res, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))

score_linear1 = psnr(ref_hdr_image, res_hdr_image)

ref_hdr_image = F.to_tensor(cv2.cvtColor(cv2.imread(ref, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))
res_hdr_image = F.to_tensor(cv2.cvtColor(cv2.imread(res, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))
score_linear2 = psnr(ref_hdr_image, res_hdr_image)
print(f'Test 5: {score_linear1 == score_linear2}')

# Test whether dataloader will change the data of groundtruth
gt = cv2.imread(r'/Users/jundaliao/Documents/HDR/Kalantari/Test2/001_gt.hdr', cv2.IMREAD_UNCHANGED)
loader_gt = cv2.imread(r'/Users/jundaliao/Documents/HDR/Kalantari/test_cv2/001_gt.hdr', cv2.IMREAD_UNCHANGED)
print(f'Test6: {(gt == loader_gt).all()}')
