import argparse
import pathlib
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import get_dataset
from data.writers import KalantariWriter, NTIREWriter
from hdr_toolkit.hdr_ops.tonemap import tonemap, tanh_norm_mu_tonemap
from hdr_toolkit.metrics.psnr import psnr
from hdr_toolkit.networks import get_model
from hdr_toolkit.util.logging import get_logger


def test(model_type, ckpt_dir, dataset, input_dir, out_dir, device, write_tonemap_gt, with_gt, act):
    out_dir = pathlib.Path(out_dir)
    ckpt_dir = pathlib.Path(ckpt_dir)
    ckpt_files = ['checkpoint.pth', 'val-t-checkpoint.pth', 'val-l-checkpoint.pth']
    out_dir_names = ['last', 'val-t', 'val-l']

    for curr_file, out_dir_name in zip(ckpt_files, out_dir_names):
        ckpt = torch.load(str(ckpt_dir.joinpath(curr_file)))
        model = get_model(model_type, out_activation=act)
        model.load_state_dict(ckpt['model'])
        model.eval()
        model.to(device)
        data_loader = DataLoader(get_dataset(dataset, input_dir, with_gt=with_gt), batch_size=1, shuffle=False)

        curr_out_dir = out_dir.joinpath(out_dir_name)
        writer = NTIREWriter(curr_out_dir) if dataset == 'ntire' else KalantariWriter(curr_out_dir)
        logger = get_logger(out_dir_name, str(curr_out_dir.joinpath('test.log')))
        scores_linear = []
        scores_tonemap = []
        with torch.no_grad():
            for batch, data in enumerate(data_loader):
                start_time = time.time()
                ldr_images = data['ldr_images']
                low = ldr_images[0].to(device)
                ref = ldr_images[1].to(device)
                high = ldr_images[2].to(device)
                hdr_pred = model(low, ref, high)
                hdr_pred = hdr_pred.squeeze()
                mu_pred = tonemap(hdr_pred, dataset=dataset)

                logger.info('elapsed time {}'.format(time.time() - start_time))
                if dataset == 'kalantari':
                    gt = data['gt'].squeeze().to(device)
                    mu_gt = tonemap(gt)
                    # psnr_l, psnr_t = psnr(hdr_pred, gt).cpu().numpy(), psnr(mu_pred, mu_gt).cpu().numpy()
                    # calculation on GPU and on CPU may have difference
                    psnr_l, psnr_t = \
                        psnr(hdr_pred.cpu().numpy(), gt.cpu().numpy(), backend='np'), \
                        psnr(mu_pred.cpu().numpy(), mu_gt.cpu().numpy(), backend='np')
                    logger.info(f'psnr-l: {psnr_l} | psnr-t: {psnr_t}')
                    scores_linear.append(psnr_l)
                    scores_tonemap.append(psnr_t)
                    norm = np.percentile(hdr_pred.cpu().numpy().astype(np.float32), 99)
                    mu_pred_to_write = tanh_norm_mu_tonemap(hdr_pred, norm)

                img_id = data['img_id']
                if dataset == 'kalantari':
                    img_id = img_id[0]
                elif dataset == 'ntire':
                    img_id = img_id.squeeze().cpu().numpy().astype(np.int32)
                else:
                    raise ValueError('invalid dataset')
                writer.write_hdr(hdr_pred.permute(1, 2, 0).cpu().numpy(), img_id)
                writer.write_tonemap(mu_pred_to_write.permute(1, 2, 0).cpu().numpy(), img_id)
                if with_gt:
                    gt = data['gt'].to(device)
                    mu_gt = tonemap(gt, dataset=dataset)
                    if write_tonemap_gt:
                        writer.write_mu_gt(mu_gt.squeeze().permute(1, 2, 0).detach().cpu().numpy(), img_id)
        logger.info(f'avg psnr-t: {np.mean(scores_tonemap)}')
        logger.info(f'avg psnr-l: {np.mean(scores_linear)}')


if __name__ == '__main__':
    # model_choices = ('ahdr', 'adnet', 'ecadnet-gc6', 'ecadnet', 'psftd', 'bahdr-var1')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', dest='model', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data', choices=['ntire', 'kalantari'], default='ntire')
    parser.add_argument('--data-with-gt', dest='with_gt', action='store_true')
    parser.add_argument('--input-dir', dest='input_dir', required=True)
    parser.add_argument('--output-dir', dest='out_dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--write-tonemap-gt', dest='t_gt', action='store_true')
    parser.add_argument('--out-activation', dest='act', choices=['relu', 'sigmoid'], required=True)
    args = parser.parse_args()

    test(args.model, args.checkpoint, args.data, args.input_dir, args.out_dir, args.device, args.t_gt, args.with_gt,
         args.act)
