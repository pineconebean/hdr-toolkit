import argparse
import pathlib
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import get_dataset
from data.im_write import KalantariWriter, NTIREWriter
from hdr_toolkit.hdr_ops.tonemap import tonemap
from hdr_toolkit.metrics.psnr import psnr
from hdr_toolkit.networks import get_model


def test(model_type, checkpoint_path, dataset, input_dir, out_dir, device, write_tonemap_gt, with_gt, act):
    out_dir = pathlib.Path(out_dir)
    model = get_model(model_type, out_activation=act)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)
    data_loader = DataLoader(get_dataset(dataset, input_dir, with_gt=with_gt), batch_size=1, shuffle=False)

    writer = NTIREWriter(out_dir) if dataset == 'ntire' else KalantariWriter(out_dir)

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

            print('elapsed time {}'.format(time.time() - start_time))
            if dataset == 'kalantari':
                gt = data['gt'].squeeze().to(device)
                mu_gt = tonemap(gt)
                print(f'psnr-l: {psnr(hdr_pred, gt)} | psnr-t: {psnr(mu_pred, mu_gt)}')

            img_id = data['img_id']
            if dataset == 'kalantari':
                img_id = img_id[0]
            elif dataset == 'ntire':
                img_id = img_id.detach().squeeze().cpu().numpy().astype(np.int32)
            else:
                raise ValueError('invalid dataset')
            writer.write_hdr(hdr_pred.permute(1, 2, 0).detach().cpu().numpy(), img_id)
            writer.write_tonemap(mu_pred.permute(1, 2, 0).detach().cpu().numpy(), img_id)
            if with_gt:
                gt = data['gt'].to(device)
                mu_gt = tonemap(gt, dataset=dataset)
                if write_tonemap_gt:
                    writer.write_mu_gt(mu_gt.squeeze().permute(1, 2, 0).detach().cpu().numpy(), img_id)


if __name__ == '__main__':
    model_choices = ('ahdr', 'adnet')
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', dest='model', choices=model_choices, required=True)
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
