import argparse
import pathlib

import torch
from torch.utils.data import DataLoader

from data.kalantari import KalantariDataset
from data.ntire import NTIREDataset
from hdr_toolkit.hdr_ops.tonemap import tonemap
from hdr_toolkit.losses.tonemap_loss import NtireMuLoss
from hdr_toolkit.metrics.psnr import psnr
from hdr_toolkit.networks import get_model
from hdr_toolkit.util.logging import get_logger


def _get_data_set(name, path, batch_size=4, two_level_dir=False):
    if name == 'kalantari':
        return DataLoader(KalantariDataset(path), batch_size=batch_size, shuffle=True)
    elif name == 'ntire':
        return DataLoader(NTIREDataset(path, two_level_dir=two_level_dir), batch_size=batch_size, shuffle=True)
    else:
        raise ValueError('Unexpected dataset')


def _save_model(model, optimizer, epoch, save_path):
    if isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    things_to_save = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }

    torch.save(things_to_save, save_path)


def train(model, epochs, batch_size, data_path, dataset, checkpoint_path, log_path, logger_name,
          learning_rate=1e-4, loss_type='mse', two_level_dir=False, use_cpu=False):
    logger = get_logger(logger_name, log_path)
    logger.info(f'{"=" * 20}Start Training{"=" * 20}\n')
    data = _get_data_set(dataset, data_path, batch_size=batch_size, two_level_dir=two_level_dir)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trained_epochs = 0
    if use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device("cuda:0")
    model.to(device)
    torch.autograd.set_detect_anomaly(True)

    # load checkpoint
    saved_weights = pathlib.Path(checkpoint_path)
    if saved_weights.exists():
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        trained_epochs = checkpoint['epoch']
        model.train()

    if dataset == 'kalantari':
        loss_func = torch.nn.L1Loss()
        for epoch in range(epochs):
            total_epochs = trained_epochs + epoch + 1
            logger.info(f'{"=" * 10} Epoch: {total_epochs} {"=" * 10}')
            for batch, d in enumerate(data):
                ldr_images = d['ldr_images']
                gt = d['gt']
                low = ldr_images[0].to(device)
                ref = ldr_images[1].to(device)
                high = ldr_images[2].to(device)
                gt = gt.to(device)

                hdr_pred = model(low, ref, high)

                loss = loss_func(tonemap(hdr_pred, dataset=dataset), tonemap(gt, dataset=dataset))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logger.info(f'Epoch: {total_epochs} |Batch: {batch} --- Loss: {loss:.8f},'
                            f' PSNR-L: {psnr(hdr_pred, gt):.4f} | PSNR-T: {psnr(tonemap(hdr_pred), tonemap(gt)):.4f}')
            _save_model(model, optimizer, total_epochs, checkpoint_path)

    elif dataset == 'ntire':
        loss_func = NtireMuLoss(loss_func=loss_type)
        for epoch in range(epochs):
            total_epochs = trained_epochs + epoch + 1
            logger.info(f'{"=" * 10} Epoch: {total_epochs} {"=" * 10}')
            for batch, d in enumerate(data):
                ldr_images = d['ldr_images']
                gt = d['gt']
                low = ldr_images[0].to(device)
                ref = ldr_images[1].to(device)
                high = ldr_images[2].to(device)
                gt = gt.to(device)

                hdr_pred = model(low, ref, high)
                loss = loss_func(hdr_pred, gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                score_psnr = psnr(hdr_pred, gt, torch.max(gt))
                logger.info(f'Epoch: {total_epochs} |Batch: {batch} --- Loss: {loss:.8f},'
                            f' PSNR-L: {score_psnr:.4f}')

            _save_model(model, optimizer, total_epochs, checkpoint_path)

    logger.info(f'{"=" * 20}End Training{"=" * 20}\n')


if __name__ == '__main__':
    model_choices = ('ahdrnet', 'adnet', 'ecadnet-gc6')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', choices=model_choices, required=True)
    parser.add_argument('--save-dir', dest='save_dir', required=True)
    parser.add_argument('--activation', choices=['relu', 'sigmoid'], default='relu')
    parser.add_argument('--loss', choices=['l1', 'mse'], default='mse')
    parser.add_argument('--data-path', dest='data_path', required=True)
    parser.add_argument('--dataset', choices=['ntire', 'kalantari'], default='ntire')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=4)
    parser.add_argument('--two-level-dir', dest='two_l_dir', action='store_true')
    parser.add_argument('--cpu', dest='cpu', action='store_true')
    args = parser.parse_args()

    train(model=get_model(args.model, out_activation=args.activation),
          epochs=args.epochs,
          batch_size=args.batch_size,
          data_path=args.data_path,
          checkpoint_path=rf'../models/{args.save_dir}/checkpoint.pth',
          logger_name=args.model,
          log_path=rf'../models/{args.save_dir}/train.log',
          dataset=args.dataset,
          loss_type=args.loss,
          two_level_dir=args.two_l_dir,
          use_cpu=args.cpu)
