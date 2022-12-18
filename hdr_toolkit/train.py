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


def _get_data_set(name, path, batch_size=4, two_level_dir=False, use_ea=False):
    if name == 'kalantari':
        return DataLoader(KalantariDataset(path, exposure_aligned=use_ea, hdr_domain=not use_ea),
                          batch_size=batch_size, shuffle=True)
    elif name == 'ntire':
        return DataLoader(NTIREDataset(path, two_level_dir=two_level_dir), batch_size=batch_size, shuffle=True)
    else:
        raise ValueError('Unexpected dataset')


def _save_model(model, optimizer, epoch, save_path, val_scores=None):
    if isinstance(model, torch.nn.DataParallel):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    things_to_save = {
        'model': model_state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch
    }

    if val_scores is not None:
        things_to_save['val_scores'] = val_scores

    torch.save(things_to_save, save_path)


def train(model, epochs, batch_size, data_path, val_data_path, dataset, save_dir, log_path, logger_name,
          learning_rate=1e-4, loss_type='mse', two_level_dir=False, use_cpu=False, val_interval=1000, use_ea=False):
    data = _get_data_set(dataset, data_path, batch_size=batch_size, two_level_dir=two_level_dir, use_ea=use_ea)
    val_data = _get_data_set(dataset, val_data_path, batch_size=batch_size, use_ea=use_ea)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    trained_epochs = 0
    best_val_scores = (0., 0.)  # psnr-l, psnr-t
    if use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device("cuda:0")
    model.to(device)
    torch.autograd.set_detect_anomaly(True)

    # load checkpoint
    save_dir_path = pathlib.Path(save_dir)
    save_dir_path.mkdir(exist_ok=True, parents=True)
    checkpoint_path = save_dir_path.joinpath('ckpt.pth')
    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path))
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer'])
        trained_epochs = checkpoint['epoch']
        # if 'val_scores' in checkpoint:
        #     best_val_scores = checkpoint['val_scores']

    logger = get_logger(logger_name, log_path)
    val_logger = get_logger('validation', str(save_dir_path.joinpath('validation.log')))
    logger.info(f'{"=" * 20}Start Training{"=" * 20}\n')
    model.train()
    if dataset == 'kalantari':
        total_batches = 0  # used for validation
        loss_func = torch.nn.L1Loss()
        for epoch in range(epochs):
            total_epochs = trained_epochs + epoch + 1
            logger.info(f'{"=" * 10} Epoch: {total_epochs} {"=" * 10}')
            for batch, d in enumerate(data):
                low, ref, high, gt = _data_to_device(d, device)
                hdr_pred = model(low, ref, high)
                if hdr_pred is None:
                    raise ValueError('hdr prediction is None')
                loss = loss_func(tonemap(hdr_pred, dataset=dataset), tonemap(gt, dataset=dataset))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logger.info(f'Epoch: {total_epochs} |Batch: {batch} --- Loss: {loss:.8f},'
                            f' PSNR-L: {psnr(hdr_pred, gt):.4f} | PSNR-T: {psnr(tonemap(hdr_pred), tonemap(gt)):.4f}')

                # perform validation
                total_batches += 1
                if total_batches % val_interval == 0:
                    best_val_scores = _kal_validation(model, optimizer, val_data, total_epochs, batch,
                                                      best_val_scores, device, save_dir, val_logger)

            best_val_scores = _kal_validation(model, optimizer, val_data, total_epochs, 'Epoch end',
                                              best_val_scores, device, save_dir, val_logger)
            _save_model(model, optimizer, total_epochs, str(checkpoint_path), val_scores=best_val_scores)

    elif dataset == 'ntire':
        loss_func = NtireMuLoss(loss_func=loss_type)
        for epoch in range(epochs):
            total_epochs = trained_epochs + epoch + 1
            logger.info(f'{"=" * 10} Epoch: {total_epochs} {"=" * 10}')
            for batch, d in enumerate(data):
                low, ref, high, gt = _data_to_device(d, device)

                hdr_pred = model(low, ref, high)
                loss = loss_func(hdr_pred, gt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                score_psnr = psnr(hdr_pred, gt, torch.max(gt))
                logger.info(f'Epoch: {total_epochs} |Batch: {batch} --- Loss: {loss:.8f},'
                            f' PSNR-L: {score_psnr:.4f}')

            _save_model(model, optimizer, total_epochs, save_dir)

    logger.info(f'{"=" * 20}End Training{"=" * 20}\n')


def _kal_validation(model, optimizer, val_data, epoch, batch, best_val_scores, device, save_dir, val_logger):
    model.eval()
    psnr_l, psnr_t = 0., 0.
    with torch.no_grad():
        for i, data in enumerate(val_data):
            low, ref, high, gt = _data_to_device(data, device)
            pred = model(low, ref, high)

            mu_pred, mu_gt = tonemap(pred), tonemap(gt)

            psnr_l = (i * psnr_l + psnr(pred, gt)) / (i + 1)
            psnr_t = (i * psnr_t + psnr(mu_pred, mu_gt)) / (i + 1)

    save_dir_path = pathlib.Path(save_dir)
    val_logger.info(f'{"=" * 20}Validation for Epoch {epoch} Batch {batch}{"=" * 20}')

    best_psnr_l, best_psnr_t = best_val_scores
    update_l, update_t = False, False
    if psnr_l > best_psnr_l:
        best_psnr_l = psnr_l
        _save_model(model, optimizer, epoch, str(save_dir_path.joinpath('val-l-ckpt.pth')))
        update_l = True
    if psnr_t > best_psnr_t:
        best_psnr_t = psnr_t
        _save_model(model, optimizer, epoch, str(save_dir_path.joinpath('val-t-ckpt.pth')))
        update_t = True

    val_logger.info(f'psnr-l: {psnr_l:.5f} ({best_psnr_l:.5f}{" up" if update_l else ""}) | '
                    f'psnr-t: {psnr_t:.5f} ({best_psnr_t:.5f}{" up" if update_t else ""})\n')
    model.train()

    return best_psnr_l, best_psnr_t


def _data_to_device(data, device):
    ldr_images = data['ldr_images']
    low = ldr_images[0].to(device)
    ref = ldr_images[1].to(device)
    high = ldr_images[2].to(device)
    gt = data['gt'].to(device)

    return low, ref, high, gt


if __name__ == '__main__':
    # model_choices = ('ahdrnet', 'adnet', 'ecadnet-gc6', 'ecadnet', 'psftd')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('--save-dir', dest='save_dir', required=True)
    parser.add_argument('--activation', choices=['relu', 'sigmoid'], default='relu')
    parser.add_argument('--loss', choices=['l1', 'mse'], default='mse')
    parser.add_argument('--data-path', dest='data_path', required=True)
    parser.add_argument('--val-data-path', dest='val_data_path', required=True)
    parser.add_argument('--val-interval', dest='val_interval', required=True, default=1000, type=int)
    parser.add_argument('--dataset', choices=['ntire', 'kalantari'], default='ntire')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=4)
    parser.add_argument('--two-level-dir', dest='two_l_dir', action='store_true')
    parser.add_argument('--cpu', dest='cpu', action='store_true')
    parser.add_argument('--ea', dest='ea', action='store_true')
    args = parser.parse_args()

    train(model=get_model(args.model, out_activation=args.activation),
          epochs=args.epochs,
          batch_size=args.batch_size,
          data_path=args.data_path,
          val_data_path=args.val_data_path,
          save_dir=rf'../models/{args.save_dir}',
          logger_name=args.model,
          log_path=rf'../models/{args.save_dir}/train.log',
          dataset=args.dataset,
          loss_type=args.loss,
          two_level_dir=args.two_l_dir,
          use_cpu=args.cpu,
          val_interval=args.val_interval,
          use_ea=args.ea)
