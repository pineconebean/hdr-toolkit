import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

import hdr_toolkit.data.data_io as io
from hdr_toolkit.data.scene_category import get_scene
from hdr_toolkit.hdr_ops.tonemap import tonemap
from hdr_toolkit.metrics.psnr import psnr, normalized_psnr, psnr_tanh_norm_mu_tonemap
from hdr_toolkit.util.logging import get_logger


def eval_kal(res_dir, ref_dir, out_dir):
    results = sorted(list(Path(res_dir).glob('*.hdr')))
    references = sorted(list(Path(ref_dir).glob('*gt.hdr')))
    out_dir = Path(out_dir)
    logger = get_logger('eval', str(out_dir.joinpath('scores.log')))

    scores_linear = []
    scores_tonemap = []

    for i, (res, ref) in enumerate(zip(results, references)):
        res, ref = str(res), str(ref)
        logger.info(f'res: {res} | ref: {ref}')

        ref_hdr_image = torch.from_numpy(cv2.cvtColor(cv2.imread(ref, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))
        res_hdr_image = torch.from_numpy(cv2.cvtColor(cv2.imread(res, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))

        score_linear = psnr(ref_hdr_image, res_hdr_image)
        score_tonemap = psnr(tonemap(ref_hdr_image), tonemap(res_hdr_image))

        scores_linear.append(score_linear)
        scores_tonemap.append(score_tonemap)

        logger.info(f'processed: {i + 1}/{len(results)}, psnr-t: {scores_tonemap[-1]}, psnr-l: {scores_linear[-1]}')

    psnr_l = np.mean(scores_linear)
    psnr_t = np.mean(scores_tonemap)
    logger.info(f'avg psnr-t: {psnr_t}')
    logger.info(f'avg psnr-l: {psnr_l}')
    with open(str(out_dir.joinpath('scores.txt')), 'w') as f:
        f.write(f'psnr-t: {psnr_t}\n')
        f.write(f'psnr-l: {psnr_l}\n')


def evaluate(res_dir, ref_dir, out_dir):
    results = sorted(list(Path(res_dir).glob('*.png')))
    references = sorted(list(Path(ref_dir).glob('*gt.png')))
    out_dir = Path(out_dir)
    logger = get_logger('eval', str(out_dir.joinpath('scores.log')))

    scores_linear = []
    scores_tonemap = []
    scene_scores_linear = {}
    scene_scores_tonemap = {}

    for i, (res, ref) in enumerate(zip(results, references)):
        sample_scene = get_scene(int(res.stem))
        res, ref = str(res), str(ref)
        logger.info(f'res: {res} | ref: {ref}')
        res_align_ratios = res.replace('.png', '_alignratio.npy')
        ref_align_ratios = ref.replace('gt.png', 'alignratio.npy')

        ref_hdr_image = io.imread_uint16_png(ref, ref_align_ratios)
        res_hdr_image = io.imread_uint16_png(res, res_align_ratios)

        score_linear = normalized_psnr(ref_hdr_image, res_hdr_image, np.max(ref_hdr_image))
        score_tonemap = psnr_tanh_norm_mu_tonemap(ref_hdr_image, res_hdr_image)

        scores_linear.append(score_linear)
        scores_tonemap.append(score_tonemap)
        scene_scores_linear.setdefault(sample_scene, []).append(score_linear)
        scene_scores_tonemap.setdefault(sample_scene, []).append(score_tonemap)

        logger.info(f'processed: {i + 1}/{len(results)}, psnr-t: {scores_tonemap[-1]}, psnr-l: {scores_linear[-1]}')

    psnr_l = np.mean(scores_linear)
    psnr_t = np.mean(scores_tonemap)
    with open(str(out_dir.joinpath('scores.txt')), 'w') as f:
        f.write(f'psnr-t: {psnr_t}\n')
        f.write(f'psnr-l: {psnr_l}\n')

    for (scene_l, scores_l), (scene_t, scores_t) in zip(scene_scores_linear.items(), scene_scores_tonemap.items()):
        avg_score_l = np.mean(scores_l)
        avg_score_t = np.mean(scores_t)
        with open(str(out_dir.joinpath('scores.txt')), 'a') as f:
            f.write(f'{scene_t} -- psnr-t: {avg_score_t}\n')
            f.write(f'{scene_l} -- psnr-l: {avg_score_l}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-dir', dest='res_dir', required=True)
    parser.add_argument('--reference-dir', dest='ref_dir', required=True)
    parser.add_argument('--output-dir', dest='out_dir')
    parser.add_argument('--dataset', dest='dataset', choices=['ntire', 'kalantari'], required=True)
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.res_dir

    if args.dataset == 'kalantari':
        eval_kal(args.res_dir, args.ref_dir, args.out_dir)
    elif args.dataset == 'ntire':
        evaluate(args.res_dir, args.ref_dir, args.out_dir)
    else:
        raise ValueError('Invalid dataset')
