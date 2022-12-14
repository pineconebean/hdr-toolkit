from pathlib import Path

import cv2
import flow_vis
import numpy as np
import torch
import torch.nn.functional as F

from hdr_toolkit.data.data_io import read_ldr, gamma_correction, ev_align
from hdr_toolkit.networks.blocks.flow.spynet import SpyNet


def adjust_exposure(img, exp, target_exp):
    ga_img = gamma_correction(img, exp, 2.2)
    return (ga_img * (2 ** target_exp)) ** (1 / 2.2)


def optical_flow_for_img(img_dir, img_id, target, adjust_direction, scale_factor=1.):
    img_path = Path(img_dir)
    exposures = np.loadtxt(img_path.joinpath(f'{img_id[:3]}_exposure.txt'))

    def read_and_adjust(name, exposure_idx):
        x = read_ldr(str(img_path.joinpath(f'{img_id}_{name}.tif')), 16, scale_factor)
        ea_x = ev_align(x, exposures[exposure_idx], 2.2)
        ga_x = gamma_correction(x, exposures[exposure_idx], 2.2)
        return x, ea_x, ga_x

    ref, ea_ref, ga_ref = read_and_adjust('medium', 1)

    if target == 'sm':
        non_ref, ea_non_ref, ga_non_ref = read_and_adjust('short', 0)
    elif target == 'lm':
        non_ref, ea_non_ref, ga_non_ref = read_and_adjust('long', 2)
    else:
        raise KeyError(f'Invalid target {target} (choose one from `sm` and `lm`)')

    if adjust_direction == 'up':
        if target == 'sm':
            adj_ref, adj_non_ref = ref, adjust_exposure(non_ref, exposures[0], exposures[1])
        else:
            adj_ref, adj_non_ref = adjust_exposure(ref, exposures[1], exposures[2]), non_ref
    elif adjust_direction == 'down':
        if target == 'sm':
            adj_ref, adj_non_ref = adjust_exposure(ref, exposures[1], exposures[0]), non_ref
        else:
            adj_ref, adj_non_ref = ref, adjust_exposure(non_ref, exposures[2], exposures[1])
    else:
        raise ValueError(f'Invalid adjust direction {adjust_direction} (choose one from `up`, `down` and `middle`')

    spy_net = SpyNet('../models/spy_net/ckpt.pth')

    out_dir = Path(f'../models/spy_net/flow_images/{img_id}/')
    out_dir.mkdir(exist_ok=True, parents=True)

    def predict_and_save(x1, x2, name):
        with torch.no_grad():
            flow = flow_vis.flow_to_color(spy_net(x1.unsqueeze(0), x2.unsqueeze(0)).squeeze().permute(1, 2, 0).numpy())
            cv2.imwrite(str(out_dir.joinpath(f'{target}-{name}-{scale_factor}.png')), flow)

    predict_and_save(ref, non_ref, 'flow_wo_adjust')
    predict_and_save(ea_ref, ea_non_ref, 'flow_ea')
    predict_and_save(ga_ref, ga_non_ref, 'flow_ga')
    predict_and_save(adj_ref, adj_non_ref, f'flow_{adjust_direction}')


def visualize_optical_flow():
    with torch.no_grad():
        img_path = Path('/Users/jundaliao/Documents/HDR/Kalantari/Test2/')
        img_id = '009'
        long = read_ldr(str(img_path.joinpath(f'{img_id}_long.tif')), 16)
        short = read_ldr(str(img_path.joinpath(f'{img_id}_short.tif')), 16)
        medium = read_ldr(str(img_path.joinpath(f'{img_id}_medium.tif')), 16)
        exposures = np.loadtxt(img_path.joinpath(f'{img_id}_exposure.txt'))

        s_medium = adjust_exposure(medium, exposures[1], exposures[0]).unsqueeze(0)
        m_short = adjust_exposure(short, exposures[0], exposures[1]).unsqueeze(0)
        cv2.imshow('test', cv2.cvtColor(m_short.squeeze().permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)
        ea_short = ev_align(short, exposures[0], 2.2).unsqueeze(0)
        ea_medium = ev_align(medium, exposures[1], 2.2).unsqueeze(0)
        ga_short = gamma_correction(short, exposures[0], 2.2).unsqueeze(0)
        ga_medium = gamma_correction(medium, exposures[1], 2.2).unsqueeze(0)
        ea_medium_half = F.interpolate(ea_medium, scale_factor=0.25, mode='bicubic')
        ea_short_half = F.interpolate(ea_short, scale_factor=0.25, mode='bicubic')

        spy_net = SpyNet('../models/spy_net/ckpt.pth')
        # ea_sm_flow = spy_net(ea_medium, ea_short).squeeze().permute(1, 2, 0).numpy()
        # ga_sm_flow = spy_net(ga_medium, ga_short).squeeze().permute(1, 2, 0).numpy()
        adjusted_sm_flow = spy_net(medium.unsqueeze(0), m_short).squeeze().permute(1, 2, 0).numpy()
        # half_sm_flow = spy_net(ea_medium_half, ea_short_half).squeeze().permute(1, 2, 0).numpy()

        # flow_color_ea_sm = flow_vis.flow_to_color(ea_sm_flow, convert_to_bgr=False)
        # flow_color_ga_sm = flow_vis.flow_to_color(ga_sm_flow, convert_to_bgr=False)
        flow_color_adjusted_sm = flow_vis.flow_to_color(adjusted_sm_flow, convert_to_bgr=False)
        # flow_half_sm = flow_vis.flow_to_color(half_sm_flow, convert_to_bgr=False)
        # cv2.imwrite('../models/spy_net/ea_sm_flow.png', flow_color_ea_sm)
        # cv2.imwrite('../models/spy_net/ga_sm_flow.png', flow_color_ga_sm)
        cv2.imwrite('../models/spy_net/adjusted_stom_flow.png', flow_color_adjusted_sm)
        # cv2.imwrite('../models/spy_net/0.25_sm_flow.png', flow_half_sm)


def main():
    img_path = '/Users/jundaliao/Documents/HDR/Kalantari/Val/'
    img_id = '005_14'

    optical_flow_for_img(img_path, img_id, 'lm', 'down', scale_factor=1)


main()
