from .ntire import NTIREDataset
from .kalantari import KalantariDataset
from hdr_toolkit.data.data_io import gamma_correction, ev_align, read_ldr, read_tiff


__all__ = ['NTIREDataset', 'KalantariDataset', 'get_dataset', 'gamma_correction', 'ev_align', 'read_ldr']


def get_dataset(dataset, read_dir, **kwargs):
    if dataset == 'ntire':
        return NTIREDataset(read_dir, **kwargs)
    elif dataset == 'kalantari':
        return KalantariDataset(read_dir, **kwargs)
    else:
        raise ValueError('Invalid dataset')
