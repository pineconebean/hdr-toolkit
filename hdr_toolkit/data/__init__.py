from .ntire import NTIREDataset
from .kalantari import KalantariDataset


def get_dataset(dataset, read_dir, **kwargs):
    if dataset == 'ntire':
        return NTIREDataset(read_dir, **kwargs)
    elif dataset == 'kalantari':
        return KalantariDataset(read_dir, **kwargs)
    else:
        raise ValueError('Invalid dataset')
