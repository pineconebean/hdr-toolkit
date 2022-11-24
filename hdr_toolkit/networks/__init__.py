from hdr_toolkit.networks.adnet import ADNet
from hdr_toolkit.networks.ahdrnet import AHDRNet


def get_model(model_type, out_activation='relu'):
    if model_type == 'ahdr':
        model = AHDRNet(out_activation=out_activation)
    elif model_type == 'adnet':
        model = ADNet(6, 3, 64, 32, out_act=out_activation)
    else:
        raise ValueError('invalid model type')
    return model
