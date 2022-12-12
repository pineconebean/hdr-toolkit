from hdr_toolkit.networks.adnet import ADNet
from hdr_toolkit.networks.ahdrnet import AHDRNet
from hdr_toolkit.networks.experimental import ECADNet, PSFTDNet, BAHDRNet, ResRefAHDR


def get_model(model_type, out_activation='relu'):
    if model_type == 'ahdr':
        model = AHDRNet(n_channels=64, out_activation=out_activation)
    elif model_type == 'adnet':
        model = ADNet(n_channels=64, n_dense_layers=3, growth_rate=32, out_activation=out_activation)
    elif model_type == 'ecadnet-gc6':
        model = ECADNet(n_channels=64, trans_conv_groups=6, out_activation=out_activation)
    elif model_type == 'ecadnet':
        model = ECADNet(n_channels=64, trans_conv_groups=6, out_activation=out_activation, use_trans=False)
    elif model_type == 'psftd':
        model = PSFTDNet(out_activation=out_activation)
    elif model_type == 'ba-default':
        model = BAHDRNet(n_channels=64, out_activation=out_activation)
    elif model_type == 'res-ref-ahdr':
        model = ResRefAHDR(n_channels=64, out_activation=out_activation)
    else:
        raise ValueError('invalid model type')
    return model
