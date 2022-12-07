from torchsummary import summary
import torch

from hdr_toolkit.networks import ADNet, ECADNet, PSFTDNet, AHDRNet
# a = PSFTDNet(extract_same_feat=False, naive_pyramid=False)
a = AHDRNet(out_activation='sigmoid')
summary(a, ((4, 6, 256, 256), (4, 6, 256, 256), (4, 6, 256, 256)), batch_dim=None,
        dtypes=[torch.float32, torch.float32, torch.float32])