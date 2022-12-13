from torchsummary import summary
import torch

from hdr_toolkit.networks import (ADNet, ECADNet, PSFTDNet, AHDRNet, BAHDRNet, ResRefAHDR, ResRefSFTNet, ResRefDANet)

# a = PSFTDNet(extract_same_feat=True, naive_pyramid_sft=True, same_conv_for_pyramid=True, share_offsets=True)
# a = AHDRNet(out_activation='sigmoid')
# a = BAHDRNet(64, out_activation='sigmoid')
# a = ResRefAHDR(64, out_activation='sigmoid')
# a = ResRefSFTNet.create('default', 64, out_activation='sigmoid')
a = ResRefDANet.create('groups16', 64, out_activation='sigmoid')
# a = ADNet(64, 3, 32, out_activation='sigmoid')
summary(a, ((4, 6, 256, 256), (4, 6, 256, 256), (4, 6, 256, 256)), batch_dim=None,
        dtypes=[torch.float32, torch.float32, torch.float32], depth=4)
