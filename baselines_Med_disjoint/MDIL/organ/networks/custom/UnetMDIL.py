# original U-Net
# Modified from https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_utils import inconv, down_block, up_block
from .utils import get_block, get_norm
import pdb
from networks.custom.encoder import UNet_encoder
from networks.custom.encoder import UNet_decoder

class UnetMDIL(nn.Module):
    def __init__(self, opts, num_classes):
        super(UnetMDIL, self).__init__()
        self.task = opts.step
        self.encoder = UNet_encoder(in_ch=1, base_ch=32, num_classes = opts.num_classes, block = 'SingleConv', step = self.task).cuda()
        # self.decoder = UNet_decoder(in_ch=1, base_ch=32, num_classes = opts.num_classes, block = 'SingleConv').cuda()
        l_classes = [16, 21, 27]
        self.decoder = nn.ModuleList([UNet_decoder(in_ch=1, base_ch=32, num_classes = l_classes[i], block = 'SingleConv').cuda() for i in range(self.task + 1)])
        self.seg_head = nn.ModuleList([nn.Conv3d(32, l_classes[i], kernel_size=1) for i in range(self.task + 1)])

        if self.task == 0:
            path = torch.load("cil/MDIL/organ/checkpoints/step/15ss-merged/feature_extractor.pth")
            pretrained_model = path['model_state']["model"]
            for name, param in pretrained_model.items():
                if name.startswith('down'):
                    self.encoder.state_dict()[name].copy_(param)
                elif name.startswith('up'):
                    self.decoder[0].state_dict()[name].copy_(param)
                    
    def forward(self, x, current_task):
        x1, x2, x3, x4, x5 = self.encoder(x)
        out = self.decoder[current_task].forward(x1, x2, x3, x4, x5)
        # out = self.decoder(x1, x2, x3, x4, x5)
        out = self.seg_head[current_task](out)

        return out
    
   

        