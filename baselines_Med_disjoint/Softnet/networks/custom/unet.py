# original U-Net
# Modified from https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_utils import inconv, down_block, up_block
from .utils import get_block, get_norm
import pdb


import torch
import numpy as np
import random
from .signet import SignetConv3d


class UNet(nn.Module):
    def __init__(self, in_ch, base_ch, scale=[[1,2,2], [2,2,2], [2,2,2], [2,2,2]], kernel_size=[[1,3,3], [2,3,3], [3,3,3], [3,3,3], [3,3,3]], num_classes=1, block='ConvNormAct', pool=True, norm='bn'):
        super().__init__()
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: should be a list to indicate the downsample scale along each axis 
                in each level, e.g. [1, 1, 2, 2] such that all axis use the same scale
                or [[1,2,2], [2,2,2], [2,2,2], [2,2,2]] for difference scale on each axis
            kernel_size: the 3D kernel size of each level
                e.g. [3,3,3,3] or [[1,3,3], [1,3,3], [3,3,3], [3,3,3]]
            num_classes: the target class number
            block: 'ConvNormAct' for origin UNet, 'BasicBlock' for ResUNet
            pool: use maxpool or use strided conv for downsample
            norm: the norm layer type, bn or in

        '''
        random.seed(1024)
        np.random.seed(1024)
        torch.manual_seed(1024)
        torch.cuda.manual_seed(1024)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
        self.scale=10

        self.inc = inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)

        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[1], norm=norm)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[2], norm=norm)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[3], norm=norm)
        self.down4 = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[4], norm=norm)

        self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)

        # self.feature = nn.Conv3d(in_channels=base_ch, 
        #                          out_channels=num_classes, 
        #                          kernel_size=(3, 3, 3), 
        #                          padding=1)
        # self.avgpool_layer = nn.AdaptiveAvgPool3d((3, 9, 16))
        # self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        
        self.mask_out = False
        if self.mask_out:
            self.outc = SignetConv3d(base_ch, num_classes, kernel_size=1)
        else:
            self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)



    def forward(self, x): 
    
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        out = self.up1(x5, x4) 
        out = self.up2(out, x3) 
        out = self.up3(out, x2)
        out = self.up4(out, x1)

        embedding = out.clone()

        out = F.normalize(out, p=2, dim=1)
        kw=F.normalize(self.outc.weight, dim=1, p=2)
        
        similarities=self.scale*F.conv3d(out, kw)
        

        return embedding, similarities

    def embedding(self, x):
        x = self.feature(x)
        x = self.avgpool_layer(x)
        x = x.permute(0, 1, 4, 3, 2)
        # x = self.l2_norm(x)
        return x
        
    def get_masks(self, task_id=0):
        task_mask = []
        for module in self.modules():
            if isinstance(module, SignetConv3d):
                task_mask.append(module.weight_mask.detach().clone().type(torch.float))
                if getattr(module, 'bias') is not None:
                    task_mask.append(module.bias_mask.detach().clone().type(torch.float))

        return task_mask
    
    def get_masks_util(self, task_id=0):
        task_mask = {
            
        }
        for name, module in self.named_modules():
            if isinstance(module, SignetConv3d):
                #print(name)
                task_mask[name] = module.weight_mask.detach().clone().type(torch.float)
                # task_mask.append(module.weight_mask.detach().clone().type(torch.float))
                if getattr(module, 'bias') is not None:
                    task_mask[name + 'bias'] = module.bias_mask.detach().clone().type(torch.float)

        return task_mask

