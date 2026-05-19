# original U-Net
# Modified from https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_utils import inconv, down_block, up_block
from .utils import get_block, get_norm
import pdb


class UNet(nn.Module):
    def __init__(self, in_ch, base_ch, scale=[[1,2,2], [2,2,2], [2,2,2], [2,2,2]], kernel_size=[[1,3,3], [2,3,3], [3,3,3], [3,3,3], [3,3,3]], num_classes=1, block='ConvNormAct', pool=True, norm='bn'):
        super().__init__()
        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
    
        self.inc = inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)

        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[1], norm=norm)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[2], norm=norm)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[3], norm=norm)
        self.down4 = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[4], norm=norm)

        self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)

        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        
        # For fake feature processing
        self.fake_fc = nn.Conv3d(base_ch, num_classes, kernel_size=1)

    def forward(self, x, fake_features=None, ret_intermediate=False): 
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        out = self.up1(x5, x4) 
        out = self.up2(out, x3) 
        out = self.up3(out, x2)
        out = self.up4(out, x1)  # This is the feature map before final classifier
        
        # Main output
        logits = self.outc(out)
        
        # Process fake features if provided
        logits_fake = None
        if fake_features is not None:
            # fake_features shape: [1, feature_dim, N_fake_samples, 1]
            # We need to process them through the final classifier
            logits_fake = self.fake_fc(fake_features)
        
        if ret_intermediate:
            return logits, out, logits_fake
        
        return logits, out if fake_features is None else (logits, out, logits_fake)

    def forward_fc(self, features):
        """Forward only through final classifier - for prototype loss"""
        return self.outc(features)

    def embedding(self, x):
        x = self.feature(x)
        x = self.avgpool_layer(x)
        x = x.permute(0, 1, 4, 3, 2)
        return x