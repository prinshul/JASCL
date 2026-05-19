# original U-Net
# Modified from https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_utils import inconv, down_block, up_block
from .utils import get_block, get_norm
from methods.svf import resolver
import pdb
import random
import numpy as np

random.seed(1024)
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class ProbabilisticClassifier(nn.Module):

    def __init__(self, num_features, num_classes):
        super().__init__()
        torch.manual_seed(1024)
        torch.cuda.manual_seed(1024)
        self.mu = nn.Conv3d(num_features, num_classes, 1, bias=False)
        self.sigma = nn.Conv3d(num_features, num_classes, 1, bias=False)
        self.grad_update = nn.Parameter(torch.zeros(self.mu.weight.shape))
        self.temp = 10
            
    
    def forward(self, x, stochastic=True):
        mu = self.mu.weight
        sigma = self.sigma.weight
        inverse_grad = self.grad_update.data
        epsilon = 1e-8
        inverse_grad = 1 / (inverse_grad + epsilon)
        min_val = inverse_grad.min()
        max_val = inverse_grad.max()
        inverse_grad = (1 + inverse_grad - min_val) / (1+(max_val - min_val))
        if stochastic:
            weight = inverse_grad * torch.randn_like(mu) + mu
        else:
            weight = mu
        
        weight = F.normalize(weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)
        score = F.conv3d(x, weight)
        score = score * self.temp
        
        return score


class UNet(nn.Module):
    
    def __init__(self, in_ch, base_ch, scale=[[1,2,2], [2,2,2], [2,2,2], [2,2,2]], kernel_size=[[1,3,3], [2,3,3], [3,3,3], [3,3,3], [3,3,3]], num_classes=1, block='ConvNormAct', pool=True, norm='bn'):        
        super().__init__()
        torch.manual_seed(1024)
        torch.cuda.manual_seed(1024)
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

        self.fc = ProbabilisticClassifier(base_ch, num_classes)

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

        feat = out.clone()
        out = self.fc(out)
        return feat, out
    
    def forward_fc(self, x): 
        return self.fc(x)