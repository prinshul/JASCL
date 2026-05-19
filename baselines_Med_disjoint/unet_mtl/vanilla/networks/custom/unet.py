# original U-Net
# Modified from https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_utils import inconv, down_block, up_block
from .utils import get_block, get_norm
import pdb


class UNet(nn.Module):
    def __init__(self, opts, in_ch, base_ch, scale=[[1,2,2], [2,2,2], [2,2,2], [2,2,2]], kernel_size=[[1,3,3], [2,3,3], [3,3,3], [3,3,3], [3,3,3]], num_classes=1, block='ConvNormAct', pool=True, norm='bn'):
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

        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
        self.base_ch_curr = base_ch
        
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
        #self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        self.outc = nn.Conv3d(base_ch, 16, kernel_size=1)
        
        self.task = [16,5,6,4,3,4]
        self.curr_step = opts.step
        self.curr_class = self.task[opts.step]
        self.curr_cum_class=0
        for up_cls in range(1,self.curr_step+1):
            self.curr_cum_class+=self.task[up_cls]
        
        
        self.single_head = nn.ModuleList([nn.Conv3d(base_ch, 1, kernel_size=1) for i in range(self.curr_cum_class)])
        
        self.step1_test = nn.Conv3d(base_ch, 5, kernel_size=1)
        self.step2_test = nn.Conv3d(base_ch, 6, kernel_size=1)
        self.step3_test = nn.Conv3d(base_ch, 4, kernel_size=1)
        
        
    
    def decoder_single(self, out):
        out_singles = []
        for smodule in self.single_head:
            out_singles.append(smodule(out))
        return torch.cat(out_singles,dim=1)
    
    def decoder_task(self, out): 
        if self.curr_step==1:
            return self.step1_test(out)
        if self.curr_step==2:
            step1_out = self.step1_test(out)
            step2_out = self.step2_test(out)
            return torch.cat([step1_out,step2_out],dim=1)
        if self.curr_step==3:
            step1_out = self.step1_test(out)
            step2_out = self.step2_test(out)
            step3_out = self.step3_test(out)
            return torch.cat([step1_out,step2_out,step3_out],dim=1)
            
        
    
    def forward(self, x, run_type): 
    
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        out = self.up1(x5, x4) 
        out = self.up2(out, x3) 
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        
        out_base = self.outc(out)
        
        outputs = []
        outputs.append(out_base)
        if run_type=='train':
            outputs.append(self.decoder_single(out))
        else:
            #print('val')
            outputs.append(self.decoder_task(out))
        outputs = torch.cat(outputs,dim=1)

        return outputs
    
    '''
    MTL heads train:single heads for each class.
    test:single heads for each step.
    
    def decoder_single(self, x1, x2, x3, x4, x5): 
        out = self.up1(x5, x4) 
        out = self.up2(out, x3) 
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        outc = self.single_head(out)
        return outc
    
    def decoder_task(self, x1, x2, x3, x4, x5, cls_cur): 
        out = self.up1(x5, x4) 
        out = self.up2(out, x3) 
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        last_layer = nn.Conv3d(self.base_ch_curr, cls_cur, kernel_size=1)
        out = last_layer(out)
        return out
    
    def forward(self, x, run_type): 
    
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        out = self.up1(x5, x4) 
        out = self.up2(out, x3) 
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        out = self.outc(out)
        
        outputs = []
        outputs.append(out)
        if run_type=='train':
            for xi in range(self.curr_cum_class):
                outputs.append(self.decoder_single(x1, x2, x3, x4, x5))
        else:
            for yi in range(1,self.curr_step+1):
                outputs.append(self.decoder_task(x1, x2, x3, x4, x5,self.task[yi]))
        outputs = torch.cat(outputs,dim=0)

        return outputs
    '''
    
    def embedding(self, x):
        x = self.feature(x)
        x = self.avgpool_layer(x)
        x = x.permute(0, 1, 4, 3, 2)
        # x = self.l2_norm(x)
        return x



