# original U-Net
# Modified from https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_utils import inconv, down_block, up_block
from .utils import get_block, get_norm
import pdb

class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()

        self.conv3x1_1 = nn.Conv3d(chann, chann, kernel_size = (3, 3, 1), padding=(1, 1, 0), bias=True)

        self.conv1x3_1 = nn.Conv3d(chann, chann, kernel_size = (1, 3, 3), padding=(0, 1, 1), bias=True)

        self.bn1 = nn.BatchNorm3d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv3d(chann, chann, (3, 3, 1), padding=(
            1*dilated, 1*dilated, 0), bias=True, dilation=(dilated, dilated, 1))

        self.conv1x3_2 = nn.Conv3d(chann, chann, (1, 3, 3), padding=(
            0, 1*dilated, 1*dilated), bias=True, dilation=(1, dilated, dilated))

        self.bn2 = nn.BatchNorm3d(chann, eps=1e-03)

        self.dropout = nn.Dropout3d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)  # +input = identity (residual connection)


class non_bottleneck_1d_RAP (nn.Module):
    def __init__(self, chann, dropprob, dilated, step =1):
        #chann = #channels, dropprob=dropout probability, dilated=dilation rate
        super().__init__()
        self.current_task = step
        self.conv3x1_1 = nn.Conv3d(chann, chann, kernel_size = (3, 3, 1), padding=(1, 1, 0), bias=True)
        self.conv1x3_1 = nn.Conv3d(chann, chann, kernel_size = (1, 3, 3), padding=(0, 1, 1), bias=True)

        # domain-specific 1x1conv
        self.parallel_conv_1 = nn.ModuleList([nn.Conv3d(chann, chann, kernel_size=1, stride=1, padding=0, bias=True) for i in range(step + 1)]) #step + 1=1 for 1st time, its only on CS
        self.bns_1 = nn.ModuleList([nn.BatchNorm3d(chann, eps=1e-03) for i in range(step + 1)])

        self.conv3x1_2 = nn.Conv3d(chann, chann, (3, 3, 1), padding=(
            1*dilated, 1*dilated, 0), bias=True, dilation=(dilated, dilated, 1))

        self.conv1x3_2 = nn.Conv3d(chann, chann, (1, 3, 3), padding=(
            0, 1*dilated, 1*dilated), bias=True, dilation=(1, dilated, dilated))

        self.parallel_conv_2 = nn.ModuleList([nn.Conv3d(chann, chann, kernel_size=1, stride=1, padding=0, bias=True) for i in range(step + 1)])
        self.bns_2 = nn.ModuleList([nn.BatchNorm3d(chann, eps=1e-03) for i in range(step + 1)])

        self.dropout = nn.Dropout3d(dropprob)

    def forward(self, input):
        task = self.current_task
        # print('input: ', input.size())
        output = self.conv3x1_1(input)
        # print("..", output.shape)
        output = F.relu(output)
        # print("..", output.shape)

        output = self.conv1x3_1(output)
        # print("..", output.shape)

        # print('output 2nd 1x3: ', output.size())

        output = output + self.parallel_conv_1[task](input) # RAP skip connection for conv2
        output = self.bns_1[task](output)
        # print("..", output.shape)

        output_ = F.relu(output)

        output = self.conv3x1_2(output_)
        output = F.relu(output)
        output = self.conv1x3_2(output)

        output = output + self.parallel_conv_2[task](output_) # RAP skip connection for conv2
        output = self.bns_2[task](output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)  # +input = identity (residual connection)

class UNet_encoder(nn.Module):
    def __init__(self, in_ch, base_ch, scale=[[1,2,2], [2,2,2], [2,2,2], [2,2,2]], kernel_size=[[1,3,3], [2,3,3], [3,3,3], [3,3,3], [3,3,3]], num_classes=1, block='ConvNormAct', pool=True, norm='bn', step = 0):
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
    
        self.inc = inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()

        for x in range(0, 5):  # 5 times
            self.layers1.append(non_bottleneck_1d_RAP(base_ch, 0.03, 1, step + 1))

        for x in range(0, 2):  # 2 times
            self.layers2.append(non_bottleneck_1d_RAP(10*base_ch, 0.3, 2, step + 1)) # dropprob for imagenet pretrained encoder is 0.1 not 0.3, here using 0.3 for imagenet pretrained encoder
            self.layers2.append(non_bottleneck_1d_RAP(10*base_ch, 0.3, 4, step + 1))
            self.layers2.append(non_bottleneck_1d_RAP(10*base_ch, 0.3, 8, step + 1))
            self.layers2.append(non_bottleneck_1d_RAP(10*base_ch, 0.3, 16, step + 1))
            
        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[1], norm=norm)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[2], norm=norm)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[3], norm=norm)
        self.down4 = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[4], norm=norm)

        # self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        # self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        # self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        # self.up4 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)

        # self.outc = nn.Conv3d(base_ch, 4*27, kernel_size=1)
        # self.outc_full = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        
        # # self.conv1_p = nn.Conv3d(320, 128, kernel_size=3, padding=1)
        # # self.conv2_p = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        # # self.conv3_p = nn.Conv3d(64, 8, kernel_size=3, padding=1)
        # self.global_avg_pool_p = nn.AdaptiveAvgPool3d(3)
        # self.fc = nn.Linear(8640, moco_dim, bias=False)


    def forward(self, x): 
        x1 = self.inc(x)
        # print(x1.shape)
        
        for layer in self.layers1:
            res = layer(x1)
        # print(res.shape)
        x2 = self.down1(res)
        # print(x2.shape)
        x3 = self.down2(x2)
        # print(x3.shape)
        x4 = self.down3(x3)
        # print(x4.shape)
        x5 = self.down4(x4)
        # print(x5.shape)

        for layer in self.layers2:
            output = layer(x5)
        # print(output.shape)
        return x1, x2, x3, x4, output
    
class UNet_decoder(nn.Module):
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

        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()

        self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)

        self.layers1.append(non_bottleneck_1d(8*base_ch, 0, 1))
        self.layers1.append(non_bottleneck_1d(8*base_ch, 0, 1))

        self.layers2.append(non_bottleneck_1d(base_ch, 0, 1))
        self.layers2.append(non_bottleneck_1d(base_ch, 0, 1))
        
    def forward(self, x1, x2, x3, x4, x5):
        out = self.up1(x5, x4) 
        
        for layer in self.layers1:
            res = layer(out)
            
        out = self.up2(res, x3) 
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        
        for layer in self.layers2:
            output = layer(out)
        return output