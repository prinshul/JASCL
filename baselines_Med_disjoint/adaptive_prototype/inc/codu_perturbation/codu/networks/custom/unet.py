# original U-Net - Modified for Adaptive Prototype Replay
# Modified from https://github.com/milesial/Pytorch-UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_utils import inconv, down_block, up_block
from .utils import get_block, get_norm
import pdb


class UNet(nn.Module):
    def __init__(self, in_ch, base_ch, scale=[[1,2,2], [2,2,2], [2,2,2], [2,2,2]], 
                 kernel_size=[[1,3,3], [2,3,3], [3,3,3], [3,3,3], [3,3,3]], 
                 num_classes=1, block='ConvNormAct', pool=True, norm='bn'):
        super().__init__()
        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
    
        self.inc = inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)

        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, 
                               down_scale=scale[0], kernel_size=kernel_size[1], norm=norm)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, 
                               down_scale=scale[1], kernel_size=kernel_size[2], norm=norm)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, 
                               down_scale=scale[2], kernel_size=kernel_size[3], norm=norm)
        self.down4 = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, 
                               down_scale=scale[3], kernel_size=kernel_size[4], norm=norm)

        self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block, block=block, 
                           up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, 
                           up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, 
                           up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, 
                           up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)

        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        
        # For fake feature processing - separate classifier head
        self.fake_classifier = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        
        # For extra background feature processing
        self.extra_bg_classifier = nn.Conv3d(base_ch, num_classes, kernel_size=1)

    def forward(self, x, fake_features=None, region_bg=None, ret_intermediate=False): 
        """
        Args:
            x: Input images [N, C, D, H, W]
            fake_features: Generated fake features from prototypes [C, 1, N_fake, 1]
            region_bg: Background region mask for extra background sampling [N, D_s, H_s, W_s]
            ret_intermediate: Whether to return intermediate features
        
        Returns:
            logits: Main output logits
            features: Intermediate feature maps (list)
            extra: Tuple of (logits_fake, logits_extra_bg)
        """
        # Standard forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        out = self.up1(x5, x4) 
        out = self.up2(out, x3) 
        out = self.up3(out, x2)
        out = self.up4(out, x1)  # Final feature map [N, base_ch, D, H, W]
        
        # Collect intermediate features for PKD loss
        features = [x2, x3, x4, out] if ret_intermediate else [out]
        
        # Main output
        logits = self.outc(out)
    
        logits_fake = None
        if fake_features is not None:
            logits_fake = self.fake_classifier(fake_features)  # [1, num_classes, N_fake, 1, 1]
        
        # Process extra background regions if provided
        logits_extra_bg = None
        if region_bg is not None and region_bg.sum() > 0:
         
            out_downsampled = out[:, :, ::16, ::16, ::16]  # Adjust stride as needed
            
            # Expand region_bg to match channel dimension
            region_bg_expanded = region_bg.unsqueeze(1)  # [N, 1, D_s, H_s, W_s]
            
            # Extract background features
            bg_features = out_downsampled * region_bg_expanded.float()  # [N, C, D_s, H_s, W_s]
            
            # Reshape for classification: [1, C, N_bg_pixels, 1, 1]
            N, C = bg_features.shape[:2]
            bg_features_flat = bg_features.permute(1, 0, 2, 3, 4)  # [C, N, D_s, H_s, W_s]
            bg_features_flat = bg_features_flat.reshape(C, -1)  # [C, N*D*H*W]
            
            # Only keep pixels where region_bg is True
            bg_mask = region_bg.flatten()  # [N*D*H*W]
            if bg_mask.sum() > 0:
                bg_features_selected = bg_features_flat[:, bg_mask]  # [C, N_bg_pixels]
                bg_features_selected = bg_features_selected.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, C, N_bg, 1, 1]
                logits_extra_bg = self.extra_bg_classifier(bg_features_selected)  # [1, num_classes, N_bg, 1, 1]
        
        if ret_intermediate:
            return logits, features, (logits_fake, logits_extra_bg)
        
        return logits

    def forward_fc(self, features):
        """Forward only through final classifier - for prototype loss"""
        return self.outc(features)

    def embedding(self, x):
        x = self.feature(x)
        x = self.avgpool_layer(x)
        x = x.permute(0, 1, 4, 3, 2)
        return x

    def freeze_bn(self, affine_freeze=False):
        """Freeze batch normalization layers"""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
                m.eval()
                if affine_freeze:
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
    
    def freeze_dropout(self):
        """Freeze dropout layers"""
        for m in self.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.eval()