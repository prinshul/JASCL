import torch
import torch.nn as nn
import torch.nn.functional as functional
from monai.networks.nets import SwinUNETR
from modules.custom_bn import ABR

# import inplace_abn
# from inplace_abn import InPlaceABNSync, InPlaceABN, ABN
# from modules.custom_bn import AIN, RandABN, RandInPlaceABNSync, ABR, InPlaceABR, InPlaceABR_R
from functools import partial

import models
from modules import DeeplabV3, DeeplabV2
# from networks.VerSe.unet import network as network
from networks.custom.UnetMDIL import UnetMDIL as network

def make_model(opts, config_vit, n_classes):
    if opts.norm_act == 'abr':
        norm = partial(ABR, activation="relu")
    if opts.network_arch == 'shape_prior':    
        model = network(norm_act = norm, in_channel=3, out_channel=opts.num_classes, training=False, config=config_vit).cuda()
    elif opts.network_arch == 'erfnet_RA_parallel':
        # model = network(in_ch=1, base_ch=32, num_classes=n_classes, block = 'SingleConv').cuda()
        model = network(opts, n_classes).cuda()
    elif opts.network_arch == 'SwinUNETR':
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=n_classes,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        ).cuda()
    return model

def get_any_model(network_arch, num_classes):
    if network_arch == 'newUNET':
        model = network(in_ch=1, base_ch=32, num_classes=num_classes, block = 'SingleConv').cuda()
    return model
    
def get_old_model(opts, num_classes):
    if opts.network_arch == 'newUNET':
        model = network(in_ch=1, base_ch=32, num_classes=num_classes, block = 'SingleConv').cuda()

    return model


class SegmentationModule(nn.Module):

    def __init__(self, body, head, head_channels, classifier):
        super(SegmentationModule, self).__init__()
        self.body = body
        self.head = head
        self.head_channels = head_channels
        self.cls = classifier

    def forward(self, x, use_classifier=True, return_feat=False, return_body=False,
                only_classifier=False, only_head=False):

        if only_classifier:
            return self.cls(x)
        elif only_head:
            return self.cls(self.head(x))
        else:
            x_b = self.body(x)
            if isinstance(x_b, dict):
                x_b = x_b["out"]
            out = self.head(x_b)

            out_size = x.shape[-2:]

            if use_classifier:
                sem_logits = self.cls(out)
                sem_logits = functional.interpolate(sem_logits, size=out_size, mode="bilinear", align_corners=False)
            else:
                sem_logits = out

            if return_feat:
                if return_body:
                    return sem_logits, out, x_b
                return sem_logits, out

            return sem_logits

    def freeze(self):
        for par in self.parameters():
            par.requires_grad = False

    def fix_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, inplace_abn.ABN):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def bn_set_momentum(self, momentum=0.0):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, ABN) or isinstance(m, AIN) or isinstance(m, ABR):
                m.momentum = momentum
