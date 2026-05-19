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
# from networks.custom.unet import UNet as network
from networks.custom.TwoArmNetwork import UNet as network
from networks.custom.TwoArmNetwork import TwoArmNetwork as network_twoarm


def make_model(opts, config_vit, n_classes, class_wise_embeddings_previous):
    if opts.norm_act == 'abr':
        norm = partial(ABR, activation="relu")
    if opts.network_arch == 'shape_prior':    
        model = network(norm_act = norm, in_channel=3, out_channel=opts.num_classes, training=False, config=config_vit).cuda()
    elif opts.network_arch == 'newUNET':
        model = network(in_ch=1, base_ch=32, num_classes=opts.num_classes, block = 'SingleConv')#.cuda()
        etf_model = network_twoarm(in_ch=1, base_ch=32, class_wise_embeddings_previous = class_wise_embeddings_previous, step = opts.step, num_classes=opts.num_classes, block = 'SingleConv')#.cuda()
    elif opts.network_arch == 'SwinUNETR':
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=opts.num_classes,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
        ).cuda()
    return model, etf_model
