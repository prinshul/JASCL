from .segmentation_module import make_model
from .trainer import Trainer
from .imprinting import *
import random
import numpy as np
import torch
random.seed(1024)
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

methods = {"FT", "SPN", "COS", "WI", 'DWI', "AMP",
           "PIFS", "LWF", "MIB", "ILT", "RT"}


def get_method(opts, task, device, logger, config_vit):
    if opts.method == 'WI':
        opts.method = 'COS'
        return WeightImprinting(task=task, device=device, logger=logger, opts=opts, config_vit = config_vit)
    elif opts.method == 'DWI':
        opts.method = 'COS'
        return DynamicWI(task=task, device=device, logger=logger, opts=opts)
    elif opts.method == 'AMP':
        opts.method = 'FT'
        return AMP(task=task, device=device, logger=logger, opts=opts)
    else:
        trainer = Trainer(task=task, device=device, logger=logger, opts=opts, config_vit = config_vit)
        return trainer
