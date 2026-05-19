# pylint: disable=W0221,C0414

import torch.nn as nn

import sys
from shared_quant import seed_current
import random, torch
import numpy as np

random.seed(seed_current)
np.random.seed(seed_current)
torch.manual_seed(seed_current)
torch.cuda.manual_seed(seed_current)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class BackboneModule(nn.Module):
    ''' Base class for all DeepLab feature extractor backbones '''

    def __init__(self, output_stride, out_channels, low_out_channels):
        super().__init__()
        self.output_stride = output_stride
        self.out_channels = out_channels
        self.low_out_channels = low_out_channels

    def forward(self, x):
        raise NotImplementedError
