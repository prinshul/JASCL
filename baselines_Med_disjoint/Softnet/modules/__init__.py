from .deeplab import DeeplabV3, DeeplabV2
from .dense import DenseModule
from .misc import GlobalAvgPool2d, SingleGPU
from .residual import IdentityResidualBlock, ResidualBlock
import torch
import random
import numpy as np
random.seed(1024)
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True