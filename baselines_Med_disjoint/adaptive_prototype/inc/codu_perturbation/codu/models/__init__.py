from .densenet import *
from .resnet import *
from .resnext import *
from .wider_resnet import *
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
