from torch.optim.lr_scheduler import _LRScheduler, StepLR

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
class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1):
        self.power = power
        self.max_iters = max_iters
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        return [ base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power
                for base_lr in self.base_lrs]