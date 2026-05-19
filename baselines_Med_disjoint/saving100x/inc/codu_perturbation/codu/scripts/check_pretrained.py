import torch
import sys
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
print("Elaborating checkpoint in: " + sys.argv[1])
ckpt = torch.load(sys.argv[1])

print(ckpt['epoch'])

for k,v in ckpt['trainer_state']['regularizer']['score'].items():
    if torch.isnan(v.sum()) or torch.isinf(v.sum()):
        print("score " + k)

if "RW" in sys.argv[1]:
    for k,v in ckpt['trainer_state']['regularizer']['fisher'].items():
        if torch.isnan(v.sum()) or torch.isinf(v.sum()):
            print("fisher " + k)
