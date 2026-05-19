import os
import torch
import argparse
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
parser = argparse.ArgumentParser()

parser.add_argument("ckpt", type=str,
                    help="The checkpoint to convert")

args = parser.parse_args()
ckpt_name = args.ckpt
ckpt = torch.load(ckpt_name, map_location="cpu")

state_dict = ckpt['model_state']

new_state = {}
for k in state_dict:
    new_state["module."+k] = state_dict[k]

print(new_state.keys())
ckpt['model_state'] = new_state
torch.save(ckpt, ckpt_name)

