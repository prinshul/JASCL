import utils
import torch
import torch.nn.functional as F
import torch.nn as nn

import random
import numpy as np

random.seed(1024)
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class MeanReduction:
    def __call__(self, x, target):
        x = x[target != 255]
        return x.mean()


def get_scheduler(opts, optim):
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optim, max_iters=opts.max_iter, power=opts.lr_power)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=opts.lr_decay_step,
                                                    gamma=opts.lr_decay_factor)
    else:
        raise NotImplementedError
    return scheduler


def get_batch(it, dataloader):
    try:
        batch = next(it)
    except StopIteration:
        # restart the generator if the previous generator is exhausted.
        it = iter(dataloader)
        batch = next(it)
    return it, batch


def get_prototype(model, ds, cl, device, interpolate_label=True, return_all=False, background=False):
    protos = []
    bkg_proto = []
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(ds):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            
            image_batch = image_batch.unsqueeze(1)
            label_batch = label_batch.unsqueeze(1)
            
            # x = label_batch.float().view(1, 1, label_batch.shape[0], label_batch.shape[1])
            # print(x.shape)
            print("Getting the embeddings")
            emb, logits = model(image_batch)
            #print(emb)
            if interpolate_label:  # to match output size
                label_batch = F.interpolate(label_batch.float().view(1, 1, label_batch.shape[0], label_batch.shape[1]),
                                    size=emb.shape[-3:], mode="trilinear").view(emb.shape[-2:]).type(torch.uint8)
            else:  # interpolate output to match label size
                emb = F.interpolate(emb, size=image_batch.shape[-3:], mode="trilinear", align_corners=False)
            emb = emb.squeeze(0)
            emb = emb.view(emb.shape[0], -1).t()  # (HxW) x F
            label_batch = label_batch.flatten()  # Now it is (HxW)
            if (label_batch == cl).float().sum() > 0 and (label_batch != cl).float().sum() > 0:
                protos.append(norm_mean(emb[label_batch == cl, :]))
                bkg_proto.append(norm_mean(emb[label_batch != cl, :]))
    if len(protos) > 0:
        protos = torch.cat(protos, dim=0)
        bkg_proto = torch.cat(bkg_proto, dim=0)
        if return_all:
            return protos if not background else (protos, bkg_proto)
        return protos.mean(dim=0) if not background else (protos.mean(dim=0), bkg_proto.mean(dim=0))
    else:
        return None



def norm_mean(x):
    # x should be N x F, return 1 x F
    return F.normalize(x, dim=1).mean(dim=0, keepdim=True)


class myReLU(torch.nn.Module):
    __constants__ = ['inplace']

    def __init__(self, inchannels=None, inplace=False):
        super().__init__()
        self.inplace = inplace

    def forward(self, input):
        return F.relu(input, inplace=self.inplace)

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str
