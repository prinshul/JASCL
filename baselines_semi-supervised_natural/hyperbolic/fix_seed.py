from torch.nn.init import kaiming_uniform_
from torch.nn.parameter import Parameter
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt.manifolds.stereographic.math as gmath

import matplotlib.pyplot as plt
import json

def median_frequency_balance(dataset, num_classes, ignore_index=255, _eps=1e-5):
    '''
    For more details refer to Section 6.3.2 in
    https://arxiv.org/pdf/1411.4734.pdf
    '''
    frequency = torch.zeros(num_classes) + _eps
    for _, seg in dataset:
        for cid in torch.unique(seg):
            if cid == ignore_index:
                continue
            frequency[cid] += torch.sum(seg == cid)
    frequency /= torch.sum(frequency)
    return torch.median(frequency) / frequency


class UnbiasedCrossEntropy(nn.Module):
    def __init__(self, old_cl=None, reduction='mean', ignore_index=255):
        super().__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.old_cl = old_cl

    def forward(self, inputs, targets):

        old_cl = self.old_cl
        outputs = torch.zeros_like(inputs)  # B, C (1+V+N), H, W
        den = torch.logsumexp(inputs, dim=1)                               # B, H, W       den of softmax
        outputs[:, 0] = torch.logsumexp(inputs[:, 0:old_cl], dim=1) - den  # B, H, W       p(O)
        outputs[:, old_cl:] = inputs[:, old_cl:] - den.unsqueeze(dim=1)    # B, N, H, W    p(N_i)

        labels = targets.clone()    # B, H, W
        labels[targets < old_cl] = 0  # just to be sure that all labels old belongs to zero

        loss = F.nll_loss(outputs, labels, ignore_index=self.ignore_index, reduction=self.reduction)

        return loss
        
        
class UnbiasedKnowledgeDistillationLoss(nn.Module):
    def __init__(self, reduction='mean', alpha=1.):
        super().__init__()
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets, mask=None):

        new_cl = inputs.shape[1] - targets.shape[1]

        targets = targets * self.alpha

        new_bkg_idx = torch.tensor([0] + [x for x in range(targets.shape[1], inputs.shape[1])]).to(inputs.device)

        den = torch.logsumexp(inputs, dim=1)                          # B, H, W
        outputs_no_bgk = inputs[:, 1:-new_cl] - den.unsqueeze(dim=1)  # B, OLD_CL, H, W
        outputs_bkg = torch.logsumexp(torch.index_select(inputs, index=new_bkg_idx, dim=1), dim=1) - den     # B, H, W

        labels = torch.softmax(targets, dim=1)                        # B, BKG + OLD_CL, H, W

        # make the average on the classes 1/n_cl \sum{c=1..n_cl} L_c
        loss = (labels[:, 0] * outputs_bkg + (labels[:, 1:] * outputs_no_bgk).sum(dim=1)) / targets.shape[1]

        if mask is not None:
            loss = loss * mask.float()

        if self.reduction == 'mean':
                outputs = -torch.mean(loss)
        elif self.reduction == 'sum':
                outputs = -torch.sum(loss)
        else:
            outputs = -loss

        return outputs
        
        

def compute_pixel_entropy(p,num_classes):
    pixel_entropy = torch.sum(-p * torch.log(p + 1e-6), dim=1).unsqueeze(
        dim=1
    ) / math.log(num_classes)
    return pixel_entropy       


PROJ_EPS = 1e-3


class HyperMapper(object):
    """A class to map between euclidean and hyperbolic space and compute distances."""

    def __init__(self, c=1.) -> None:
        """Initialize the hyperbolic mapper.

        Args:
            c (float, optional): Hyperbolic curvature. Defaults to 1.0
        """
        self.c = c
        self.K = torch.tensor(-self.c, dtype=float)

    def expmap(self, x, dim=-1):
        """Exponential mapping from Euclidean to hyperbolic space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (..., d)
        """
        x_hyp = gmath.expmap0(x.double(), k=self.K, dim=dim)
        x_hyp = gmath.project(x_hyp, k=self.K, dim=dim)
        return x_hyp

    def expmap2(self, inputs, dim=-1):
        PROJ_EPS = 1e-3
        EPS = 1e-15
        sqrt_c = torch.sqrt(torch.abs(self.K))
        inputs = inputs + EPS    # protect div b 0
        norm = torch.norm(inputs, dim=dim)  
        gamma = torch.tanh(sqrt_c * norm) / (sqrt_c * norm)  # sh ncls
        scaled_inputs = gamma.unsqueeze(dim) * inputs
        return gmath.project(scaled_inputs, k=self.K, dim=dim, eps=PROJ_EPS)

    def logmap(self, x):
        """Logarithmic mapping from hyperbolic to Euclidean space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (..., d)
        """
        return gmath.project(gmath.logmap0(x.double(), k=self.K), k=self.K)

    def poincare_distance(self, x, y):
        """Poincare distance between two points in hyperbolic space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)
            y (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (...)
        """
        return gmath.dist(x, y, k=self.K)
    
    def poincare_distance_origin(self, x, dim=-1):
        """Poincare distance between two points in hyperbolic space.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (...)
        """
        return gmath.dist0(x, k=self.K, dim=dim)

    def cosine_distance(self, x, y):
        """Cosine distance between two points.

        Args:
            x (torch.Tensor): Tensor of shape (..., d)
            y (torch.Tensor): Tensor of shape (..., d)

        Returns:
            torch.Tensor: Tensor of shape (...)
        """
        x = F.normalize(x, dim=-1, p=2)
        y = F.normalize(y, dim=-1, p=2)
        return 2 - 2 * (x * y).sum(dim=-1)       



    