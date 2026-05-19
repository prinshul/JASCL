# pylint: disable=arguments-differ, too-many-arguments
''' DeepLab V3+ '''

import torch
import torch.nn as nn
import torch.nn.functional as nn_func

from models.deeplabv3 import nn_ext
from models.deeplabv3.backbone import BackboneModule
import torch.nn.functional as F

import sys
from shared_quant import seed_current
import random
import numpy as np

random.seed(seed_current)
np.random.seed(seed_current)
torch.manual_seed(seed_current)
torch.cuda.manual_seed(seed_current)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class ProbabilisticClassifier(nn.Module):

    def __init__(self, num_features, num_classes, kernel_size=3, padding=1):
        super().__init__()
        torch.manual_seed(1024)
        torch.cuda.manual_seed(1024)
        self.mu = nn.Conv2d(num_features, num_classes, kernel_size=3, padding=1, bias=False)
        self.sigma = nn.Conv2d(num_features, num_classes, kernel_size=3, padding=1, bias=False)
        self.grad_update = nn.Parameter(torch.zeros(self.mu.weight.shape))
        self.temp = 10
        self._init_weights()
        print('conv2d Stochastic Classifier')
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.modules.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            
    
    def forward(self, x, stochastic=True):
        mu = self.mu.weight
        sigma = self.sigma.weight

        inverse_grad = self.grad_update.data
        #print("grad inverse_grad ",inverse_grad)
        epsilon = 1e-8
        inverse_grad = 1 / (inverse_grad + epsilon)
        #print("inverse_grad ",inverse_grad)
        min_val = inverse_grad.min()
        max_val = inverse_grad.max()
        #if max_val > min_val:
        inverse_grad = (1 + inverse_grad - min_val) / (1+(max_val - min_val))
        #else:
        #    inverse_grad = torch.ones_like(inverse_grad)
        #print("standard inverse_grad ",inverse_grad)
        if stochastic:
            weight = inverse_grad * torch.randn_like(mu) + mu
        else:
            weight = mu
        
        weight = F.normalize(weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        score = F.conv2d(x, weight)
        score = score*self.temp

        return score
    

'''


class ProbabilisticClassifier(nn.Module):

    def __init__(self, num_features, num_classes, kernel_size=3, padding=1):
        super().__init__()
        torch.manual_seed(1024)
        torch.cuda.manual_seed(1024)
        self.mu = nn.Conv2d(num_features, num_classes, kernel_size=3, padding=1, bias=False)
        self.sigma = nn.Conv2d(num_features, num_classes, kernel_size=3, padding=1, bias=False)
        self.temp = 10
        self._init_weights()
        print('conv2d Stochastic Classifier')
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, torch.nn.modules.Linear):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            
    
    def forward(self, x, stochastic=True):
        mu = self.mu.weight
        sigma = self.sigma.weight

        if stochastic:
            sigma = F.softplus(sigma - 4) # when sigma=0, softplus(sigma-4)=0.0181
            weight = sigma * torch.randn_like(mu) + mu
        else:
            weight = mu
        
        weight = F.normalize(weight, p=2, dim=1)
        x = F.normalize(x, p=2, dim=1)

        score = F.conv2d(x, weight)
        score = score*self.temp

        return score

'''



class DeepLab(nn.Module):
    ''' DeepLab V3+ module '''

    class ASPP(nn.Module):
        ''' Atrous spatial pyramid pooling module '''

        def __init__(self, in_channels, output_stride=16):
            super().__init__()

            if output_stride not in {8, 16}:
                raise ValueError('Invalid output_stride; Supported values: {8, 16}')
            dilation_factor = 1 if output_stride == 16 else 2

            self.aspp = nn.ModuleList([
                nn_ext.Conv2dLayer(in_channels, 256, kernel_size=1, dilation=1),
                nn_ext.Conv2dLayer(in_channels, 256, kernel_size=3,
                                   dilation=6 * dilation_factor, padding=6 * dilation_factor),
                nn_ext.Conv2dLayer(in_channels, 256, kernel_size=3,
                                   dilation=12 * dilation_factor, padding=12 * dilation_factor),
                nn_ext.Conv2dLayer(in_channels, 256, kernel_size=3,
                                   dilation=18 * dilation_factor, padding=18 * dilation_factor)])

            self.global_avg_pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Conv2d(in_channels, 256, kernel_size=1),
                nn.ReLU(inplace=True))

            self.output_conv = nn_ext.Conv2dLayer(256 * 4 + 256, 256, kernel_size=1)

        def forward(self, x):
            x_aspp = (aspp(x) for aspp in self.aspp)
            x_pool = self.global_avg_pool(x)
            x_pool = nn_func.interpolate(x_pool, size=x.shape[2:4])
            feats = torch.cat((*x_aspp, x_pool), dim=1)
            feats = self.output_conv(feats)
            return feats

    class Decoder(nn.Module):
        ''' DeepLab V3+ decoder module '''

        def __init__(self, low_in_channels, num_classes):
            super().__init__()

            self.conv_low = nn_ext.Conv2dLayer(low_in_channels, 48, kernel_size=1)
            if num_classes==11:
                self.conv_logit = nn.Conv2d(48 + 256, num_classes, kernel_size=3, padding=1)
            else:
                self.conv_logit = ProbabilisticClassifier(48 + 256, num_classes, kernel_size=3, padding=1)
            
        def forward(self, feats, low_feats):
            low_feats = self.conv_low(low_feats)
            feats = nn_func.interpolate(feats, size=low_feats.shape[2:4],
                                        mode='bilinear', align_corners=True)
            feats = torch.cat((feats, low_feats), dim=1)
            #print("feats from decoder ",feats.shape)
            logits = self.conv_logit(feats)
            return logits, feats

    def __init__(self, backbone, num_classes):
        super().__init__()
        if not isinstance(backbone, BackboneModule):
            raise RuntimeError('Backbone must extend model.backbone.BackboneModue')

        self.backbone = backbone
        self.aspp = DeepLab.ASPP(in_channels=backbone.out_channels,
                                 output_stride=backbone.output_stride)
        self.decoder = DeepLab.Decoder(low_in_channels=backbone.low_out_channels,
                                       num_classes=num_classes)

        self._init_weights()

    def forward(self, x_in):
        x, x_low = self.backbone(x_in)
        x = self.aspp(x)
        #fts = [x, x_low]
        logits, fts = self.decoder(x, x_low)
        logits = nn_func.interpolate(logits, size=x_in.shape[2:4],
                                     mode='bilinear', align_corners=True)
        return logits, fts

    def _init_weights(self):
        ''' Initializes weights of the model.
            - Conv2d parameters initialized using Kaiming normal
            - Batchnorm affine parameters initialized as Identity
        '''
        for module in self.modules():
            if isinstance(module, torch.nn.modules.Conv2d):
                torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            elif isinstance(module, torch.nn.modules.BatchNorm2d):
                torch.nn.init.constant_(module.weight, 1.0)
                torch.nn.init.constant_(module.bias, 0.0)

if __name__ == '__main__':

    from model.backbone import Xception

    def test_out_shapes(model, in_shape, out_shape):
        ''' Model shape test '''
        x = torch.rand(*in_shape)
        if next(model.parameters()).is_cuda:
            x = x.cuda()
        with torch.no_grad():
            y = model.forward(x)
        assert y.shape == out_shape, 'Output size mismatch! Expected: %s, Actual: %s' % (
            out_shape, y.shape)

    def fps(model, in_shape):
        ''' Model FPS test '''
        x = torch.rand(*in_shape)
        if next(model.parameters()).is_cuda:
            x = x.cuda()

        import timeit
        with torch.no_grad():
            duration = timeit.timeit(lambda: model.forward(x), number=100)
        return in_shape[0] * 100 / duration

    deeplab = DeepLab(Xception(output_stride=16), num_classes=20)
    if torch.cuda.is_available():
        deeplab.cuda()
    test_out_shapes(deeplab, (1, 3, 1280, 720), (1, 20, 1280, 720))
    test_out_shapes(deeplab, (1, 3, 640, 360), (1, 20, 640, 360))

    deeplab = DeepLab(Xception(output_stride=8), num_classes=20)
    if torch.cuda.is_available():
        deeplab.cuda()
    test_out_shapes(deeplab, (1, 3, 1280, 720), (1, 20, 1280, 720))
    test_out_shapes(deeplab, (1, 3, 640, 360), (1, 20, 640, 360))

    deeplab = DeepLab(Xception(output_stride=16), num_classes=20)
    if torch.cuda.is_available():
        deeplab.cuda()
    print('FPS (Xception, out_stride=16, size=(1280, 720))', fps(deeplab, (1, 3, 1280, 720)))
    print('FPS (Xception, out_stride=16, size=(640, 360))', fps(deeplab, (1, 3, 640, 360)))

    deeplab = DeepLab(Xception(output_stride=8), num_classes=20)
    if torch.cuda.is_available():
        deeplab.cuda()
    print('FPS (Xception, out_stride=8, size=(1280, 720))', fps(deeplab, (1, 3, 1280, 720)))
    print('FPS (Xception, out_stride=8, size=(640, 360))', fps(deeplab, (1, 3, 640, 360)))
