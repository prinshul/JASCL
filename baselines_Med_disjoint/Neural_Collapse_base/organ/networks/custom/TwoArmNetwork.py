import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_utils import inconv, down_block, up_block
from .utils import get_block, get_norm
import pdb
import numpy as np
import math 
import random
from torch.nn.modules.loss import CrossEntropyLoss

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def generate_random_orthogonal_matrix(feat_in, num_classes):
    class_wise_means = torch.rand(feat_in, num_classes)
    orthogonal_matrix, _ = torch.qr(class_wise_means)
    # assert torch.allclose(torch.matmul(orthogonal_matrix.T, orthogonal_matrix), torch.eye(num_classes), atol=1.e-7), \
    #     "The max irregular value is : {}".format(
    #         torch.max(torch.abs(torch.matmul(orthogonal_matrix.T, orthogonal_matrix) - torch.eye(num_classes))))
   
    return orthogonal_matrix

class UNet(nn.Module):
    def __init__(self, in_ch, base_ch, scale=[[1,2,2], [2,2,2], [2,2,2], [2,2,2]], kernel_size=[[1,3,3], [2,3,3], [3,3,3], [3,3,3], [3,3,3]], num_classes=1, block='ConvNormAct', pool=True, norm='bn'):
        super().__init__()
        '''
        Args:
            in_ch: the num of input channel
            base_ch: the num of channels in the entry level
            scale: should be a list to indicate the downsample scale along each axis 
                in each level, e.g. [1, 1, 2, 2] such that all axis use the same scale
                or [[1,2,2], [2,2,2], [2,2,2], [2,2,2]] for difference scale on each axis
            kernel_size: the 3D kernel size of each level
                e.g. [3,3,3,3] or [[1,3,3], [1,3,3], [3,3,3], [3,3,3]]
            num_classes: the target class number
            block: 'ConvNormAct' for origin UNet, 'BasicBlock' for ResUNet
            pool: use maxpool or use strided conv for downsample
            norm: the norm layer type, bn or in

        '''

        num_block = 2 
        block = get_block(block)
        norm = get_norm(norm)
    
        self.inc = inconv(in_ch, base_ch, block=block, kernel_size=kernel_size[0], norm=norm)

        self.down1 = down_block(base_ch, 2*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[0], kernel_size=kernel_size[1], norm=norm)
        self.down2 = down_block(2*base_ch, 4*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[1], kernel_size=kernel_size[2], norm=norm)
        self.down3 = down_block(4*base_ch, 8*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[2], kernel_size=kernel_size[3], norm=norm)
        self.down4 = down_block(8*base_ch, 10*base_ch, num_block=num_block, block=block, pool=pool, down_scale=scale[3], kernel_size=kernel_size[4], norm=norm)

        self.up1 = up_block(10*base_ch, 8*base_ch, num_block=num_block, block=block, up_scale=scale[3], kernel_size=kernel_size[3], norm=norm)
        self.up2 = up_block(8*base_ch, 4*base_ch, num_block=num_block, block=block, up_scale=scale[2], kernel_size=kernel_size[2], norm=norm)
        self.up3 = up_block(4*base_ch, 2*base_ch, num_block=num_block, block=block, up_scale=scale[1], kernel_size=kernel_size[1], norm=norm)
        self.up4 = up_block(2*base_ch, base_ch, num_block=num_block, block=block, up_scale=scale[0], kernel_size=kernel_size[0], norm=norm)
        self.outc = nn.Conv3d(base_ch, num_classes, kernel_size=1)
        
    def forward(self, x): 
    
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # pixel_wise_embeddings = x5.clone()
        out = self.up1(x5, x4) 
        out = self.up2(out, x3) 
        out = self.up3(out, x2)
        out = self.up4(out, x1)
        pixel_wise_embeddings = out.clone()
        out = self.outc(out)

        return pixel_wise_embeddings, out

class ClassProtoCalculator(nn.Module):
    def __init__(self, feat_dim, num_classes, class_wise_embeddings_previous, step):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.step = step
        # self.class_wise_embeddings_total = class_wise_embeddings_previous
        if class_wise_embeddings_previous is not None:
            self.class_wise_embeddings_total = nn.Parameter(class_wise_embeddings_previous, requires_grad=False)
            self.num_iterations = 1
        else:
            self.class_wise_embeddings_total = nn.Parameter(torch.full((self.num_classes, self.feat_dim), 1e-10), requires_grad=False)
            self.num_iterations = 0
    def save_embeddings(self, filename):
        torch.save(self.class_wise_embeddings_total, filename)
        
    def forward(self, image_batch, label_batch, emb_batch, device, interpolate_label=False, return_all=True, background=False):

        batch_size, height, width, depth = label_batch.size()
        
        image_batch = image_batch.unsqueeze(1)
        label_batch = label_batch.unsqueeze(1)

        class_embeddings_total = []
        for batch in range(batch_size):
            # Get the segmentation mask for this batch
            mask_ = label_batch[batch]
            image_ = image_batch[batch]
            class_wise_embeddings_batch = []
            emb_ = emb_batch[batch] 
            classes = torch.unique(mask_)

            mask_ = mask_.squeeze(1)
            emb_ = emb_.unsqueeze(0)
            
            protos = []
            # bkg_proto = []
            for cl in range(self.num_classes):
                if cl in classes:
                    # print("Class:", cl)
                    if interpolate_label:  # to match output size
                        label_batch = F.interpolate(mask_.float().view(1, 1, mask_.shape[0], mask_.shape[1]),
                                            size=emb_.shape[-3:], mode="trilinear").view(emb.shape[-2:]).type(torch.uint8)
                    else:  # interpolate output to match label size
                        emb = F.interpolate(emb_, size=image_.shape[-3:], mode="trilinear", align_corners=False)
                    emb = emb.squeeze(0)
                    emb = emb.view(emb.shape[0], -1).t()  # (HxW) x F
                    mask = mask_.flatten()  # Now it is (HxW)
                    if (mask == cl).float().sum() > 0 :#and (mask != cl).float().sum() > 0:
                        protos.append(self.norm_mean(emb[mask == cl, :]).squeeze(0))
                        # bkg_proto.append(self.norm_mean(emb[mask != cl, :]))
                else:
                    # print(cl)
                    protos.append(self.class_wise_embeddings_total[cl,:])

             # Stack class-wise embeddings for this batch
            class_wise_embeddings_batch = torch.stack(protos)

            # Append class-wise embeddings for this batch to the main list
            class_embeddings_total.append(class_wise_embeddings_batch)
        
        # Stack class-wise embeddings for all batches
        class_embeddings_total = torch.stack(class_embeddings_total)
        class_embeddings_total = class_embeddings_total.mean(dim = 0)

        tmp = (self.class_wise_embeddings_total.data + class_embeddings_total.data)*0.5
        self.class_wise_embeddings_total.data =  tmp
        self.num_iterations += 1
        
        if self.num_iterations % 100 == 0:
            filename = 'cil/Neural_Collapse/organ/class_mean/class_mean_{}.pth'.format(self.step)
            self.save_embeddings(filename)
        
        return self.class_wise_embeddings_total

    def norm_mean(self, x):
        # x should be N x F, return 1 x F
        return F.normalize(x, dim=1).mean(dim=0, keepdim=True)

class ClassMeanCalculator(nn.Module):
    def __init__(self, feat_dim, num_classes, class_wise_embeddings_previous, step):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.step = step
        # self.class_wise_embeddings_total = class_wise_embeddings_previous
        if class_wise_embeddings_previous is not None:
            self.class_wise_embeddings_total = nn.Parameter(class_wise_embeddings_previous, requires_grad=False)
            self.num_iterations = 1
        else:
            self.class_wise_embeddings_total = nn.Parameter(torch.full((num_classes, feat_dim), 1e-10), requires_grad=False)
            self.num_iterations = 0
    
    def save_embeddings(self, filename):
        # To save the average, divide by the number of iterations
        # averaged_embeddings = self.class_wise_embeddings_total / self.num_iterations
        torch.save(self.class_wise_embeddings_total, filename)
        
    def forward(self, segmentation_mask, pixel_wise_embeddings):
        # Get the dimensions of the data
        batch_size, height, width, depth = segmentation_mask.size()
        num_classes = self.num_classes
        # Reshape pixel-wise embeddings to match the mask dimensions
        pixel_wise_embeddings = torch.mean(pixel_wise_embeddings, dim=1)

        # Create an empty tensor to store class-wise embeddings
        class_embeddings_total = []
        _, h, w, d, _ = pixel_wise_embeddings.shape
        # Loop through each batch
        for batch in range(batch_size):
            # Get the segmentation mask for this batch
            mask = segmentation_mask[batch]
            class_wise_embeddings_batch = []

            for c in range(num_classes):
                # Create a binary mask for this class
                if c in torch.unique(mask):
                    class_mask = (mask == c)
                    class_mask = class_mask.to(torch.float32)

                    class_mask = F.interpolate(class_mask.unsqueeze(0).unsqueeze(0), size=pixel_wise_embeddings[batch].shape[:-1], mode='nearest')
                    class_mask = class_mask.squeeze(0).squeeze(0)
                    class_mask = class_mask.unsqueeze(-1).expand(-1, -1, -1, self.feat_dim)
                
                    class_embedding = pixel_wise_embeddings[batch] * class_mask
                    
                    class_embedding = class_embedding.view(h* w* d, self.feat_dim)
                    class_embedding = class_embedding.sum(dim=0) / (class_mask.sum() + 1e-5)
                    class_wise_embeddings_batch.append(class_embedding)
                else:
                    class_wise_embeddings_batch.append(self.class_wise_embeddings_total[c,:])
                    
            # Stack class-wise embeddings for this batch
            class_wise_embeddings_batch = torch.stack(class_wise_embeddings_batch)

            # Append class-wise embeddings for this batch to the main list
            class_embeddings_total.append(class_wise_embeddings_batch)

        # Stack class-wise embeddings for all batches
        class_embeddings_total = torch.stack(class_embeddings_total)
        class_embeddings_total = class_embeddings_total.mean(dim = 0)

        tmp = (self.class_wise_embeddings_total.data + class_embeddings_total.data)*0.5
        self.class_wise_embeddings_total.data =  tmp
        self.num_iterations += 1

        if self.num_iterations % 100 == 0:
            filename = 'cil/Neural_Collapse/organ/class_mean/class_mean_{}.pth'.format(self.step)
            self.save_embeddings(filename)
        # self.class_wise_embeddings_total = (self.class_wise_embeddings_total + class_embeddings_total)/2
        return self.class_wise_embeddings_total
    
class TwoArmNetwork(nn.Module):
    def __init__(self, in_ch, base_ch, class_wise_embeddings_previous, step, scale=[[1,2,2], [2,2,2], [2,2,2], [2,2,2]], kernel_size=[[1,3,3], [2,3,3], [3,3,3], [3,3,3], [3,3,3]], num_classes=1, block='ConvNormAct', pool=True, norm='bn'):
        super().__init__()

        # Define the UNet encoder
        # self.encoder = UNet(in_ch, base_ch, scale, kernel_size, num_classes, block, pool, norm)
        self.num_classes = 27
        self.feat_dim = 32
        self.step = step
        # Define the class mean calculator
        self.etf_classifier = ETFHead(self.step, self.num_classes, self.feat_dim)
        # self.linear_layer = PixelEmbeddingCNN(320, 6, 6, 6, self.feat_dim)
        self.class_mean_calculator = ClassProtoCalculator(self.feat_dim, self.num_classes, class_wise_embeddings_previous, step)

    def forward(self, image, label, patch_embeddings):
        # patch_embeddings, _ = self.encoder(image)
        batch_size, channels, h, w, d = patch_embeddings.size()        
    
        # Calculate class-wise means from the segmentation mask and embeddings
        class_wise_means = self.class_mean_calculator(image, label, patch_embeddings, 'cuda', interpolate_label=False, return_all=True, background=False)
        # print(class_wise_means)
        loss_etf1, loss_etf2 = self.etf_classifier(class_wise_means, label)
        return class_wise_means, loss_etf1, loss_etf2

# Define a simple CNN model for pixel-level embeddings
class PixelEmbeddingCNN(nn.Module):
    def __init__(self, channels, h, w, d, ndim):
        super().__init__()
        self.conv1_p = nn.Conv3d(320, 8, kernel_size=3, padding=1)
        # self.conv2_p = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        # self.conv3_p = nn.Conv3d(64, 8, kernel_size=3, padding=1)
        self.global_avg_pool_p = nn.AdaptiveAvgPool3d(h)
        self.fc_p = nn.Linear(8* h * w * d, 8 * h * w * d * ndim)

    def forward(self, x):
        x = self.conv1_p(x)
        # x = self.conv2_p(x)
        # x = self.conv3_p(x)
        x = self.global_avg_pool_p(x)
        x = x.view(x.size(0), -1)  
        x = self.fc_p(x)
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x


class ETFHead(nn.Module):
    def __init__(self, step, num_classes, feat_in):
        super().__init__()
        self.num_classes = num_classes
        self.feat_in = feat_in
        self.step = step
        if self.step == 0:
            self.orth_vec = generate_random_orthogonal_matrix(self.feat_in, self.num_classes)
            torch.save(self.orth_vec, 'cil/Neural_Collapse/organ/ortho_vec/orthovec_{}.pth'.format(self.step))
        else:
            self.orth_vec = torch.load('cil/Neural_Collapse/organ/ortho_vec/orthovec_0.pth')
        #     assert torch.allclose(torch.matmul(self.orth_vec.T, self.orth_vec), torch.eye(num_classes), atol=1.e-7), \
        # "The max irregular value is : {}".format(
        #     torch.max(torch.abs(torch.matmul(orthogonal_matrix.T, orthogonal_matrix) - torch.eye(num_classes))))
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc = torch.mul(torch.ones(self.num_classes, self.num_classes), (1 / self.num_classes))
        self.etf_vec = torch.mul(torch.matmul(self.orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(self.num_classes / (self.num_classes - 1)))
        torch.save(self.etf_vec, 'cil/Neural_Collapse/organ/ortho_vec/etf_vec.pth')
        self.loss = DRLoss()
    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x
    
    def forward(self, x, gt_label):
        """Forward training data."""
        x = self.pre_logits(x)
        x = x.view(self.num_classes , -1)
        losses = 0
        loss1 = self.loss(x.T.cuda(), self.etf_vec.cuda())
        if self.step > 0:
            for c in torch.unique(gt_label):
                target = self.etf_vec[:, c]
                inp = x[c, :]
                inp = inp.unsqueeze(0)
                target = target.unsqueeze(0)
                loss2 = self.loss(inp.cuda(), target.cuda())
                losses += loss2
        else:
            losses = None
        return loss1, losses
    
class DRLoss(nn.Module):
    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 reg_lambda=0.
                 ):
        super().__init__()

        self.reduction = reduction
        self.loss_weight = loss_weight
        self.reg_lambda = reg_lambda

    def forward(
            self,
            feat,
            target,
            h_norm2=None,
            m_norm2=None,
            avg_factor=None,
    ):
        # final_loss = torch.sqrt(torch.mean((feat - target) ** 2))
        assert avg_factor is None
        y = feat * target
        
        dot = torch.sum(feat * target, dim=1)
        if h_norm2 is None:
            h_norm2 = torch.ones_like(dot)
        if m_norm2 is None:
            m_norm2 = torch.ones_like(dot)
        # dot[torch.isnan(dot)] = 0
        loss = 0.5 * torch.mean(((dot - (m_norm2 * h_norm2)) ** 2) / h_norm2)
        final_loss = loss * self.loss_weight
        # # final_loss = final_loss.detach()  # Detach to remove the old graph
        # # final_loss.requires_grad = True
        return final_loss
        