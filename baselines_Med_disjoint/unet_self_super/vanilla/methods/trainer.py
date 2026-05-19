import torch
from torch import distributed
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from utils.loss import KnowledgeDistillationLoss, CosineLoss, \
    UnbiasedKnowledgeDistillationLoss, UnbiasedCrossEntropy, CosineKnowledgeDistillationLoss
from .segmentation_module import make_model
from modules.classifier import IncrementalClassifier, CosineClassifier, SPNetClassifier
from .utils import get_scheduler, MeanReduction
from networks.VerSe.unet import network as network
from torch.nn.modules.loss import CrossEntropyLoss
from utils.VerSe_utils import DiceLoss, print_network
from scipy.ndimage import rotate, shift, zoom, map_coordinates, gaussian_filter
from scipy import ndimage
import SimpleITK as sitk
import numpy as np
import os
import random
import sys
CLIP = 10

random.seed(1024)
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

def random_rotation(image, label, angle_range=(-10, 10)):
    angle = random.uniform(*angle_range)
    axes = random.choice([(0, 1), (0, 2), (1, 2)])
    image = torch.tensor(rotate(image.numpy(), angle, axes=axes, reshape=False, mode='nearest'))
    label = torch.tensor(rotate(label.numpy(), angle, axes=axes, reshape=False, mode='nearest'))
    return image, label

def random_translation(image, label, max_shift=10):
    shift_values = [random.uniform(-max_shift, max_shift) for _ in range(3)]
    image = torch.tensor(shift(image.numpy(), shift_values, mode='nearest'))
    label = torch.tensor(shift(label.numpy(), shift_values, mode='nearest'))
    return image, label

def random_scaling(image, label, scale_range=(0.9, 1.1)):
    _,w,h,d = image.shape
    scale_factor = random.uniform(*scale_range)
    image = torch.tensor(zoom(image.numpy(), scale_factor, mode='nearest'))
    label = torch.tensor(zoom(label.numpy(), scale_factor, mode='nearest'))
    
    
    return image[:w,:h,:d], label[:w,:h,:d]

def random_flip(image, label):
    #axis = random.choice([0, 1, 2])
    axis = 2 
    image = torch.flip(image, [axis])
    label = torch.flip(label, [axis])
    return image, label
    
def add_noise(image, label, noise_level=0.01):
    noise = noise_level * torch.randn_like(image)
    return image + noise, label

def adjust_contrast(image, label, factor):
    mean = torch.mean(image)
    return (image - mean) * factor + mean, label

def intensity_scaling(image, label, factor):
    return image * factor, label

def random_resized_crop(image, label, scale):
    output_size = image.shape
    image, label = image.squeeze(0).squeeze(0), label.squeeze(0).squeeze(0)
    depth, height, width = image.shape
    
    
    # Calculate crop size
    crop_scale = random.uniform(*scale)
    crop_depth = int(crop_scale * depth)
    crop_height = int(crop_scale * height)
    crop_width = int(crop_scale * width)

    # Ensure the crop size does not exceed the image dimensions
    crop_depth = min(crop_depth, depth)
    crop_height = min(crop_height, height)
    crop_width = min(crop_width, width)

    # Randomly select the top-left corner of the crop
    top = random.randint(0, depth - crop_depth)
    left = random.randint(0, height - crop_height)
    front = random.randint(0, width - crop_width)

    # Crop the image (and label if provided)
    cropped_image = image[top:top + crop_depth, left:left + crop_height, front:front + crop_width]
    
    if label is not None:
        cropped_label = label[top:top + crop_depth, left:left + crop_height, front:front + crop_width]

    # Resize the cropped image (and label if provided) to the desired output size
    cropped_image = F.interpolate(cropped_image.unsqueeze(0).unsqueeze(0), size=output_size, mode='trilinear', align_corners=False).squeeze(0).squeeze(0)
    
    if label is not None:
        cropped_label = F.interpolate(cropped_label.unsqueeze(0).unsqueeze(0), size=output_size, mode='nearest').squeeze(0).squeeze(0)
        return cropped_image.unsqueeze(0), cropped_label.unsqueeze(0)
    
    return cropped_image.unsqueeze(0)
    
def apply_augmentation(image, label, augmentations):
    for aug in augmentations:
        if aug['type'] == 'rotate':
            image, label = random_rotation(image, label, aug.get('angle_range', (-10, 10)))
        elif aug['type'] == 'translate':
            image, label = random_translation(image, label, aug.get('max_shift', 10))
        elif aug['type'] == 'scale':
            image, label = random_scaling(image, label, aug.get('scale_range', (0.9, 1.1)))
        elif aug['type'] == 'flip':
            image, label = random_flip(image, label)
        elif aug['type'] == 'add_noise':
            image, label = add_noise(image, label, aug.get('noise_level', 0.01))
        elif aug['type'] == 'elastic_deformation':
            image, label = elastic_deformation(image, label, aug.get('alpha', 1), aug.get('sigma', 0.2))
        elif aug['type'] == 'adjust_contrast':
            image, label = adjust_contrast(image, label, aug.get('factor', 1.5))
        elif aug['type'] == 'intensity_scaling':
            image, label = intensity_scaling(image, label, aug.get('factor', 1.2))
        elif aug['type'] == 'random_crop':
            image, label = random_resized_crop(image, label, aug.get('scale',(0.8,1.0)))
    return image, label


class Trainer:
    def __init__(self, task, device, logger, opts, config_vit):
        self.logger = logger
        self.device = device
        self.task = task
        self.opts = opts
        self.novel_classes = self.task.get_n_classes()[-1]
        self.labels_old = task.get_old_labels(bkg=False)
       
        self.step = task.step        
        self.labels = self.task.get_novel_labels()
        self.total_classes = self.labels + self.labels_old
        self.need_model_old = (self.opts.born_again or self.opts.mib_kd > 0 or self.opts.loss_kd > 0 or
                               self.opts.l2_loss > 0 or self.opts.l1_loss > 0 or self.opts.cos_loss > 0)

        self.n_channels = -1  # features size, will be initialized in make model
        self.model = self.make_model(config_vit)
        self.model = self.model.to(device)

        self.distributed = False
        self.model_old = None

        if self.opts.fix_bn:
            self.model.fix_bn()

        if self.opts.bn_momentum is not None:
            self.model.bn_set_momentum(self.opts.bn_momentum)

        self.initialize(self.opts)  # initialize model parameters (e.g. perform WI)

        self.born_again = self.opts.born_again
        self.dist_warm_start = self.opts.dist_warm_start
        model_old_as_new = self.opts.born_again or self.opts.dist_warm_start
        if self.need_model_old:
            self.model_old = self.make_model(config_vit, is_old=not model_old_as_new)
            # put the old model into distributed memory and freeze it
            for par in self.model_old.parameters():
                par.requires_grad = False
            self.model_old.to(device)
            self.model_old.eval()

        # xxx Set up optimizer
        self.train_only_novel = self.opts.train_only_novel

        # self.optimizer = torch.optim.SGD(params, lr=self.opts.lr, momentum=0.9, weight_decay=self.opts.weight_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 25, eta_min=5e-6)
        self.logger.debug("Optimizer:\n%s" % self.optimizer)
       
       # Feature distillation
        if opts.l2_loss > 0 or opts.cos_loss > 0 or opts.l1_loss > 0:
            assert self.model_old is not None, "Error, model old is None but distillation specified"
            if opts.l2_loss > 0:
                self.feat_loss = opts.l2_loss
                self.feat_criterion = nn.MSELoss()
            elif opts.l1_loss > 0:
                self.feat_loss = opts.l1_loss
                self.feat_criterion = nn.L1Loss()
            elif opts.cos_loss > 0:
                self.feat_loss = opts.cos_loss
                self.feat_criterion = CosineLoss()
        else:
            self.feat_criterion = None
        
        # Output distillation
        if opts.loss_kd > 0 or opts.mib_kd > 0:
            assert self.model_old is not None, "Error, model old is None but distillation specified"
            if opts.loss_kd > 0:
                if opts.ckd:
                    self.kd_criterion = CosineKnowledgeDistillationLoss(reduction='mean')
                else:
                    self.kd_criterion = KnowledgeDistillationLoss(reduction="mean", alpha=opts.kd_alpha)
                self.kd_loss = opts.loss_kd
            if opts.mib_kd > 0:
                self.kd_loss = opts.mib_kd
                self.kd_criterion = UnbiasedKnowledgeDistillationLoss(reduction="mean")
        else:
            self.kd_criterion = None
        
         # Body distillation
        if opts.loss_de > 0:
            assert self.model_old is not None, "Error, model old is None but distillation specified"
            self.de_loss = opts.loss_de
            self.de_criterion = nn.MSELoss()
        else:
            self.de_criterion = None
            
    def make_model(self, config_vit, is_old=False):
        # classifier, self.n_channels = self.get_classifier(is_old)
        n_classes = self.task.get_n_classes()[:-1] if is_old else self.task.get_n_classes()
        model = make_model(self.opts, config_vit, n_classes[0])
        return model

    def distribute(self):
        self.opts = self.opts
        if self.model is not None:
            # Put the model on GPU
            self.distributed = True
            self.model = DistributedDataParallel(self.model, device_ids=[self.opts.device_id],
                                                 output_device=self.opts.device_id, find_unused_parameters=True)

    def get_classifier(self, is_old=False):
        # here distinguish methods!
        self.opts = self.opts
        if self.opts.method == "SPN":
            classes = self.task.get_old_labels() if is_old else self.task.get_order()
            cls = SPNetClassifier(self.opts, classes)
            n_feat = cls.channels
        elif self.opts.method == 'COS':
            n_feat = self.opts.n_feat
            n_classes = self.task.get_n_classes()[:-1] if is_old else self.task.get_n_classes()
            cls = CosineClassifier(n_classes, channels=n_feat)
        else:
            n_feat = self.opts.n_feat
            n_classes = self.task.get_n_classes()[:-1] if is_old else self.task.get_n_classes()
            cls = IncrementalClassifier(n_classes, channels=n_feat)
        return cls, n_feat

    def initialize(self, opts):
        if self.opts.init_mib and self.opts.method == "FT":
            device = self.device
            model = self.model.module if self.distributed else self.model

            classifier = model.cls
            imprinting_w = classifier.cls[0].weight[0]
            bkg_bias = classifier.cls[0].bias[0]

            bias_diff = torch.log(torch.FloatTensor([self.task.get_n_classes()[-1] + 1])).to(device)

            new_bias = (bkg_bias - bias_diff)

            classifier.cls[-1].weight.data.copy_(imprinting_w)
            classifier.cls[-1].bias.data.copy_(new_bias)

            classifier.cls[0].bias[0].data.copy_(new_bias.squeeze(0))

    def warm_up(self, dataset, epochs=1):
        self.warm_up_(dataset, epochs)
        # warm start means make KD after weight imprinting or similar
        if self.dist_warm_start:
            self.model_old.load_state_dict(self.model.state_dict())

    def warm_up_(self, dataset, epochs=1):
        pass

    def cool_down(self, dataset, epochs=1):
        pass
    
    def visualize_pred(self, labels, prediction, cur_epoch, cur_step):
        prediction_arr = sitk.GetImageFromArray(prediction.detach().cpu().numpy())
        label_arr = sitk.GetImageFromArray(labels.detach().cpu().numpy())
        
        curr_path = "/hdd3/cil/baselines/set1_baselines/vanilla_model/vanilla/data/"
        sitk.WriteImage(label_arr, os.path.join('data/merged/dataset/prediction/ground_truth','{}_{}_{}.nii.gz'.format(cur_epoch, cur_step, self.opts.network_arch)))
        sitk.WriteImage(prediction_arr, os.path.join('data/merged/dataset/prediction/predicted_mask','{}_{}_{}.nii.gz'.format(cur_epoch, cur_step, self.opts.network_arch)))

    def visualize_pred_test(self, labels, images, prediction, i_batch):
        prediction_arr = sitk.GetImageFromArray(prediction.detach().cpu().numpy())
        label_arr = sitk.GetImageFromArray(labels.detach().cpu().numpy())
        image_arr = sitk.GetImageFromArray(images.detach().cpu().numpy())
        
        curr_path = "/hdd3/cil/baselines/set1_baselines/vanilla_model/vanilla/data/"
        sitk.WriteImage(label_arr, os.path.join('merged/dataset/prediction/test/ground_truth','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))
        sitk.WriteImage(prediction_arr, os.path.join('merged/dataset/prediction/test/predicted_mask','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))
        sitk.WriteImage(image_arr, os.path.join('merged/dataset/prediction/test/images','{}_{}.nii.gz'.format(i_batch, self.opts.network_arch)))
    
    def ssl_train(self, opts, optimizer, dice_weight, ce_weight, cur_epoch, train_loader, metrics=None, print_int=10, n_iter=1, snapshot_path = None):
        """Train and return epoch loss"""
        if metrics is not None:
            metrics.reset()
        logger = self.logger
        optim = optimizer
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        device = self.device
        model = self.model
        
        tasks = {0: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], # 1-15 is TS
                1: [16, 17, 18, 19, 20], # 16-20 AMOS
                2: [21, 22, 23, 24, 25, 26], # 21-26 BCV
                3: [27, 28, 29, 30], # 27-30 MOTS
                4: [31,32,33],
                5: [34,35,36,37]
            }
        
        curr_classes = tasks[opts.step]
        num_classes = self.opts.num_classes
        max_iterations = self.opts.max_epochs * len(train_loader)
        max_epoch = self.opts.max_epochs
        iter_num = cur_epoch * len(train_loader)
        model.train()
        
        criterion = SupConLoss(temperature=opts.temp)
        
        mean_loss = 0
        # model.train()
        aug1 = [
            {'type': 'random_scale', 'scale_range': (0.87,1.0)},
            {'type': 'adjust_contrast', 'factor': 1.2},
            {'type': 'flip'},
        ]
        
        aug2 = [
            {'type': 'random_scale', 'scale_range': (0.92,1.0)},
            {'type': 'adjust_contrast', 'factor': 1.1},
        ]
        
        
        #prev_image, prev_label = None, None
        len_loader = len(train_loader)
        print(curr_classes)
        '''
        mask = torch.eye(4,dtype=torch.float32).to(self.device)
        mask[0][1],mask[1][0] = 1.0,1.0
        mask[2][3],mask[3][2] = 1.0,1.0
        '''
        
        for i_batch, sampled_batch in enumerate(train_loader):
            
            # print("Patient ids", sampled_batch['case_name'])
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # print("Ground Truth", torch.unique(label_batch))
            #if image_batch.shape[0]==1:
            #    image_batch = torch.cat([image_batch,prev_image],dim=0)
            #    label_batch = torch.cat([label_batch,prev_label],dim=0)
            
            aug1_image0, aug1_label0 = apply_augmentation(image_batch[0], label_batch[0], aug1)
            aug2_image0, aug2_label0 = apply_augmentation(image_batch[0], label_batch[0], aug2)
            
            #aug1_image1, aug1_label1 = apply_augmentation(image_batch[1], label_batch[1], aug1)
            #aug2_image1, aug2_label1 = apply_augmentation(image_batch[1], label_batch[1], aug2)
            
            ## unsqueeze images and labels
            aug1_image0, aug2_image0 = aug1_image0.unsqueeze(0), aug2_image0.unsqueeze(0)
            aug1_label0, aug2_label0 = aug1_label0.unsqueeze(0), aug2_label0.unsqueeze(0)
            
            #aug1_image1, aug2_image1 = aug1_image1.unsqueeze(0), aug2_image1.unsqueeze(0)
            #aug1_label1, aug2_label1 = aug1_label1.unsqueeze(0), aug2_label1.unsqueeze(0)
            #print(torch.unique(aug1_label0),torch.unique(aug1_label0))
            #print(torch.unique(aug1_label1),torch.unique(aug2_label1))
            #sys.exit()
            img0 = torch.cat([aug1_image0,aug2_image0],dim=0).cuda()
            lbl0 = torch.cat([aug1_label0,aug2_label0],dim=0).cuda()
            
            #print(img0.shape,lbl0.shape)
            
            cls_label = []
            cls_ssl = []
            curr_classes = torch.unique(lbl0)
            #print(curr_classes)
            
            for cls in curr_classes:
                if cls==0: continue
                #print(cls)
                cls_label.append(cls)
                
                curr_mask0  = (lbl0.int() == cls).float()
                curr_mask0  = curr_mask0.unsqueeze(1)
                curr_image0 = img0*curr_mask0
                curr_image0 = curr_image0.cuda()
                _,ssl0,_ = model(curr_image0)
                
                #del img0,lbl0,curr_mask0,curr_image0
                
                #print(cls, ssl0.shape)
                cls_ssl.append(ssl0.unsqueeze(0))
                '''
                img1 = torch.cat([aug1_image1,aug2_image1],dim=0).cuda()
                lbl1 = torch.cat([aug1_label1,aug2_label1],dim=0).cuda()
                curr_mask1  = (lbl1.int() == cls).float()
                curr_mask1  = curr_mask1.unsqueeze(1)
                curr_image1 = img1*curr_mask1
                curr_image1 = curr_image1.cuda()
                _,ssl1,_ = model(curr_image1)
                del img1,lbl1,curr_mask1,curr_image1
                
                ssl = torch.cat([ssl0,ssl1],dim=0)
                '''
                
                
                '''
                curr_mask = (label_batch.int() == cls).float()
                curr_mask = curr_mask.unsqueeze(1)
                curr_image=image_batch*curr_mask
                
                #curr_label = label_batch*curr_mask
                #curr_label[curr_label==0.0]=0.0
                
                #curr_image,curr_label = curr_image.cuda(),curr_label.cuda()
                curr_image = curr_image.cuda()
                #print(curr_image.shape)
                
                cls_label.append(cls)
                
                
                _,ssl,_ = model(curr_image)
                cls_ssl.append(ssl.unsqueeze(0))
                '''
            
            cls_label_tensor = torch.tensor(cls_label).cuda()
            features=torch.cat(cls_ssl,dim=0).cuda()
            #print(mask)
            #print(cls_label_tensor)
            #print(features.shape)
            
            con_loss = criterion(features,labels=cls_label_tensor)
            
            mean_loss += con_loss
            optimizer.zero_grad()
            con_loss.backward()
            optimizer.step()
            
            
            for param_group in optimizer.param_groups:
                lr_ = param_group['lr']
            iter_num = iter_num + 1
            if iter_num % 8 == 0:
                logger.info('epoch : %d, iteration : %d, con-train loss : %f' % (
                    cur_epoch, iter_num, con_loss.item()))
                    
            #if len_loader%2==1 and len_loader-i_batch==2:            
            #    prev_image, prev_label = image_batch[0].unsqueeze(0).cpu(), label_batch[0].unsqueeze(0).cpu()          
        mean_loss = float(mean_loss / len(train_loader))
        
        logger.info('epoch : %d, mean con-train loss : %f' % (cur_epoch, mean_loss))
        
        return mean_loss
    
    
    def rotate_image(self, image, label, degree, axes=(3, 4)):
    
        k = degree // 90  # Calculate the number of 90-degree rotations

        rotated_image = torch.rot90(image, k=k, dims=axes)
        rotated_label = torch.rot90(label, k=k, dims=(2,3))
        return rotated_image, rotated_label
        
    
    
    def train(self, opts, optimizer, dice_weight, ce_weight, cur_epoch, train_loader, metrics=None, print_int=10, n_iter=1, snapshot_path = None):
        """Train and return epoch loss"""
        if metrics is not None:
            metrics.reset()
        logger = self.logger
        optim = optimizer
        logger.info("Epoch %d, lr = %f" % (cur_epoch, optim.param_groups[0]['lr']))

        device = self.device
        model = self.model
        
        num_classes = self.opts.num_classes
        max_iterations = self.opts.max_epochs * len(train_loader)
        max_epoch = self.opts.max_epochs
        iter_num = cur_epoch * len(train_loader)
        model.train()
        class_weights = torch.FloatTensor(ce_weight).cuda()
        ce_loss = CrossEntropyLoss(weight=class_weights, ignore_index=255)
        dice_loss = DiceLoss(num_classes, dice_weight)
        rot_loss = nn.CrossEntropyLoss()
        rot_weight = opts.rot_weight
        
        mean_loss = 0
        mean_dice = 0
        mean_loss_ce = 0
        # model.train()

        # for iteration in range(n_iter):
            # train_loader.sampler.set_epoch(cur_epoch*n_iter + iteration)  # setup dataloader sampler
        for i_batch, sampled_batch in enumerate(train_loader):
            # print("Patient ids", sampled_batch['case_name'])
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            # print("Ground Truth", torch.unique(label_batch))
            
            for ang in range(4):
                ang_to_rot = [0,90,180,270]
                
                curr_rot = ang_to_rot[ang]
                
                if image_batch.shape[0]==1:
                    curr_lbl = torch.tensor(ang).unsqueeze(0).to(self.device)
                else:
                    curr_lbl = torch.cat([torch.tensor(ang).unsqueeze(0),torch.tensor(ang).unsqueeze(0)],dim=0).to(self.device)
                
                if curr_rot != 0:
                    image_batch, label_batch = self.rotate_image(image_batch, label_batch, curr_rot) 
                    
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                # outputs, feat = model(image_batch)
                
                ## out,ssl,rot
                outputs,_,rot = model(image_batch)
                # print(image_batch.shape, outputs.shape)
                # print("Prediction", torch.unique(outputs))
                loss_ce_1 = ce_loss(outputs, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(outputs, label_batch[:].long(), self.total_classes, softmax=True)

                loss_ce = loss_ce_1
                loss_dice = loss_dice_1
                
                rloss = torch.tensor([0.]).to(self.device)
                rloss = rot_weight*rot_loss(rot,curr_lbl)
                
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                
                
                if not self.opts.vanila:
                    loss_tot = loss + rloss
                    if rloss <= CLIP:
                        loss_tot = loss + rloss
                    else:
                        print(f"Warning, rloss is {rloss}! Term ignored")
                        loss_tot = loss
                        
                    #loss_tot = loss + rloss
                else:
                    loss_tot = loss

                #print(loss_tot,loss,rloss)
                
                dice = 1 - loss_dice_1
                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce
                optimizer.zero_grad()
                loss_tot.backward()
                optimizer.step()
                
                

                for param_group in optimizer.param_groups:
                    lr_ = param_group['lr']
                iter_num = iter_num + 1
                if iter_num % 8 == 0:
                    logger.info('epoch : %d, iteration : %d, train loss : %f, train loss_ce: %f, train loss_dice: %f, train dice : %f' % (
                        cur_epoch, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))
                
                if metrics is not None:
                    class_dice = metrics.calculate_dice_each_class(class_wise_dice) 
                    
                
                        
        # mean loss and mean dice
        mean_loss = float(mean_loss / len(train_loader))
        mean_dice = float(mean_dice / len(train_loader))
        mean_loss_ce = float(mean_loss_ce / len(train_loader))
        # mean_class_dice = list(map(lambda x: x/len(train_loader) if x != 'X' else x, class_dice))

        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                # print(class_freq[i])
                # print(class_dice[i])
                class_dice[i] = class_dice[i]/class_freq[i]
        
        # print('epoch :', cur_epoch, 'Train Loss :', mean_loss, 'Train dice :', mean_dice)
        logger.info('epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        print('epoch : %d, mean train loss : %f, mean train ce loss: %f, mean train dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))

        save_interval = 50  # int(max_epoch/5)
        # if cur_epoch > int(max_epoch / 5) and (cur_epoch + 1) % save_interval == 0:
        if (cur_epoch + 1) >= 350 and (cur_epoch + 1) % save_interval == 0:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(cur_epoch + 1) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logger.info("save model to {}".format(save_mode_path))

        if cur_epoch + 1 == max_epoch:
            save_mode_path = os.path.join(snapshot_path, 'epoch_' + str(cur_epoch + 1) + '.pth')
            torch.save(model.state_dict(), save_mode_path)
            logger.info("save model to {}".format(save_mode_path))

        return class_dice, mean_dice

    def validate(self, dice_weight, ce_weight, loader, metrics, ret_samples_ids=None, novel=False, cur_epoch=None, snapshot_path = None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        # criterion = self.criterion
        logger = self.logger

        model.eval()
        iter_num = cur_epoch * len(loader)
        num_classes = self.opts.num_classes
        class_weights = torch.FloatTensor(ce_weight).cuda()
        ce_loss = CrossEntropyLoss(weight=class_weights, ignore_index=255)
        dice_loss = DiceLoss(num_classes, dice_weight)
        mean_loss = 0
        mean_dice = 0
        mean_loss_ce = 0
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(loader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                # outputs, _ = model(image_batch)
                outputs,_,_ = model(image_batch)
                
                loss_ce_1 = ce_loss(outputs, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(outputs, label_batch, self.total_classes, softmax=True)


                loss_ce = loss_ce_1
                loss_dice = loss_dice_1
                dice = 1 - loss_dice_1
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce

                # _, prediction = outputs.max(dim=1)
                # metrics.update(label_batch[:].long(), prediction)4
                class_dice = metrics.calculate_dice_each_class(class_wise_dice)            

                
                iter_num += 1

                if iter_num % 3 == 0:
                    logger.info('epoch : %d, iteration : %d, val loss : %f, val loss_ce: %f, val loss_dice: %f, val dice : %f' % (
                        cur_epoch, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))
                    print('epoch : %d, iteration : %d, val loss : %f, val loss_ce: %f, val loss_dice: %f, val dice : %f' % (
                        cur_epoch, iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))

                # if(iter_num % 2 == 0):
                #     res_array = np.zeros(26).tolist()
                #     for i in range(26):
                #         if class_freq[i] != 0:
                #             res_array[i] = class_dice[i]/class_freq[i]
                            
                    # print("val dices:", res_array)
        # mean loss and mean dice
        mean_loss = float(mean_loss / len(loader))
        mean_dice = float(mean_dice / len(loader))
        mean_loss_ce = float(mean_loss_ce / len(loader))
        # mean_class_dice = list(map(lambda x: x/len(loader) if x != 'X' else x, class_dice))

        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i]/class_freq[i]
                
        logger.info('epoch : %d, mean val loss : %f, mean val ce loss: %f, mean val dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        print('epoch : %d, mean val loss : %f, mean val ce loss: %f, mean val dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        return class_dice, mean_dice
    
    def test(self, dice_weight, ce_weight, loader, metrics, ret_samples_ids=None, novel=False, cur_epoch=None, snapshot_path = None):
        """Do validation and return specified samples"""
        metrics.reset()
        model = self.model
        device = self.device
        # criterion = self.criterion
        logger = self.logger

        model.eval()
        iter_num = cur_epoch * len(loader)
        num_classes = self.opts.num_classes
        class_weights = torch.FloatTensor(ce_weight).cuda()
        ce_loss = CrossEntropyLoss(weight=class_weights, ignore_index=255)
        dice_loss = DiceLoss(num_classes, dice_weight)
        mean_loss = 0
        mean_dice = 0
        mean_loss_ce = 0
        
        
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(loader):
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                # outputs, _ = model(image_batch)
                outputs,_,_ = model(image_batch)
                # print("Ground Truth", torch.unique(label_batch))
                loss_ce_1 = ce_loss(outputs, label_batch[:].long())
                class_freq, class_wise_dice, loss_dice_1 = dice_loss(outputs, label_batch, self.total_classes, softmax=True)

                # print("Ground Truth", torch.unique(label_batch))
                # print("Prediction", torch.unique(outputs))

                loss_ce = loss_ce_1
                loss_dice = loss_dice_1
                dice = 1 - loss_dice_1
                loss = 0.5 * loss_ce + 0.5 * loss_dice
                mean_loss += loss
                mean_dice += dice
                mean_loss_ce += loss_ce

                # _, prediction = outputs.max(dim=1)
                # metrics.update(label_batch[:].long(), prediction)

                # _, prediction = outputs.max(dim=1)
                
                # print("Prediction", torch.unique(prediction))
                    
                class_dice = metrics.calculate_dice_each_class(class_wise_dice)            

                # self.visualize_pred_test(label_batch, image_batch, outputs, i_batch)           

                
                if iter_num % 5 == 0:
                    print('iteration : %d, test loss : %f, test loss_ce: %f, test loss_dice: %f, test dice : %f' % (
                        iter_num, loss.item(), loss_ce.item(), loss_dice.item(), dice.item()))


                iter_num += 1

        # mean loss and mean dice
        mean_loss = float(mean_loss / len(loader))
        mean_dice = float(mean_dice / len(loader))
        mean_loss_ce = float(mean_loss_ce / len(loader))
        
        # print(class_dice)
        # print(class_freq)
        for i in range(self.opts.num_classes):
            if class_freq[i] != 0:
                class_dice[i] = class_dice[i]/class_freq[i]

        # mean_class_dice = list(map(lambda x: x/len(loader) if x != 'X' else x, class_dice))

        logger.info('epoch : %d, mean val loss : %f, mean val ce loss: %f, mean val dice : %f' % (cur_epoch, mean_loss, mean_loss_ce, mean_dice))
        print('Mean Test loss : %f, mean Test ce loss: %f, Mean Test dice : %f' % (mean_loss, mean_loss_ce, mean_dice))
        return class_dice, mean_dice
    
    def state_dict(self):
        state = {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict()}
        return state
    def load_body(self, model_dict):
        new_state = {}
        for k, v in model_dict.items():
            # if "hybrid_model" in k or "encoder1" in k:
            if "outc" not in k:
                new_state[k] = v
        return new_state
    

    def load_dict_imprint(self, path, strict=True):
        for name, param in path.items():
            if name.startswith('outc'):
                continue  # Skip the final layer weights
            if name in self.model.state_dict():
                self.model.state_dict()[name].copy_(param)

        # Next, randomize the parameters of the final layer for the additional columns
        # with torch.no_grad():
        #     self.model.outc.weight[:len(self.labels_old) + 1, :] = path['outc.weight']            
        # for name, param in path.items():
        #     if name.startswith('outc'):
        #         self.model.state_dict()[name].requires_grad = True
        #     else:
        #         self.model.state_dict()[name].requires_grad = False
        
        
        # for param in self.model.parameters():
        #     param.requires_grad = True
        
        for name, param in self.model.named_parameters():
            if name.startswith('outc') or name.startswith('up'):
                param.requires_grad = True
            else:
                param.requires_grad = True
        # for param in self.model.parameters():
        #     print(param.requires_grad)
        # for name, param in path.items():
        #     print(name)
        #     print(self.model.state_dict()[name].requires_grad)
        #     # param.requires_grad = True
    def load_dict(self, path, strict=True):
        body = self.load_body(path)
        self.model.load_state_dict(body, strict)
    
    def load_dict_full_model(self, path, strict=True):
        self.model.load_state_dict(path, strict)
        
