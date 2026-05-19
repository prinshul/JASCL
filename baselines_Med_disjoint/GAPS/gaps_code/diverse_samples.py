import utils
import argparser
import os
from utils.logger import Logger

from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from statistics import mean

import numpy as np
import random
import torch
from torch.utils import data
from torch import distributed

from networks.VerSe.unet import CONFIGS as CONFIGS_ViT_seg
import json
from dataset import get_dataset
from metrics import StreamSegMetrics
from task import Task

from methods import get_method
import time
from networks.VerSe.unet import network as network
from utils.VerSe_utils import DiceLoss, print_network
from tensorboardX import SummaryWriter
from tqdm import tqdm
import pickle 
import copy


import pickle5 as pkl
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import shutil
import fnmatch
import zipfile
from scipy import ndimage
from scipy.spatial.distance import cdist

import torch.nn.functional as F
import random


def save_ckpt(path, model, epoch):
    """ save current model
    """
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
    }

    torch.save(state, path)

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch) 

def get_step_ckpt(model_path):
    # xxx Get step checkpoint
    step_checkpoint = None
    if model_path is not None:
        path = model_path
        
            #path = f"checkpoints/step/{task_name}/{name}_{opts.step}.pth"
    # generate model from path
    if os.path.exists(path):
        step_checkpoint = torch.load(path, map_location="cuda")
        step_checkpoint['path'] = path
    else:
        raise FileNotFoundError(f"Step checkpoint not found in {path}")

    return step_checkpoint

def store_results(class_wise_dice):
    keys = list(range(opts.num_classes))
    values = class_wise_dice

    result = dict(zip(keys, values))
    return result


# =====  Log metrics on Tensorboard =====
def log_val(logger, val_metrics, val_score, val_loss, cur_epoch):
    logger.info(val_metrics.to_str(val_score))

    # visualize validation score and samples
    logger.add_scalar("V-Loss", val_loss, cur_epoch)
    logger.add_scalar("Val_Overall_Acc", val_score['Overall Acc'], cur_epoch)
    logger.add_scalar("Val_MeanIoU", val_score['Mean IoU'], cur_epoch)
    logger.add_table("Val_Class_IoU", val_score['Class IoU'], cur_epoch)
    logger.add_table("Val_Acc_IoU", val_score['Class Acc'], cur_epoch)
    # logger.add_figure("Val_Confusion_Matrix", val_score['Confusion Matrix'], cur_epoch)


def log_samples(logger, ret_samples, denorm, label2color, cur_epoch):
    for k, (img, target, pred) in enumerate(ret_samples):
        img = (denorm(img) * 255).astype(np.uint8)
        target = label2color(target).transpose(2, 0, 1).astype(np.uint8)
        pred = label2color(pred).transpose(2, 0, 1).astype(np.uint8)

        concat_img = np.concatenate((img, target, pred), axis=2)  # concat along width
        logger.add_image(f'Sample_{k}', concat_img, cur_epoch)

def mean_dice(task, class_wise_dice):
    mean_dice = []
    for i in range(opts.num_classes):
        if i in task.novel_classes:
            mean_dice.append(class_wise_dice[i])
    return mean(mean_dice)

def mean_dice_test1(task, class_wise_dice):
    mean_dice = []
    for i in range(opts.num_classes):
        if i in task.old_classes :
            mean_dice.append(class_wise_dice[i])
    return mean(mean_dice)

def mean_dice_test2(task, class_wise_dice):
    mean_dice = []
    for i in range(opts.num_classes):
        if i in task.novel_classes or i in task.old_classes:
            mean_dice.append(class_wise_dice[i])
    return mean(mean_dice)


def print_lbl(img):
    print(img[-20:])
    image_rd = sitk.ReadImage(img)
    image_rd = sitk.GetArrayFromImage(image_rd)
    print(image_rd.shape)
    print(np.unique(image_rd))

def read_img(img):
    print(img)
    image_rd = sitk.ReadImage(img)
    image_rd = sitk.GetArrayFromImage(image_rd)
    #print(image_rd.shape)
    return image_rd



def flip_xz_yz(image, label):
    flip_id = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
    image = np.ascontiguousarray(image[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    label = np.ascontiguousarray(label[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    return image, label


def random_rotate(image, label, min_value):
    angle = np.random.randint(-15, 15)  # -20--20
    rotate_axes = [(0, 1), (1, 2), (0, 2)]
    k = np.random.randint(0, 3)
    image = ndimage.interpolation.rotate(image, angle, axes=rotate_axes[k], reshape=False, order=3, mode='constant',
                                         cval=min_value)
    label = ndimage.interpolation.rotate(label, angle, axes=rotate_axes[k], reshape=False, order=0, mode='constant',
                                         cval=0.0)

    return image, label

class RandomGenerator(object):
    def __init__(self, output_size, mode):
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        min_value = np.min(image)
        # centercop
        # crop alongside with the ground truth
        # print(".............", np.unique(label))
        
            
        index = np.nonzero(label)
        index = np.transpose(index)  # 转置后变成二维矩阵，每一行有三个索引元素，分别对应z,x,y三个方向
        
        z_min = np.min(index[:, 0])
        z_max = np.max(index[:, 0])
        y_min = np.min(index[:, 1])
        y_max = np.max(index[:, 1])
        x_min = np.min(index[:, 2])
        x_max = np.max(index[:, 2])

        # middle point
        z_middle = np.int((z_min + z_max) / 2)
        y_middle = np.int((y_min + y_max) / 2)
        x_middle = np.int((x_min + x_max) / 2)

        Delta_z = np.int((z_max - z_min) / 3)  # 3
        Delta_y = np.int((y_max - y_min) / 4)  # 8
        Delta_x = np.int((x_max - x_min) / 4)  # 8

        # random number of x, y, z
        # z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)
        y_random = random.randint(y_middle - Delta_y, y_middle + Delta_y)
        x_random = random.randint(x_middle - Delta_x, x_middle + Delta_x)
        
        thre = z_min + Delta_z + np.int(self.output_size[0] / 2)
        if z_middle > thre:          # 此时z_middle + Delta_z < z_max
            delta_Z = z_middle - z_min - np.int(self.output_size[0] / 4)                         # 正常 np.int(self.output_size[0] / 2)，此时再大点，保证可以超出现有的范围
            z_random = random.randint(z_middle - delta_Z, z_middle + delta_Z)
        else:
            z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)

        # crop patch
        crop_z_down = z_random - np.int(self.output_size[0] / 2)
        crop_z_up = z_random + np.int(self.output_size[0] / 2)
        crop_y_down = y_random - np.int(self.output_size[1] / 2)
        crop_y_up = y_random + np.int(self.output_size[1] / 2)
        crop_x_down = x_random - np.int(self.output_size[2] / 2)
        crop_x_up = x_random + np.int(self.output_size[2] / 2)

        # padding
        if crop_z_down < 0 or crop_z_up > image.shape[0]:
            delta_z = np.maximum(np.abs(crop_z_down), np.abs(crop_z_up - image.shape[0]))
            image = np.pad(image, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=min_value)
            label = np.pad(label, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=0.0)
            crop_z_down = crop_z_down + delta_z
            crop_z_up = crop_z_up + delta_z

        if crop_y_down < 0 or crop_y_up > image.shape[1]:
            delta_y = np.maximum(np.abs(crop_y_down), np.abs(crop_y_up - image.shape[1]))
            image = np.pad(image, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=min_value)
            label = np.pad(label, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=0.0)
            crop_y_down = crop_y_down + delta_y
            crop_y_up = crop_y_up + delta_y

        if crop_x_down < 0 or crop_x_up > image.shape[2]:
            delta_x = np.maximum(np.abs(crop_x_down), np.abs(crop_x_up - image.shape[2]))
            image = np.pad(image, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=min_value)
            label = np.pad(label, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=0.0)
            crop_x_down = crop_x_down + delta_x
            crop_x_up = crop_x_up + delta_x

        label = label[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
        image = image[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]

        label = np.round(label)

        # data augmentation
        if self.mode == 'train':
            if random.random() > 0.5:
                image, label = flip_xz_yz(image, label)
            if random.random() > 0.5:                      # elif random.random() > 0.5:
                image, label = random_rotate(image, label, min_value)
                label = np.round(label)
 
        image = torch.from_numpy(image.astype(np.float)).float()
        label = torch.from_numpy(label.astype(np.float32)).float()

        sample = {'image': image, 'label': label.long(), 'case_name': sample['case_name']}

        return sample




data_path = "cil/processed_common_organ_data/merged/"

splits_path = "cil/processed_common_organ_data/merged/split/"
data_train = "cil/processed_common_organ_data/merged/dataset/images/train/"
ann_train = "cil/processed_common_organ_data/merged/dataset/annotations/train/"
gaps_ckpt = "/hdd2/cil/running_base/GAPS/gaps_code/checkpoints/step/15ss-merged/"

data_exmp = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/images/exemplar/"
ann_exmp = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/annotations/exemplar/"

data_exmp_step1 = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/images/exemplar_step1/"
ann_exmp_step1 = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/annotations/exemplar_step1/"

data_exmp1 = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/images/exemplar1/"
data_exmp2 = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/images/exemplar2/"

ann_exmp1 = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/annotations/exemplar1/"
ann_exmp2 = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/annotations/exemplar2/"

os.makedirs(data_exmp, exist_ok=True)
os.makedirs(ann_exmp, exist_ok=True)

os.makedirs(data_exmp_step1, exist_ok=True)
os.makedirs(ann_exmp_step1, exist_ok=True)

os.makedirs(data_exmp1, exist_ok=True)
os.makedirs(data_exmp2, exist_ok=True)

os.makedirs(ann_exmp1, exist_ok=True)
os.makedirs(ann_exmp2, exist_ok=True)






def get_base_weights(model_path):
    step_ckpt = torch.load(model_path, map_location="cuda")
    #print(step_ckpt)
    base_weight = step_ckpt['model_state']['model']['outc.weight'].detach().clone().requires_grad_(False)
    if step_ckpt['model_state']['model']['outc.bias'] is not None:
        base_bias = step_ckpt['model_state']['model']['outc.bias'].detach().clone().requires_grad_(False)
        del step_ckpt
        return base_weight, base_bias
    else:
        del step_ckpt
        return base_weight, None
    #print(step_ckpt['model_state']['model']['outc.weight'].shape)


###############################################################################################################
###############################################################################################################
def step_zero_no_replay(opts,task,logger,rank,device):
    print("Step 0 ")
    task_name = f"{opts.task}-{opts.dataset}"
    name = f"{opts.name}-s{task.nshot}-i{task.ishot}" if task.nshot != -1 else f"{opts.name}"
    model_path = gaps_ckpt+"FT_0_newUNET_dynamic.pth"
    opts.model_path = model_path
    
    
    
    dataset_name = opts.dataset
    batch_size = opts.batch_size * opts.n_gpu
    
    ####################################################################################
    opts.is_pretrain = True
    opts.exp = 'TU_' + dataset_name + str(opts.img_size)
    snapshot_path = "../model/{}/{}".format(opts.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if opts.is_pretrain else snapshot_path
    snapshot_path += '_' + opts.vit_name
    snapshot_path = snapshot_path + '_skip' + str(opts.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(opts.vit_patches_size) if opts.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(opts.max_iterations)[0:2]+'k' if opts.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(opts.max_epochs) if opts.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(opts.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(opts.base_lr) if opts.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(opts.img_size)
    snapshot_path = snapshot_path + '_s'+str(opts.seed) if opts.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    config_vit = CONFIGS_ViT_seg[opts.vit_name]
    config_vit.n_classes = opts.num_classes
    config_vit.n_skip = opts.n_skip
    config_vit.batch_size = opts.batch_size
    # number of patches
    if opts.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(opts.img_size[0] / opts.vit_patches_size), int(opts.img_size[1] / opts.vit_patches_size), int(opts.img_size[2] / opts.vit_patches_size))
    ###
    config_vit.n_patches = int(opts.img_size[0] / opts.vit_patches_size) * int(opts.img_size[1] / opts.vit_patches_size) * int(opts.img_size[2] / opts.vit_patches_size)
    config_vit.n_patches = int(opts.img_size[0] / opts.vit_patches_size) * int(opts.img_size[1] / opts.vit_patches_size) * int(opts.img_size[2] / opts.vit_patches_size)
    config_vit.h = int(opts.img_size[0] / opts.vit_patches_size)
    config_vit.w = int(opts.img_size[1] / opts.vit_patches_size)
    config_vit.l = int(opts.img_size[2] / opts.vit_patches_size)
    
    
    ##########################################################################################################################
    
    

    model = get_method(opts, task, device, logger, config_vit)
    
    # Put the model on GPU  // Make it always, also after train, to remediate the cool_down method
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # checkpoint = torch.load("cil/FSS/checkpoints/step/7ss-verse/27_july_FT_0.pth")

    model.load_dict_full_model(checkpoint["model_state"]["model"])
    print("Model restored from {}".format(model_path))
    logger.info(f"*** Model restored from {model_path}")
    del checkpoint
    
    
    base_weight, base_bias = get_base_weights(model_path)
    print("Base Weights ",base_weight.shape, base_bias.shape)


    for name, param in model.model.named_parameters():
        #print(name, param.requires_grad)
        if name=='outc.weight':
            model_outc_w = param.detach().clone().requires_grad_(False)
        if name=='outc.bias':
            model_outc_b = param.detach().clone().requires_grad_(False)


    print("Model loaded weights ",model_outc_w.shape,model_outc_b.shape)
    #print(model_outc_w)

    print((base_weight == model_outc_w).all())
    print((base_bias == model_outc_b).all())
    base_weight=torch.squeeze(base_weight).cpu().numpy()
    
    print("Checking for diverse samples in Step 0 training samples only")
        
    train_split = splits_path+"train.txt"
    sample_list = open(train_split).readlines()
    #print(sample_list)

    train_pickle = splits_path+'inverse_dict_new_train.pkl'
    class_to_images = pkl.load(open(train_pickle, 'rb'))
    print(class_to_images)

    base_class = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    base_class_zero = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    base_class_num = 16

    base_samples = set()
    for cl in base_class:
        for sm in class_to_images[cl]:
            base_samples.add(sm)
    base_samples = list(base_samples)

    #print(base_samples)

    for i in range(0,len(base_samples)):
        base_samples[i] = sample_list[base_samples[i]][:-1]+".nii.gz"
    print("Step 0 = ",base_samples)

    Rnd_G = RandomGenerator(output_size=[128,160,96], mode = 'train')

    base_dist={}
    ## Base step img, lbl
    for i in range(0,len(base_samples)):
        print("\nBase step sample = ",base_samples[i])
        base_img = read_img(data_train+base_samples[i])#.to(device)
        base_lbl = read_img(ann_train+base_samples[i])#.to(device)
        thresh_base = (int(base_class_num) - 1) + 0.5
        base_lbl[base_lbl < 0.5] = 0.0  # maybe some voxels is a minus value
        base_lbl[base_lbl > thresh_base] = 255
        sample_base_unedit = {'image': base_img, 'label': base_lbl, 'case_name':base_samples[i][:-7]}

        while True:
            sample_base_crop = Rnd_G(sample_base_unedit)

            base_labels = torch.unique(sample_base_crop['label'])
            print(base_labels)

            if len(base_labels)>0:
                break

        #learned_proto = get_protoype()
        print(type(sample_base_crop))

        print(sample_base_crop['image'].shape)


        print("Prototype Taking")

        pr_all = torch.zeros((base_class_num,32))
        for c in base_class_zero:
            #print(c)
            img, lbl = sample_base_crop['image'], sample_base_crop['label']
            img, lbl = img.to(device), lbl.to(device)
            pr_w = model.diverse_proto(img, lbl, c)
            if pr_w is not None:
                #print(c, "\t ",pr_w.shape)
                #print(pr_w)
                pr_all[c]=pr_w.detach().requires_grad_(False)
        #print(pr_all.shape)
        pr_all=pr_all.cpu().numpy()
        
        print("Protoype and base prototype shape ",pr_all.shape, base_weight.shape)
        dist=cdist(base_weight, pr_all).mean()
        print("Distance = ",dist)
        base_dist[base_samples[i]]=dist
   
    base_dist_sorted = sorted(base_dist.items(), key=lambda x:x[1])
    print(base_dist_sorted)
    
    ## 4/40 Diverse Samples selection Uniform distance
    div_samples = 6
    base_div_list = []
    for i in range(0,len(base_dist_sorted),div_samples):
        base_div_list.append(base_dist_sorted[i][0])
    print(base_div_list)

    
    print("Copying base samples for GAPS")
    for i in range(0,len(base_div_list)):
        print(base_div_list[i])
        shutil.copy(data_train+base_div_list[i], data_exmp)
        shutil.copy(ann_train+base_div_list[i], ann_exmp)
        
    print('Base Step ended')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
###############################################################################################################################    
###############################################################################################################################    
    
def step_more_with_replay(opts,task,logger,rank,device):
    
    
    print("Step ",opts.step)
    print('GAPS will follow')
    task_name = f"{opts.task}-{opts.dataset}"
    name = f"{opts.name}-s{task.nshot}-i{task.ishot}" if task.nshot != -1 else f"{opts.name}"
    
    
    model_path = gaps_ckpt+'FT-s5-i0_1_newUNET_dynamic.pth'
    opts.model_path = model_path
    dataset_name = opts.dataset

    ## Batch size 
    batch_size = opts.batch_size * opts.n_gpu
    print(batch_size)
    
    opts.is_pretrain = True
    opts.exp = 'TU_' + dataset_name + str(opts.img_size)
    snapshot_path = "../model/{}/{}".format(opts.exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain' if opts.is_pretrain else snapshot_path
    snapshot_path += '_' + opts.vit_name
    snapshot_path = snapshot_path + '_skip' + str(opts.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(opts.vit_patches_size) if opts.vit_patches_size!=16 else snapshot_path
    snapshot_path = snapshot_path+'_'+str(opts.max_iterations)[0:2]+'k' if opts.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' +str(opts.max_epochs) if opts.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path+'_bs'+str(opts.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(opts.base_lr) if opts.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_'+str(opts.img_size)
    snapshot_path = snapshot_path + '_s'+str(opts.seed) if opts.seed!=1234 else snapshot_path

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    
    ######################## Not used #########################################################################
    
    config_vit = CONFIGS_ViT_seg[opts.vit_name]
    config_vit.n_classes = opts.num_classes
    config_vit.n_skip = opts.n_skip
    config_vit.batch_size = opts.batch_size
    # number of patches
    if opts.vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(opts.img_size[0] / opts.vit_patches_size), int(opts.img_size[1] / opts.vit_patches_size), int(opts.img_size[2] / opts.vit_patches_size))
    ###
    config_vit.n_patches = int(opts.img_size[0] / opts.vit_patches_size) * int(opts.img_size[1] / opts.vit_patches_size) * int(opts.img_size[2] / opts.vit_patches_size)
    config_vit.n_patches = int(opts.img_size[0] / opts.vit_patches_size) * int(opts.img_size[1] / opts.vit_patches_size) * int(opts.img_size[2] / opts.vit_patches_size)
    config_vit.h = int(opts.img_size[0] / opts.vit_patches_size)
    config_vit.w = int(opts.img_size[1] / opts.vit_patches_size)
    config_vit.l = int(opts.img_size[2] / opts.vit_patches_size)
    
    ################################################################################################
        
    
    
    
    
    
    print(task.step)
    
    
    
    model = get_method(opts, task, device, logger, config_vit)
    
    #print(model)
    # Put the model on GPU  // Make it always, also after train, to remediate the cool_down method
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # checkpoint = torch.load("cil/FSS/checkpoints/step/7ss-verse/27_july_FT_0.pth")

    model.load_dict_full_model(checkpoint["model_state"]["model"])
    print("Model restored from {}".format(model_path))
    logger.info(f"*** Model restored from {model_path}")
    del checkpoint
    
    
    step_weight, step_bias = get_base_weights(model_path)
    print("Step Weights ",step_weight.shape, step_bias.shape)


    for name, param in model.model.named_parameters():
        #print(name, param.requires_grad)
        if name=='outc.weight':
            model_outc_w = param.detach().clone().requires_grad_(False)
        if name=='outc.bias':
            model_outc_b = param.detach().clone().requires_grad_(False)


    print("Model loaded weights ",model_outc_w.shape,model_outc_b.shape)
    #print(model_outc_w)

    print((step_weight == model_outc_w).all())
    print((step_bias == model_outc_b).all())
    step_weight=torch.squeeze(step_weight).cpu().numpy()
    
    print("Checking for diverse samples in Step 1 training samples only")
        
    train_split = splits_path+"train.txt"
    sample_list = open(train_split).readlines()
    #print(sample_list)

    train_pickle = splits_path+'inverse_dict_new_train.pkl'
    class_to_images = pkl.load(open(train_pickle, 'rb'))
    print(class_to_images)

    step_class = [16,17,18,19,20]
    step_class_zero = [0,16,17,18,19,20]
    step_class_num = 21

    step_samples = set()
    for cl in step_class:
        for sm in class_to_images[cl]:
            step_samples.add(sm)
    step_samples = list(step_samples)

    #print(base_samples)

    for i in range(0,len(step_samples)):
        step_samples[i] = sample_list[step_samples[i]][:-1]+".nii.gz"
    print("Step 1 = ",step_samples)

    Rnd_G = RandomGenerator(output_size=[128,160,96], mode = 'train')

    step_dist={}
    ## Base step img, lbl
    for i in range(0,len(step_samples)):
        print("\nStep 1 step sample = ",step_samples[i])
        step_img = read_img(data_train+step_samples[i])#.to(device)
        step_lbl = read_img(ann_train+step_samples[i])#.to(device)
        thresh_base = (int(step_class_num) - 1) + 0.5
        step_lbl[step_lbl < 0.5] = 0.0  # maybe some voxels is a minus value
        step_lbl[step_lbl > thresh_base] = 255
        sample_step_unedit = {'image': step_img, 'label': step_lbl, 'case_name': step_samples[i][:-7]}

        while True:
            sample_step_crop = Rnd_G(sample_step_unedit)

            step_labels = torch.unique(sample_step_crop['label'])
            print(step_labels)

            if len(step_labels)>0:
                break

        #learned_proto = get_protoype()
        print(type(sample_step_crop))

        print(sample_step_crop['image'].shape)


        print("Prototype Taking")

        pr_all = torch.zeros((step_class_num,32))
        for c in step_class_zero:
            #print(c)
            img, lbl = sample_step_crop['image'], sample_step_crop['label']
            img, lbl = img.to(device), lbl.to(device)
            pr_w = model.diverse_proto(img, lbl, c)
            if pr_w is not None:
                #print(c, "\t ",pr_w.shape)
                #print(pr_w)
                pr_all[c]=pr_w.detach().requires_grad_(False)
        #print(pr_all.shape)
        pr_all=pr_all.cpu().numpy()
        
        print("Protoype and base prototype shape ",pr_all.shape, step_weight.shape)
        dist=cdist(step_weight, pr_all).mean()
        print("Distance = ",dist)
        step_dist[step_samples[i]]=dist
   
    step_dist_sorted = sorted(step_dist.items(), key=lambda x:x[1])
    print(step_dist_sorted)
    
    ## 4/40 Diverse Samples selection Uniform distance
    div_samples = 6
    step_div_list = []
    for i in range(0,len(step_dist_sorted),div_samples):
        step_div_list.append(step_dist_sorted[i][0])
    print(step_div_list)

    
    print("Copying base samples for GAPS")
    for i in range(0,len(step_div_list)):
        print(step_div_list[i])
        shutil.copy(data_train+step_div_list[i], data_exmp_step1)
        shutil.copy(ann_train+step_div_list[i], ann_exmp_step1)
        
    
   
    print('Step ended')
    


   
    
    
    
    
    
    
    
    
    
    
    
    
def main(opts):
    # =============================== Main ==================================================================================
    # =======================================================================================================================
    # distributed.init_process_group(backend='nccl', init_method='env://')
    if opts.device is not None:
        device_id = opts.device
    else:
        device_id = opts.local_rank
    device = torch.device(device_id)
    # rank, world_size = distributed.get_rank(), distributed.get_world_size()
    if opts.device is not None:
        torch.cuda.set_device(opts.device)
    else:
        torch.cuda.set_device(device_id)
    opts.device_id = device_id
    torch.cuda.empty_cache()
    task = Task(opts)
    
    
    # Initialize logging
    task_name = f"{opts.task}-{opts.dataset}"
    name = f"{opts.name}-s{task.nshot}-i{task.ishot}" if task.nshot != -1 else f"{opts.name}"
    if task.nshot != -1:
        logdir_full = f"{opts.logdir}/{task_name}/{name}/"
    else:
        logdir_full = f"{opts.logdir}/{task_name}/{opts.name}/"
    # if rank == 0:
    #     logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=opts.visualize, step=opts.step)
    # else:
    #     logger = Logger(logdir_full, rank=rank, debug=opts.debug, summary=False)
    rank = 0
    logger = Logger(logdir_full, rank)
    logger.print(f"Device: {device}")
    
    
    
    
    if opts.step==0:
        step_zero_no_replay(opts,task,logger,rank,device)
        torch.cuda.empty_cache()
    else:
        step_more_with_replay(opts,task,logger,rank,device)
        torch.cuda.empty_cache()
    
    

if __name__ == '__main__':
    parser = argparser.get_argparser()
    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)


    main(opts)



