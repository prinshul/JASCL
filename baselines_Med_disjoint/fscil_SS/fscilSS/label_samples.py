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
data_train = "cil/processed_common_organ_data/merged/dataset/images/train/"
ann_train = "cil/processed_common_organ_data/merged/dataset/annotations/train/"
splits_path = "cil/processed_common_organ_data/merged/split/"

splits_fscil = "/hdd2/cil/running_base/fscil_SS/fscilSS/data/merged/split/"
fscil_ckpt = "/hdd2/cil/running_base/fscil_SS/fscilSS/checkpoints/step/15ss-merged/"
fscil_fts = "/hdd2/cil/running_base/fscil_SS/fscilSS/checkpoints/step/feature_extractors/"

unlb_amos = "/hdd2/cil/running_base/fscil_SS/fscilSS/data/unlabeled/amos/"
unlb_bcv = "/hdd2/cil/running_base/fscil_SS/fscilSS/data/unlabeled/bcv_ts/"


data_exmp1 = "/hdd2/cil/running_base/fscil_SS/fscilSS/data/merged/dataset/images/exemplar1/"
ann_exmp1 = "/hdd2/cil/running_base/fscil_SS/fscilSS/data/merged/dataset/annotations/exemplar1/"

data_exmp2 = "/hdd2/cil/running_base/fscil_SS/fscilSS/data/merged/dataset/images/exemplar2/"
ann_exmp2 = "/hdd2/cil/running_base/fscil_SS/fscilSS/data/merged/dataset/annotations/exemplar2/"

data_copy = "/hdd2/cil/running_base/fscil_SS/fscilSS/data/merged/dataset/images/copy/"

os.makedirs(data_copy, exist_ok=True)
os.makedirs(data_exmp1, exist_ok=True)
os.makedirs(ann_exmp1, exist_ok=True)


os.makedirs(data_exmp2, exist_ok=True)
os.makedirs(ann_exmp2, exist_ok=True)

###############################################################################################################
###############################################################################################################
def step_zero_no_replay(opts,task,logger,rank,device):
    print("Step 0 ")
    print('Base Step ended')
    
    
    
    
    
    
    
    
###############################################################################################################################    
###############################################################################################################################    
    
def step_more_with_replay(opts,task,logger,rank,device, nearest_neigh=True, label_now=False):
    
    
    print("Step ",opts.step)
    print('Labelling will follow')
    task_name = f"{opts.task}-{opts.dataset}"
    name = f"{opts.name}-s{task.nshot}-i{task.ishot}" if task.nshot != -1 else f"{opts.name}"
    
    
    if opts.step==1 and nearest_neigh:
        shutil.rmtree(data_exmp1)
        shutil.rmtree(ann_exmp1)
        shutil.rmtree(data_copy)
        
    os.makedirs(data_copy, exist_ok=True)
    os.makedirs(data_exmp1, exist_ok=True)
    os.makedirs(ann_exmp1, exist_ok=True)
    
    if opts.step==2 and nearest_neigh:
        shutil.rmtree(data_exmp2)
        shutil.rmtree(ann_exmp2)
        shutil.rmtree(data_copy)
    os.makedirs(data_copy, exist_ok=True)
    os.makedirs(data_exmp2, exist_ok=True)
    os.makedirs(ann_exmp2, exist_ok=True)
    
    if opts.step==1:
        model_path = fscil_fts+'FT-s5-i0_1_newUNET_dynamic.pth'
        model_label_path = fscil_ckpt+'FT-s5-i0_1_newUNET_dynamic.pth'
    if opts.step==2:
        model_path = fscil_fts+'FT-s5-i0_2_newUNET_dynamic.pth'
        model_label_path = fscil_ckpt+'FT-s5-i0_2_newUNET_dynamic.pth'
        
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
    
    train_split = splits_path+"train.txt"
    sample_list = open(train_split).readlines()
    #print(sample_list)

    train_pickle = splits_path+'inverse_dict_new_train.pkl'
    class_to_images = pkl.load(open(train_pickle, 'rb'))
    print(class_to_images)

    if opts.step==1:
        step_class = [16,17,18,19,20]
        step_class_num = 21
        unlb_path = unlb_amos
    else:
        step_class = [21,22,23,24,25,26]
        step_class_num = 27
        unlb_path = unlb_bcv

    unlb_list = os.listdir(unlb_path)


    step_samples = set()
    for cl in step_class:
        for sm in class_to_images[cl]:
            step_samples.add(sm)
    step_samples = list(step_samples)

    #print(base_samples)

    for i in range(0,len(step_samples)):
        step_samples[i] = sample_list[step_samples[i]][:-1]+".nii.gz"
    print("Step {} = ".format(opts.step),step_samples)

    
    
    
    
    if nearest_neigh:

        model = get_method(opts, task, device, logger, config_vit)
        
        #print(model)
        
        # Put the model on GPU  // Make it always, also after train, to remediate the cool_down method
        checkpoint = torch.load(model_path, map_location="cpu")

        # checkpoint = torch.load("cil/FSS/checkpoints/step/7ss-verse/27_july_FT_0.pth")

        model.load_dict_full_model(checkpoint["model_state"]["model"])
        print("Model restored from {}".format(model_path))
        logger.info(f"*** Model restored from {model_path}")
        del checkpoint




        print("Checking for Nearest neighbours of training samples only")

        

        Rnd_G = RandomGenerator(output_size=[128,160,96], mode = 'train')


        ## step img, lbl
        for i in range(0,len(step_samples)):
            print("Step step sample = ",step_samples[i])
            step_img = read_img(data_train+step_samples[i])#.to(device)
            step_lbl = read_img(ann_train+step_samples[i])#.to(device)
            thresh_step = (int(step_class_num) - 1) + 0.5
            step_lbl[step_lbl < 0.5] = 0.0  # maybe some voxels is a minus value
            step_lbl[step_lbl > thresh_step] = 255
            sample_step_unedit = {'image': step_img, 'label': step_lbl, 'case_name':step_samples[i][:-7]}

            while True:
                sample_step_crop = Rnd_G(sample_step_unedit)

                step_labels = torch.unique(sample_step_crop['label'])
                print(step_labels)

                if len(step_labels)>0:
                    break

            #learned_proto = get_protoype()

            img, lbl = sample_step_crop['image'], sample_step_crop['label']
            img, lbl = img.to(device), lbl.to(device)



            print("Loading unlabeled samples ")
            step_dist=model.nearest_sample(Rnd_G, img, lbl, unlb_path, unlb_list)
            del img, lbl

            step_dist_sorted = sorted(step_dist.items(), key=lambda x:x[1])
            print(step_dist_sorted)

            ## Nearest Neighbours
            near_samples = 2
            step_div_list =[]

            for ij in range(0,near_samples):
                step_div_list.append(step_dist_sorted[ij][0])
            print(step_div_list)

            for uni in range(0,len(step_div_list)):
                shutil.copy(unlb_path+step_div_list[uni],data_copy)
            
            unlb_list = list(set(unlb_list) - set(step_div_list))  
            
            if opts.step==1 and len(os.listdir(data_copy))>=10:
                break
                
            if opts.step==2 and len(os.listdir(data_copy))>=12:
                break
            ## Copy the unlabeled samples image into exemplar 1 for step 1

        del model
        model=None
        print("Nearest neighbours selected")
    
    if label_now:
        print("\n\n Labelling of nearest neigbours will follow")
        label_model = get_method(opts, task, device, logger, config_vit)

        #print(model)
        
        # Put the model on GPU  // Make it always, also after train, to remediate the cool_down method
        checkpoint_label = torch.load(model_label_path, map_location="cpu")

        # checkpoint = torch.load("cil/FSS/checkpoints/step/7ss-verse/27_july_FT_0.pth")

        label_model.load_dict_full_model(checkpoint_label["model_state"]["model"])
        print("Model for label restored from {}".format(model_label_path))
        logger.info(f"*** Model label restored from {model_label_path}")
        del checkpoint_label

        if opts.step==1:
            data_to_label = data_exmp1
            save_to_label = ann_exmp1
        if opts.step==2:
            data_to_label = data_exmp2
            save_to_label = ann_exmp2
        
        Rnd_G = RandomGenerator(output_size=[128,160,96], mode = 'train')
        unlb_for_label = os.listdir(data_copy)
        print(unlb_for_label)
        for i in range(0, len(unlb_for_label)):
            print("Unlabeled sample = ",unlb_for_label[i])
            step_img = read_img(data_copy+unlb_for_label[i])#.to(device)
            step_lbl = read_img(ann_train+step_samples[i])
            #thresh_step = (int(step_class_num) - 1) + 0.5
            #step_lbl[step_lbl < 0.5] = 0.0  # maybe some voxels is a minus value
            #step_lbl[step_lbl > thresh_step] = 255
            sample_step_unedit = {'image': step_img, 'label': step_lbl, 'case_name':unlb_for_label[i][:-7]}

            sample_step_crop = Rnd_G(sample_step_unedit)
            
            img_unlb_to = sample_step_crop['image']
            img_unlb_to = img_unlb_to.to(device)
            
            print(img_unlb_to.shape)
            #img_to_label = torch.from_numpy(img_to_label).to(device)
            label_model.save_labels_step(img_unlb_to,data_to_label, save_to_label, unlb_for_label[i])
            del img_unlb_to, sample_step_unedit, sample_step_crop, step_lbl
        del label_model
    
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
        step_more_with_replay(opts,task,logger,rank,device,nearest_neigh=True, label_now=False)
        torch.cuda.empty_cache()
        step_more_with_replay(opts,task,logger,rank,device,nearest_neigh=False, label_now=True)
        torch.cuda.empty_cache()
    
    

if __name__ == '__main__':
    parser = argparser.get_argparser()
    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)


    main(opts)

