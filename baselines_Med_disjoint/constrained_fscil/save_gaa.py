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

def get_step_ckpt(opts, logger, task_name, name):
    # xxx Get step checkpoint
    step_checkpoint = None
    if opts.step_ckpt is not None:
        path = opts.step_ckpt
    else:
        if opts.step - 1 == 0:
            path = f"checkpoints/step/{task_name}/{opts.name}_{opts.step - 1}_{opts.network_arch}_dynamic.pth"
        else:
            path = f"checkpoints/step/{task_name}/{name}_{opts.step - 1}_{opts.network_arch}_dynamic.pth"
            #path = f"checkpoints/step/{task_name}/{name}_{opts.step}.pth"
    # generate model from path
    if os.path.exists(path):
        step_checkpoint = torch.load(path, map_location="cpu")
        step_checkpoint['path'] = path
    elif opts.debug:
        logger.info(
            f"[!] WARNING: Unable to find of step {opts.step - 1}! Do you really want to do from scratch?")
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

def main(opts):
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

    checkpoint_path = f"checkpoints/step/{task_name}/{name}_{opts.step}_{opts.network_arch}_dynamic.pth"
    os.makedirs(f"checkpoints/step/{task_name}", exist_ok=True)

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed(opts.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    
    print("Seed:", opts.seed)
    
    
    dataset_name = opts.dataset


    batch_size = opts.batch_size * opts.n_gpu
    
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
    
    train_dst, val_dst = get_dataset(opts, task, train=True)
    test_dst_all, test_dst_novel = get_dataset(opts, task, train=False)

    print("The length of train set is: {}".format(len(train_dst)))
    print("The length of val set is: {}".format(len(val_dst)))
    print("The length of test set is: {}".format(len(test_dst_all)))

    logger.info(f"Dataset: {opts.dataset}, Train set: {len(train_dst)}, Val set: {len(val_dst)}")

    def worker_init_fn(worker_id):
        random.seed(opts.seed)
        
    train_loader = DataLoader(train_dst, batch_size=batch_size, shuffle = True,
                                   num_workers=opts.num_workers, pin_memory=True,      # m_worker: 8    ---->>   4
                             worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dst, batch_size=batch_size, shuffle = True,
                                 num_workers=opts.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    test_loader_all = DataLoader(test_dst_all, batch_size=batch_size, shuffle = False,
                                   num_workers=opts.num_workers, pin_memory=True,      # m_worker: 8    ---->>   4
                             worker_init_fn=worker_init_fn)
    
    

    model = get_method(opts, task, device, logger, config_vit)

    #if task.step > 0:
    tk=opts.step
    opts.step=1
    task_name = f"{opts.task}-{opts.dataset}"
    name = f"{opts.name}-s{task.nshot}-i{task.ishot}" if task.nshot != -1 else f"{opts.name}"
    step_ckpt = get_step_ckpt(opts, logger, task_name, name)
    print(name)
    opts.step=tk
    task_name = f"{opts.task}-{opts.dataset}"
    name = f"{opts.name}-s{task.nshot}-i{task.ishot}" if task.nshot != -1 else f"{opts.name}"
    assert step_ckpt is not None, "Step checkpoint is None!"
    # print(step_ckpt['model_state']["model"].keys())

    old_model = step_ckpt['model_state']["model"]
    if opts.load_weight_strategy:
        model.load_dict_imprint(old_model, strict=False, saving_gaa=True)  # False because of incr. classifiers
    else:
        model.load_dict(old_model, strict=False)
    print(f"[!] Previous model loaded from {step_ckpt['path']}")
    logger.info(f"[!] Previous model loaded from {step_ckpt['path']}")
    # clean memory
    del step_ckpt

    for name, param in model.model.named_parameters():
        print(name, param.requires_grad)
        
    
   
    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # de-normalization for original images
    
    model.save_gaa(dataset=train_dst, step=task.step)


if __name__ == '__main__':
    parser = argparser.get_argparser()
    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)


    main(opts)