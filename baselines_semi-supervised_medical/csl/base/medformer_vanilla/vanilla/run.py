from datetime import datetime
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
    

    print("train_loader")
    for i_batch, sampled_batch in enumerate(train_loader):
            print("Patient ids", sampled_batch['case_name'])
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            print(image_batch.shape,label_batch.shape)
            print(np.unique(label_batch))
    
    print("\n\n\n\n\nval_loader")
    for i_batch, sampled_batch in enumerate(val_loader):
            print("Patient ids", sampled_batch['case_name'])
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            print(image_batch.shape,label_batch.shape)
            print(np.unique(label_batch))

    train_iterations = 1 if task.step == 0 else 20 // task.nshot
    if opts.iter is not None:
        opts.epochs = opts.iter // (len(train_loader) * train_iterations)
    opts.max_iter = opts.epochs * len(train_loader) * train_iterations
    if opts.max_iter == 0:
        opts.max_iter = 1
        
    # logger.info(f"Total batch size is {min(opts.batch_size, len(train_dst)) * world_size}")
    logger.info(f"Total batch size is {min(opts.batch_size, len(train_dst))}")
    logger.info(f"Train loader contains {len(train_loader)} iterations per epoch, multiplied by {train_iterations}")
    logger.info(f"Total iterations are {opts.max_iter}, corresponding to {opts.epochs} epochs")
    
    # xxx Set up model
    logger.info(f"Backbone: {opts.backbone}")

    model = get_method(opts, task, device, logger, config_vit)

    # if opts.n_gpu > 1:
    #     model = torch.nn.DataParallel(model.model)
        
    # parameter_num, net = print_network(model)
    # logger.info("Total number of network parameters: {}".format(parameter_num))
    # logger.info("network structure: {}".format(net))
    logger.info(f"[!] Model made with{'out' if opts.no_pretrained else ''} pre-trained")
    # IF step > 0 you need to reload pretrained
    if task.step > 0:
        step_ckpt = get_step_ckpt(opts, logger, task_name, name)
        assert step_ckpt is not None, "Step checkpoint is None!"
        # print(step_ckpt['model_state']["model"].keys())

        old_model = step_ckpt['model_state']["model"]
        if opts.load_weight_strategy:
            model.load_dict_imprint(old_model, strict=False)  # False because of incr. classifiers
        else:
            model.load_dict(old_model, strict=False)
        print(f"[!] Previous model loaded from {step_ckpt['path']}")
        logger.info(f"[!] Previous model loaded from {step_ckpt['path']}")
        # clean memory
        del step_ckpt

    for name, param in model.model.named_parameters():
        print(name, param.requires_grad)
        
    # xxx Model warm up
    logger.debug(model)
    if not opts.vanila:
        if task.step > 0 and not opts.continue_ckpt and opts.ckpt is None:
            logger.info("Warm up lap!")
            model.warm_up(train_loader, old_model)
    
    # put the model on DDP
    # model.distribute()
    
    # xxx Handle checkpoint to resume training
    cur_epoch = 0
    if opts.continue_ckpt:
        opts.ckpt = checkpoint_path
        print("opts.ckpt", opts.ckpt)
    if opts.ckpt is not None:
        assert os.path.isfile(opts.ckpt), "Error, ckpt not found. Check the correct directory"
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
        cur_epoch = checkpoint["epoch"] + 1 if not opts.born_again else 0
        print("cur_epoch ", cur_epoch)
        model.load_dict_state(checkpoint['model_state']["model"])
        print("[!] Model restored from %s" % opts.ckpt)
        logger.info("[!] Model restored from %s" % opts.ckpt)
        del checkpoint
    else:
        logger.info("[!] Train from the beginning of the task")

    # xxx Train procedure
    # print opts before starting training to log all parameters
    logger.add_table("Opts", vars(opts))

    label2color = utils.Label2Color(cmap=utils.color_map(opts.dataset))  # convert labels to images
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])  # de-normalization for original images

    train_metrics = StreamSegMetrics(len(task.get_order()), task.get_n_classes()[0])
    val_metrics = StreamSegMetrics(len(task.get_order()), task.get_n_classes()[0])
    results = {}

    # check if random is equal here.
    logger.print(torch.randint(0, 100, (1, 1)))

    max_epoch = opts.max_epochs
    max_iterations = opts.max_epochs * len(train_loader)          

    logger.info("{} iterations per epoch. {} max iterations ".format(len(train_loader), max_iterations))
    writer = SummaryWriter(snapshot_path + '/log')
    # load progress bar
    iterator = tqdm(range(max_epoch), ncols=70)
    optimizer = torch.optim.AdamW(model.model.parameters(), lr=5e-4, eps=1e-8, betas=(0.9, 0.999), weight_decay=1e-5)
    
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 44, eta_min=0, last_epoch=-1, verbose=False)          
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 25, eta_min=5e-6)
    #  CosineAnnealingWarmRestarts
    # record best model with highest train and val dice
    highest_train_dice = 0
    epo_train = 1
    highest_val_dice = 0
    epo_val = 1
    # these two weight need to be changed
    
    dice_weight = [0.25] + [0.75] * (opts.num_classes - 1)
    ce_weight = [0.25] + [0.75] * (opts.num_classes - 1)
    print("Num classes:", opts.num_classes)
    # train/val here
    print("Total epochs ",max_epoch)
    while cur_epoch <= max_epoch and not opts.test:
        # =====  Train  =====
        start = time.time()
        class_dice_train, train_dice = model.train(optimizer, dice_weight, ce_weight, cur_epoch=cur_epoch, train_loader=train_loader,
                                 metrics=train_metrics, print_int=opts.print_interval,
                                 n_iter=train_iterations, snapshot_path = snapshot_path)
        # train_score = train_metrics.get_results()
        end = time.time()
        
        mean_train_dice = mean_dice(task, class_dice_train) 
        logger.info("Mean Train dice: {}".format(mean_train_dice))
        # =====  Validation  =====
        # if (cur_epoch + 1) % opts.val_interval == 0:
        logger.info("validate on val set...")
        class_dice_val, val_dice = model.validate(dice_weight, ce_weight, loader=val_loader, metrics=val_metrics, ret_samples_ids=None, cur_epoch=cur_epoch, snapshot_path = snapshot_path)
        # val_score = val_metrics.get_results()
        mean_val_dice = mean_dice(task, class_dice_val) 
        logger.print("Mean Val dice: {}".format(mean_val_dice))
        logger.print("Done validation")
        # logger.info(f"End of Validation {cur_epoch}/{opts.epochs}, Validation dice={val_dice}")

        if cur_epoch % 50 == 0:
            print("*** Test the model on previous classes...")

            class_dice_test, test_dice = model.test(dice_weight, ce_weight, loader=test_loader_all, metrics=val_metrics, ret_samples_ids=None, cur_epoch=cur_epoch, snapshot_path = snapshot_path)
            dice_dict_test = {x: class_dice_test[x] for x in range(opts.num_classes)}
            print("***** Test Class wise Dice:", dice_dict_test)
            logger.print("Test Dice: {}".format(mean_dice_test1(task, class_dice_test)))
           
            #curr_path = "/hdd3/cil/baselines/set2_baselines/vanilla_model/vanilla/"
            with open('results/dice_step_{}_{}_test_{}_dynamic.txt'.format(task.step, cur_epoch, opts.network_arch), 'w') as file:
                for key, value in dice_dict_test.items():
                    file.write(f'{key}: {value}\n')
        # log_val(logger, val_metrics, val_score, val_loss, cur_epoch)
        
        if mean_train_dice > highest_train_dice:
            highest_train_dice = mean_train_dice
            epo_train = cur_epoch + 1
            highest_class_wise_dice_train = class_dice_train
        if mean_val_dice > highest_val_dice:
            highest_val_dice = mean_val_dice
            epo_val = cur_epoch + 1
            save_ckpt(checkpoint_path, model, cur_epoch)
            highest_class_wise_dice_val = class_dice_val

            # logger.info("save best model to {} at epoch {}, val dice: {}".format(save_mode_path, epo_val, val_dice))

        lr_scheduler.step()
        torch.cuda.empty_cache()

        logger.info('highest train dice: %f at epoch %d, highest val dice : %f at epoch %d' % (
        highest_train_dice, epo_train, highest_val_dice, epo_val))

        cur_epoch += 1
    
    print("*** Highest Training Class Wise Dice:{}".format(highest_class_wise_dice_train))
    print("*** Highest Validation Class Wise Dice:{}".format(highest_class_wise_dice_val))

    class_wise_dice_train_dict = store_results(highest_class_wise_dice_train)
    class_wise_dice_val_dict = store_results(highest_class_wise_dice_val)

    
    with open('results/dice_step_{}_train_{}_dynamic.txt'.format(task.step, opts.network_arch), 'w') as file:
        for key, value in class_wise_dice_train_dict.items():
            file.write(f'{key}: {value}\n')
        
    with open('results/dice_step_{}_val_{}_dynamic.txt'.format(task.step, opts.network_arch), 'w') as file:
        for key, value in class_wise_dice_val_dict.items():
            file.write(f'{key}: {value}\n')
        
    if not opts.test:
        # =====  Finalize Model  =====
        logger.info("Cooling down...")
        if rank == 0:
            model.cool_down(train_dst)

    # =====  Save Model  =====
    if not opts.test and rank == 0:  # save best model at the last iteration
        save_ckpt(checkpoint_path, model, cur_epoch)
        logger.info("[!] Checkpoint saved.")

    # xxx Test code!
    logger.info("*** Test the model on all seen classes...")
    print("*** Test the model on all seen classes...")
    
    test_dst_all, test_dst_novel = get_dataset(opts, task, train=False)
    print("The length of test set is: {}".format(len(test_dst_all)))

    logger.info(f"Dataset: {opts.dataset}, Test set: {len(test_dst_all)}")
    # make data loader for all classes
    test_loader_all = DataLoader(test_dst_all, batch_size=batch_size, shuffle = False,
                                   num_workers=opts.num_workers, pin_memory=True,      # m_worker: 8    ---->>   4
                             worker_init_fn=worker_init_fn)

    if rank == 0 and opts.sample_num > 0:
        sample_ids = np.random.choice(len(test_loader_all), opts.sample_num, replace=False)  # sample idxs for visual.
        logger.info(f"The samples id are {sample_ids}")
    else:
        sample_ids = None

    # Put the model on GPU  // Make it always, also after train, to remediate the cool_down method
    if opts.test and opts.ckpt is not None:
        checkpoint = torch.load(opts.ckpt, map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # checkpoint = torch.load("/home/cil/FSS/checkpoints/step/7ss-verse/27_july_FT_0.pth")

    model.load_dict_full_model(checkpoint["model_state"]["model"])
    print("Model restored from {}".format(checkpoint_path))
    logger.info(f"*** Model restored from {checkpoint_path}")
    del checkpoint

    class_dice_test, test_dice = model.test(dice_weight, ce_weight, loader=test_loader_all, metrics=val_metrics, ret_samples_ids=None, cur_epoch=cur_epoch, snapshot_path = snapshot_path)
    # class_dice_test, test_dice = model.test(dice_weight, ce_weight, loader=val_loader, metrics=val_metrics, ret_samples_ids=None, cur_epoch=cur_epoch, snapshot_path = snapshot_path)
    print("*** Test Class Wise Dice:{}".format(class_dice_test))
    class_wise_dice_test_dict = store_results(class_dice_test)
    
    with open('results/dice_step_{}_test_{}_dynamic.txt'.format(task.step, opts.network_arch), 'w') as file:
        for key, value in class_wise_dice_test_dict.items():
            file.write(f'{key}: {value}\n')
        
    # val_score = val_metrics.get_results()
    # conf_matrixes = val_metrics.get_conf_matrixes(mode = 'test')
    
    print("Done test on all")
    logger.print("Done test on all")
    
    mean_test_dice = mean_dice_test2(task, class_dice_test)
    print(f"*** End of Test on all, Test Dice={mean_test_dice}")
    logger.info(f"*** End of Test on all, Test Dice={mean_test_dice}")
    
    logger.close()

if __name__ == '__main__':

    # current date and time
    now_time = datetime.now()
    s1 = now_time.strftime("%m/%d/%Y, %H:%M:%S")
    # mm/dd/YY H:M:S format
    print("Start time :", s1)
    
    print("\nGPU Details : ")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))
    print("\n\n",os.getcwd())
    
    parser = argparser.get_argparser()
    opts = parser.parse_args()
    opts = argparser.modify_command_options(opts)


    main(opts)
    
    now_time = datetime.now()
    s1 = now_time.strftime("%m/%d/%Y, %H:%M:%S")
    # mm/dd/YY H:M:S format
    print("End time :", s1)