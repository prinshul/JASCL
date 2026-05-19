import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import time
import sys
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime as date_and_time
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from tensorboardX import SummaryWriter

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist, decollate_batch, DistributedSampler
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

from model.Universal_model import Universal_model
from dataset.dataloader import get_loader
from utils import loss
from utils.utils import dice_score, check_data, TEMPLATE, NUM_CLASS #get_key,
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR


torch.multiprocessing.set_sharing_strategy('file_system')


def train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE):
    model.train()
    loss_bce_ave = 0
    loss_dice_ave = 0
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    for step, batch in enumerate(epoch_iterator):
        x, y, name = batch["image"].to(args.device), batch["post_label"].float().to(args.device), batch['name']
        logit_map = model(x)

        term_seg_Dice = loss_seg_DICE.forward(logit_map, y, name, TEMPLATE, args.data_txt_path)
        term_seg_BCE = loss_seg_CE.forward(logit_map, y, name, TEMPLATE, args.data_txt_path)
        loss = term_seg_BCE + term_seg_Dice
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        epoch_iterator.set_description(
            "Epoch=%d: Training (%d / %d Steps) (dice_loss=%2.5f, bce_loss=%2.5f)" % (
                args.epoch, step, len(train_loader), term_seg_Dice.item(), term_seg_BCE.item())
        )
        sys.stdout.flush()
        loss_bce_ave += term_seg_BCE.item()
        loss_dice_ave += term_seg_Dice.item()
        torch.cuda.empty_cache()
    print('Epoch=%d: ave_dice_loss=%2.5f, ave_bce_loss=%2.5f' % (args.epoch, loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)))
    sys.stdout.flush()
    return loss_dice_ave/len(epoch_iterator), loss_bce_ave/len(epoch_iterator)

def validation(model, ValLoader, args):
    model.eval()
    dice_list = {}
    for key in TEMPLATE.keys():
        dice_list[key] = np.zeros((2, NUM_CLASS)) # 1st row for dice, 2nd row for count
    for index, batch in enumerate(tqdm(ValLoader)):
        # print('%d processd' % (index))
        image, label, name = batch["image"].cuda(), batch["post_label"], batch["name"]
        print(name, image.shape)
        with torch.no_grad():
            pred = sliding_window_inference(image, (args.roi_x, args.roi_y, args.roi_z), 1, model)
            pred_sigmoid = F.sigmoid(pred)
        
        B = pred_sigmoid.shape[0]
        for b in range(B):
            template_key = get_key(name[b])
            organ_list = TEMPLATE[template_key]
            for organ in organ_list:
                dice_organ = dice_score(pred_sigmoid[b,organ-1,:,:,:], label[b,organ-1,:,:,:].cuda())
                dice_list[template_key][0][organ-1] += dice_organ.item()
                dice_list[template_key][1][organ-1] += 1
    
    ave_organ_dice = np.zeros((2, NUM_CLASS))
    if args.local_rank == 0:
        with open('out/'+args.log_name+f'/val_{args.epoch}.txt', 'w') as f:
            for key in TEMPLATE.keys():
                organ_list = TEMPLATE[key]
                content = 'Task%s| '%(key)
                for organ in organ_list:
                    dice = dice_list[key][0][organ-1] / dice_list[key][1][organ-1]
                    content += '%s: %.4f, '%(ORGAN_NAME[organ-1], dice)
                    ave_organ_dice[0][organ-1] += dice_list[key][0][organ-1]
                    ave_organ_dice[1][organ-1] += dice_list[key][1][organ-1]
                print(content)
                f.write(content)
                f.write('\n')
            content = 'Average | '
            for i in range(NUM_CLASS):
                content += '%s: %.4f, '%(ORGAN_NAME[i], ave_organ_dice[0][organ-1] / ave_organ_dice[1][organ-1])
            print(content)
            f.write(content)
            f.write('\n')
            
            


def process(args):
    rank = 0

    if args.dist:
        dist.init_process_group(backend="nccl", init_method="env://")
        rank = args.local_rank
    args.device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(args.device)

    # prepare the 3D model
    model = Universal_model(img_size=(args.roi_x, args.roi_y, args.roi_z),
                    in_channels=1,
                    out_channels=NUM_CLASS,
                    backbone=args.backbone,
                    encoding=args.trans_encoding
                    )

    #Load pre-trained weights
    if args.pretrain is not None:
        model.load_params(torch.load(args.pretrain)["state_dict"])

    if args.trans_encoding == 'word_embedding': ## txt_encoding.pth
        word_embedding_path = args.word_embedding[:-4]+"_"+str(NUM_CLASS)+".pth"
        print("Word embeddings from ",word_embedding_path)
        word_embedding = torch.load(word_embedding_path)
        model.organ_embedding.data = word_embedding.float()
        print('load word embedding')

    model.to(args.device)
    model.train()
    if args.dist:
        model = DistributedDataParallel(model, device_ids=[args.device])

    # criterion and optimizer
    # loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    loss_seg_DICE = loss.DiceLoss(num_classes=NUM_CLASS).to(args.device)
    loss_seg_CE = loss.Multi_BCELoss(num_classes=NUM_CLASS).to(args.device)
    if args.backbone == 'unetpp':
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,
                              nesterov=False, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=args.warmup_epoch, max_epochs=args.max_epoch)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    
    if args.resume:
        checkpoint = torch.load(args.resume)
        #print(checkpoint)
        if args.dist:
            model.load_state_dict(checkpoint['net'])
        else:
            store_dict = model.state_dict()
            model_dict = checkpoint['net']
            
            new_state = {}
            for key,value in model_dict.items():
                #if 'running_mean' not in key: 
                    #if 'running_var' not in key:
                if 'organ_embedding' in key:
                    new_state[key]=word_embedding
                elif 'num_batches_tracked' not in key:
                    new_state[key] = value
            
            #for key in model_dict.keys():
                #if 'organ_embedding' not in key:
                    #store_dict['.'.join(key.split('.')[1:])] = model_dict[key]
                    
            model.load_state_dict(new_state)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        #args.epoch = checkpoint['epoch']
        #scheduler.load_state_dict(checkpoint['scheduler'])
        
        print('success resume from ', args.resume)

    torch.backends.cudnn.benchmark = True
    
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    
    train_loader, train_sampler = get_loader(args)
    
    
    print("\n\nlog_name : ",args.log_name)
    print("data_root_path : ",args.data_root_path)
    print("data_txt_path : ",args.data_txt_path)
    print("NUM_CLASS : ",NUM_CLASS)
    print("backbone : ",args.backbone)
    print("batch_size : ",args.batch_size)
    print("epoch : ",args.epoch)
    print("max_epoch : ",args.max_epoch)
    print("resume : ",args.resume)
    print("pretrain : ",args.pretrain)
    print("Model and results at : ",args.log_name+"/"+args.data_txt_path)
    sys.stdout.flush()
    
    if rank == 0:
        writer = SummaryWriter(log_dir=args.log_name+"/"+args.data_txt_path+"/logs")
        print('Writing Tensorboard logs to ', args.log_name+"/"+args.data_txt_path+"/logs")

    while args.epoch <= args.max_epoch:
        if args.dist:
            dist.barrier()
            train_sampler.set_epoch(args.epoch)
        scheduler.step()

        loss_dice, loss_bce = train(args, train_loader, model, optimizer, loss_seg_DICE, loss_seg_CE)
        if rank == 0:
            writer.add_scalar('train_dice_loss', loss_dice, args.epoch)
            writer.add_scalar('train_bce_loss', loss_bce, args.epoch)
            writer.add_scalar('lr', scheduler.get_lr(), args.epoch)

        if (args.epoch % args.store_num == 0 and args.epoch != 0) and rank == 0:
            checkpoint = {
                "net": model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                "epoch": args.epoch
            }
            if not os.path.isdir(args.log_name+"/"+args.data_txt_path):
                os.mkdir(args.log_name+"/"+args.data_txt_path)
            torch.save(checkpoint,args.log_name+'/'+args.data_txt_path + '/epoch_' + str(args.epoch) + '.pth')
            print('save model success at ',args.epoch)

        args.epoch += 1
    
    if args.dist:
        dist.destroy_process_group()

def main():
    parser = argparse.ArgumentParser()
    ## for distributed training
    parser.add_argument('--dist', dest='dist', type=bool, default=False,
                        help='distributed training or not')
    parser.add_argument("--local_rank", type=int)
    parser.add_argument("--device")
    parser.add_argument("--epoch", default=0)
    parser.add_argument("--seed", default=42)
    ## logging
    parser.add_argument('--log_name', default='trained_checkpoints', help='The path resume from checkpoint')
    ## model load
    parser.add_argument('--backbone', default='unet', help='backbone [swinunetr or unet or dints or unetpp]')
    parser.add_argument('--resume', default=None, help='The path resume from checkpoint')
    parser.add_argument('--pretrain', default=None,  #swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt
                        help='The path of pretrain model. Eg, ./pretrained_weights/swin_unetr.base_5000ep_f48_lr2e-4_pretrained.pt')
    parser.add_argument('--trans_encoding', default='word_embedding', 
                        help='the type of encoding: rand_embedding or word_embedding')
    parser.add_argument('--word_embedding', default='./pretrained_weights/txt_encoding.pth', 
                        help='The path of word embedding')
    ## hyperparameter
    parser.add_argument('--max_epoch', default=200, type=int, help='Number of training epoches')
    parser.add_argument('--store_num', default=50, type=int, help='Store model how often')
    parser.add_argument('--warmup_epoch', default=100, type=int, help='number of warmup epochs')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-5, help='Weight Decay')
    ## dataset
    parser.add_argument('--dataset_list', nargs='+', default=['PAOT_123457891213', 'PAOT_10_inner']) # 'PAOT', 'felix'
    ### please check this argment carefully
    ### PAOT: include PAOT_123457891213 and PAOT_10
    ### PAOT_123457891213: include 1 2 3 4 5 7 8 9 12 13
    ### PAOT_10_inner: same with NVIDIA for comparison
    ### PAOT_10: original division
    parser.add_argument('--data_root_path', \
    default='med_3D/dataset/set2/set2_full/merged/',\
    help='data root path')
    parser.add_argument('--data_txt_path', default='step1_train', help='data txt path')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--num_workers', default=4, type=int, help='workers numebr for DataLoader')
    parser.add_argument('--a_min', default=-175, type=float, help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=250, type=float, help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float, help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float, help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float, help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int, help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int, help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int, help='roi size in z direction')
    parser.add_argument('--num_samples', default=2, type=int, help='sample number in each ct')

    parser.add_argument('--phase', default='train', help='train or validation or test')
    parser.add_argument('--uniform_sample', action="store_true", default=False, help='whether utilize uniform sample strategy')
    parser.add_argument('--datasetkey', nargs='+', default=['01', '02', '03', '04', '05', 
                                            '07', '08', '09', '12', '13', '10_03', 
                                            '10_06', '10_07', '10_08', '10_09', '10_10'],
                                            help='the content for ')
    parser.add_argument('--cache_dataset', action="store_true", default=False, help='whether use cache dataset')
    parser.add_argument('--cache_rate', default=0.005, type=float, help='The percentage of cached data in total')

    args = parser.parse_args()
    
    process(args=args)

if __name__ == "__main__":

    now_time = date_and_time.now()
    s1 = now_time.strftime("%d/%m/%Y, %H:%M:%S")
    # mm/dd/YY H:M:S format
    print("\n\nStart time :", s1)
    
    print("\nGPU Details : ")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))

    main()
    
    
    # current date and time
    now_time = date_and_time.now()
    s1 = now_time.strftime("%d/%m/%Y, %H:%M:%S")
    # mm/dd/YY H:M:S format
    print("\n\nEnd time :", s1)
# python -m torch.distributed.launch --nproc_per_node=2 --master_port=1234 train.py --dist True --uniform_sample
