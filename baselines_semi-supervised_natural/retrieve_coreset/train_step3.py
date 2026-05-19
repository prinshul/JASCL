'''
Training protocol for 1st dataset as per our algorithm: (This helps reparameterize the architecture in subsequent steps)
-> init encoder with imagenet pretrained weights.
-> add the DS layers in encoder
-> train the entire architecture on 1st step dataset without freezing any layers.

'''
import os
import random
import time
import sys
import numpy as np
import torch
import math
import shutil
import re
from datetime import datetime
from PIL import Image, ImageOps
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage

from collections import OrderedDict
from dataset import VOC12, cityscapes, IDD, BDD100k
from transform import Relabel, ToLabel, Colorize
import itertools
#import config_task
import pickle
import importlib
from iouEval import iouEval, getColorEntry

import torch.nn as nn
import torch.nn.functional as F
#from models.erfnet_RA_parallel import Net as Net_RAP
# from erfnet_RA_series import Net as Net_RAS
# from erfnet_RCM import Net as Net_RCM
# from erfnet_bn import Net as Net_BN
# from erfnet_onlyRAP import Net as Net_onlyRAP
from models.unet import UNet
from models.deeplabv3 import DeepLab
from models.deeplabv3.backbone import Xception

from shutil import copyfile
from torch.utils.tensorboard import SummaryWriter
from fix_seed import *
import sys
sys.path.insert(1, "semi_fscil/natural/codes/")
from shared_quant import *
# from torchsummary import summary
import warnings
warnings.filterwarnings('ignore')

'''
NUM_CHANNELS = 3
# default value given, will be overwritten by args.num_classes #cityscapes=20, IDD=27, BDD=20 (same as cityscapes)
#NUM_CLASSES = 20
NUM_CLASSES = 11 # Step 1

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()
'''
# Augmentations - different function implemented to perform random augments on both image and target
random.seed(seed_current)
np.random.seed(seed_current)
torch.manual_seed(seed_current)
torch.cuda.manual_seed(seed_current)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

class MyCoTransform(object):
    def __init__(self, augment=True,step_train=False,height=512, width=1024):
        self.augment = augment
        self.height = height
        self.width = width
        self.step_train = step_train
        pass

    def __call__(self, input, target):
        input = Resize([self.height, self.width], Image.BILINEAR)(input)
        target = Resize([self.height, self.width], Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

            # Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2)
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX, transY, 0, 0), fill=0)
            target = ImageOps.expand(target, border=(transX, transY, 0, 0),
                                     fill=255)  # pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))

        input = ToTensor()(input)
        target = ToLabel()(target)
        # print('relabeling 255 as: ', NUM_CLASSES-1)
        if self.step_train:
            target = Relabel(0, NUM_CLASSES - 1)(target)
        target = Relabel(255, NUM_CLASSES - 1)(target)

        return input, target



class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


def train(args, model, start_epoch, best_acc):
    global NUM_CLASSES, NUM_CLASSES_OLD
    NUM_CLASSES = args.num_classes[args.current_task]
    NUM_CLASSES_OLD = args.num_classes[args.current_task-1]
    print('NUM_CLASSES: ', NUM_CLASSES)


    

    tf_dir = 'runs_{}_{}_{}_{}{}_step{}'.format(
        args.dataset, args.model, args.num_epochs, args.batch_size, args.model_name_suffix, len(args.num_classes))
    writer = SummaryWriter('Adaptations/' + tf_dir)

    data_name = args.dataset

    weight_step3 = weights_step3

    weight_step3[NUM_CLASSES - 1] = 0
    


    co_transform = MyCoTransform(augment=True, step_train=True, height=args.height, width=args.width)  # 1024)
    co_transform_val = MyCoTransform(augment=False, step_train=True, height=args.height, width=args.width)  # 1024)
    
    
    ########################################################################
    ################## Loading step 3 IDD repeat files ############################
    
    
    nshot = str(args.nshot)
    print("\nTaking files from path ",shots_datadir+'nshot_'+nshot)
    
    with open(shots_datadir+'nshot_'+nshot+'/IDD_train_step3.pkl', 'rb') as file: 
        step3_idd_rep_train_files=pickle.load(file)
    
    with open(shots_datadir+'nshot_'+nshot+'/IDD_val_step3.pkl', 'rb') as file: 
        step3_idd_rep_val_files = pickle.load(file) 
        
    with open(shots_datadir+'nshot_'+nshot+'/IDD_unlabel_step3.pkl', 'rb') as file: 
        step3_idd_unlabel_files = pickle.load(file)
       
    dataset_step3_train = IDD(IDD_datadir,step3_idd_rep_train_files, co_transform, 'train')
    dataset_step3_val = IDD(IDD_datadir,step3_idd_rep_val_files, co_transform_val, 'train')
    dataset_unlabel = IDD(IDD_datadir,step3_idd_unlabel_files, co_transform_val, 'train')
    
    print("Few Shot per class in Step 3 ",args.nshot)
    print("Total files in Step 3 IDD train : ",len(step3_idd_rep_train_files))
    print("Total files in Step 3 IDD val : ",len(step3_idd_rep_val_files))
    print("Total files in Step 3 IDD unlabled : ",len(step3_idd_unlabel_files))
    
    
    
    ########################################################################
    ########################################################################
    
    
    
    # train_loader, train criterion
    print('loading new data for train in step3_mix')
    if  data_name == 'step3_mix':
        print('taking IDD in step 3')
        loader = DataLoader(dataset_step3_train, num_workers=args.num_workers,
                        batch_size=args.batch_size, shuffle=True)
        loader_val = DataLoader(dataset_step3_val, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=False)
        loader_unlabel = DataLoader(dataset_unlabel, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=False)
    
    print("Total length of train loader ",len(loader))
    print("Total length of val loader ",len(loader_val)) 
    print("Total length of unlabeled loader ",len(loader_unlabel))     
    
    if args.cuda:
        class_weights = weight_step3.to(device)
        class_weights = torch.clamp(class_weights, 0.1, 10.0)

    print('\n\n\n')
    for name, m in model.named_parameters():
        print(name, m.requires_grad)

    

    automated_log_path = savedir + "/automated_log.txt"
    modeltxtpath = savedir + "/model.txt"

    if (not os.path.exists(automated_log_path)):  # dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))
    
    ########################################################################
    ########################################################################
    
    
    
    criterion = nn.CrossEntropyLoss(ignore_index=NUM_CLASSES - 1, weight=class_weights)
    #criterion = UnbiasedCrossEntropy(old_cl=NUM_CLASSES_OLD-1, ignore_index=NUM_CLASSES - 1, reduction='none')
    print(type(criterion))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=4e-5)

    lr_update = lambda epoch: (1 - epoch / args.num_epochs) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_update)
    
    ## KL divergence loss
    #kl_loss = UnbiasedKnowledgeDistillationLoss(alpha=args.alpha)
    #kl_loss = kl_loss.cuda()
    input_shape = [512,1024]
    current_classes = NUM_CLASSES - NUM_CLASSES_OLD
    unlabeled_imgs = current_classes*args.subset_size
    unlabeled_batches = int(unlabeled_imgs/4)
    
    print("args.fix_thresh ",args.fix_thresh)
    print("\nstart_epoch = ",start_epoch,"\nbest_acc = ",best_acc)

    for epoch in range(start_epoch, args.num_epochs+1):
        # ensure its set to the correct #classes for training the current dataset
        
        print("-----TRAINING - EPOCH---", epoch, "-----")

        scheduler.step(epoch)  # scheduler 2

        epoch_loss = []
        time_train = []
        e_ce_loss = []
        #e_kld_loss = []

        doIouTrain = args.iouTrain

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        #model_old.eval()
        
        #KLD_loss = torch.tensor(0.)
        labeled_grad = None 
        random_step = random.sample(range(1, len(loader)), 1)[0]
        
        for step, (images, labels) in enumerate(loader):
            if epoch == start_epoch:
                print('image size new: ', images.size())
                print('labels size new: ', labels.size())
                print('labels are: ', np.unique(labels.numpy()))
                # writer.add_graph(model(), images.cuda(), True) #not working (Segmentation fault (core dumped))
            
            start_time = time.time()
            if args.cuda:
                inputs = images.cuda()
                targets = labels.cuda()

            outputs,_ = model(inputs)
            #print('outputs',outputs.shape)
            
            
            #with torch.no_grad():
            #    # pass same input through the old model as it is, calc KLD as KD between old CS and new CS ; and backprop only thru the enc shared weights.
            #    outputs_prev_model = model_old(inputs)

            ce_loss = criterion(outputs, targets[:, 0])  # cross entropy, main classification loss
            ce_loss = ce_loss.mean() 
            
            # KLD on the output probability distributions of the teacher (outputs_prev_model) and student (outputs_prev_task)
            #KLD_loss = kl_loss(outputs, outputs_prev_model)

            # probably also compute kld on the intermediate feature maps (output of encoder) - not done for now.

            total_loss = ce_loss #+ args.lambdac * KLD_loss
            
            
            
            optimizer.zero_grad()
            total_loss.backward()  
            
            if step == random_step:
                #print(optimizer.param_groups)
                labeled_grad = optimizer.param_groups[0]['params'][-2]
                
                
            optimizer.step()

            epoch_loss.append(total_loss.item())
            time_train.append(time.time() - start_time)
            e_ce_loss.append(ce_loss.item())
            #e_kld_loss.append(KLD_loss.item())
        
            if (doIouTrain):
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                # print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))
        
        
        
        if epoch%args.pseudo_label==0:
            print("\nSubset selection from Unlabeled samples - ",unlabeled_batches," batches.")
            start_sel_time = time.time()
            
            grad_prods = []
            #with torch.no_grad():
            for step_un_sel, (images_un_sel, labels_un_sel) in enumerate(loader_unlabel):
                
                if args.cuda:
                    inputs_un_sel = images_un_sel.cuda()
                    #targets_un_sel = labels_un_sel.cuda()
                    
                outputs_un_sel, _ = model(inputs_un_sel)
                
                out_conv = torch.argmax(outputs_un_sel, axis=1).unsqueeze(1)
                #print("out_conv ",out_conv.shape)
                #print(out_conv.shape)
                
                ce_loss = criterion(outputs_un_sel, out_conv[:, 0])  # cross entropy, main classification loss
                ce_loss = ce_loss.mean() 
                total_loss = ce_loss
                
                params = OrderedDict(model.named_parameters())
                grads = torch.autograd.grad(total_loss,
                                params.values(),
                                create_graph=False)
                
                #print(len(grads))
                unlabeled_grad = grads[-2]
                
                #for (name, param), grad in zip(params.items(), grads):
                    #updated_params[name] = param - step_size[name] * grad
                #    print(name,grad.shape)
                
                #print(len(grads[-1]))
                #print(labeled_grad.shape)
                #print(unlabeled_grad.shape)
                
                gains = torch.matmul(labeled_grad, unlabeled_grad)
                #print(gains.shape)
                #print(gains.sum())
                #print(gains.sum().item())
                grad_prods.append(gains.sum().item())
                #print(grads[-1].shape)
                del params, grads, total_loss, ce_loss, outputs_un_sel, out_conv, inputs_un_sel
                #sys.exit()
                
            selected_step = sorted(range(len(grad_prods)), key=lambda i: grad_prods[i], reverse=True)[:unlabeled_batches]
            
            print("Time taken for selection : %.4f s" % (time.time() - start_sel_time))
        
            print("\nTraining on selected subset from Unlabeled samples")
            for step_un, (images_un, labels_un) in enumerate(loader_unlabel):
                if step_un not in selected_step: continue
                start_time = time.time()
                if args.cuda:
                    inputs_un = images_un.cuda()
                    #targets_un = labels_un.cuda()
                    
                outputs_un, feats = model(inputs_un)
                
                outputs_un[outputs_un<args.fix_thresh]=0
                out_conv = torch.argmax(outputs_un, axis=1).unsqueeze(1)
                #print("out_conv ",out_conv.shape)
                #print(out_conv.shape)
                
                ce_loss = criterion(outputs_un, out_conv[:, 0])  # cross entropy, main classification loss
                ce_loss = ce_loss.mean() 
                
                # KLD on the output probability distributions of the teacher (outputs_prev_model) and student (outputs_prev_task)
                #KLD_loss = kl_loss(outputs_un, outputs_prev_model)

                # probably also compute kld on the intermediate feature maps (output of encoder) - not done for now.

                total_loss = ce_loss #+ args.lambdac * KLD_loss

                optimizer.zero_grad()
                total_loss.backward()  # should backprop ce_loss in all new Ds and shared params.
                # should backprop the KLD_loss only in the shared encoder params - it will be passed through the DS_CS params but they will be freezed so not updated
                optimizer.step()

                epoch_loss.append(total_loss.item())
                time_train.append(time.time() - start_time)
                e_ce_loss.append(ce_loss.item())
                #e_kld_loss.append(KLD_loss.item())
                
                average = sum(epoch_loss) / len(epoch_loss)
                if args.steps_loss > 0 and step_un % args.steps_loss == 0:
                    average = sum(epoch_loss) / len(epoch_loss)
                    print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})',
                          "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        average_epoch_loss_ce = sum(e_ce_loss) / len(e_ce_loss)
        #average_epoch_loss_kld = sum(e_kld_loss) / len(e_kld_loss)
        print('epoch took: ', sum(time_train))

        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print("EPOCH IoU on TRAIN set: ", iouStr, "%")

        average_loss_val = 0.0
        val_acc = 0.0
        average_loss_val_cs = 0.0  # placeholder var name for old dataset, not always cs
        val_acc_cs = 0.0  # placeholder var name for old dataset, not always cs

        
        print("----- VALIDATING - EPOCH", epoch, "-----")
        # validate current task
        average_loss_val, val_acc = eval(args, model, 
         loader_val, criterion, current_task, args.num_classes, epoch)
        # validate previous (step 1) task
        #average_loss_val_cs, val_acc_cs = eval(
        #    model, loader_val_old, criterion_old, 0, args.num_classes, epoch)
        #print('Step 1 loss and acc: ', average_loss_val_cs, val_acc_cs)

        # logging tensorboard plots - epoch wise loss and accuracy. Not calculating iouTrain as that will slow down training
        info = {'total_train_loss': average_epoch_loss_train, 'ce_loss_train': average_epoch_loss_ce, 'val_loss_{}'.format(
            data_name): average_loss_val, 'val_acc_{}'.format(data_name): val_acc} #, 'val_loss_{}'.format(args.dataset_old): average_loss_val_cs, 'val_acc_{}'.format(args.dataset_old): val_acc_cs}
        print(info)
        
        for tag, value in info.items():
            writer.add_scalar(tag, value, epoch)

        # remember best valIoU and save checkpoint
        if val_acc == 0:
            current_acc = -average_loss_val
        else:
            current_acc = val_acc
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)

        'runs_{}_{}_{}_{}{}_step{}'.format(
            args.dataset, args.model, args.num_epochs, args.batch_size, args.model_name_suffix, len(args.num_classes))

        filenameCheckpoint = savedir + \
            '/checkpoint_{}_{}_{}_{}{}_step{}.pth.tar'.format(
                args.dataset, args.model, args.num_epochs, args.batch_size, args.model_name_suffix, len(args.num_classes))
        filenameBest = savedir + \
            '/model_best_{}_{}_{}_{}{}_step{}.pth.tar'.format(
                args.dataset, args.model, args.num_epochs, args.batch_size, args.model_name_suffix, len(args.num_classes))

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        if (is_best):
            # torch.save(model.state_dict(), filenamebest)
            # print(f'save: {filenamebest} (epoch: {epoch})')
            with open(savedir + "/best.txt", "w") as myfile:
                myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, val_acc))

        # SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        # Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (
                epoch, average_epoch_loss_train, average_loss_val, iouTrain, val_acc, usedLr))

    return(model)


def eval(args, model, dataset_loader, criterion, task, num_classes, epoch):
    # Validate on 500 val images after each epoch of training
    
    #KLD_loss = torch.tensor(0.)
    model.eval()
    #model_old.eval()
    epoch_loss_val = []
    time_val = []
    num_cls = num_classes[task]
    print('number of classes in current task: ', num_cls)
    print('validating task: ', task)
    iouEvalVal = iouEval(num_cls, num_cls-1)

    with torch.no_grad():
        for step, (images, labels) in enumerate(dataset_loader):
            start_time = time.time()
            inputs = images.to(device)
            targets = labels.to(device)
            
            if step == 1:
                print('------------------', outputs.size(), targets.size())

            outputs,_ = model(inputs)
            
            
            #with torch.no_grad():
                # pass same input through the old model as it is, calc KLD as KD between old CS and new CS ; and backprop only thru the enc shared weights.
                #outputs_prev_model = model_old(inputs)
            
            ce_loss = criterion(outputs, targets[:, 0])  # cross entropy, main classification loss
            ce_loss = ce_loss.mean() 
            
            # KLD on the output probability distributions of the teacher (outputs_prev_model) and student (outputs_prev_task)
            #KLD_loss = kl_loss(outputs, outputs_prev_model)

            # probably also compute kld on the intermediate feature maps (output of encoder) - not done for now.

            loss = ce_loss #+ args.lambdac * KLD_loss
            
            epoch_loss_val.append(loss.item())
            time_val.append(time.time() - start_time)

            iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

            if 50 > 0 and step % 50 == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / 6))

    average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

    iouVal = 0
    iouVal, iou_classes = iouEvalVal.getIoU()
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print("EPOCH IoU on VAL set: ", iouVal.numpy()*100, "%")
    print('check val fn, loss, acc: ', average_epoch_loss_val, iouVal)
    return average_epoch_loss_val, iouVal




def test(args, model):
    # Test on testing data
    global NUM_CLASSES
    NUM_CLASSES = args.num_classes[args.current_task]
    print('NUM_CLASSES: ', NUM_CLASSES)
    
    co_transform_test = MyCoTransform(augment=False, height=args.height, width=args.width)  # 1024)
    co_transform_step_train = MyCoTransform(augment=False, step_train=True, height=args.height, width=args.width)  # 1024)
    
    
    ######################### Step 1 Test files #######################################
    test_step1_files = os.listdir(BDD_datadir+'test/labels/')
    print("Total number of test files from Step 1 ",len(test_step1_files))
    
    dataset_test_step1 = BDD100k(BDD_datadir,test_step1_files,co_transform_test,'test')
        
    ######################### Step 2 Test files #######################################
    
    with open(shots_datadir+'city_test_step2.pkl', 'rb') as file: 
        test_step2_files=pickle.load(file)
        
    city_test_step2 = cityscapes(city_datadir,test_step2_files, co_transform_step_train, 'test')
        
    #test_step2_all = torch.utils.data.ConcatDataset([dataset_test_step1, IDD_test_step2])
    print("Total test files in Step 2 city val : ",len(city_test_step2))
    
    
    ######################### Step 3 Test files #######################################
    
    with open(shots_datadir+'IDD_test_step3.pkl', 'rb') as file: 
        test_step3_idd_files=pickle.load(file)
    
    print("Total test files in Step 3 IDD test : ",len(test_step3_idd_files))
    
    IDD_test_step3_rep = IDD(IDD_datadir,test_step3_idd_files,co_transform_step_train, 'test')
    
    
    ######################## Test loader #################################
    
    test_step3_all = torch.utils.data.ConcatDataset([dataset_test_step1,city_test_step2,
                    IDD_test_step3_rep])
    
    
    print("Total test files in Step 3 test : ",len(test_step3_all))
    
    test_loader_step3 = DataLoader(test_step3_all, num_workers=args.num_workers,
                        batch_size=args.batch_size, shuffle=False)
    
    
    print("Total test files in Step 1 BDD testloader : ",len(dataset_test_step1)/args.batch_size)
    print("Total test files in Step 2 city testloader : ",len(city_test_step2)/args.batch_size)
    print("Total test files in Step 3 IDD testloader : ",len(IDD_test_step3_rep)/args.batch_size)
    print("Total test files in testloader : ",len(test_loader_step3))
    
    ######################### Testing #######################################
    
    
    model.eval()
    task = args.current_task
    num_cls = args.num_classes[task]
    print('number of classes in current task: ', num_cls)
    print('Testing task: ', task)
    iouEvalVal = iouEval(num_cls, num_cls-1)

    with torch.no_grad():
        for step, (images, labels) in enumerate(test_loader_step3):
            start_time = time.time()
            inputs = images.to(device)
            targets = labels.to(device)
            
            outputs,_ = model(inputs)
            iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
    
        

    iouVal = 0
    iouVal, iou_classes = iouEvalVal.getIoU()
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print("EPOCH IoU on Test set : ", iouVal.numpy()*100, "%")
    print("IoU on test set ",iouStr)
    for i in range(len(iou_classes)):
        print(str(i)+" : "+str(iou_classes[i].numpy()*100)+" %")
    
    #savedir = f'save/{args.savedir}'
    test_path = savedir + "/test_data_best_model.txt"
    
    #print(iou_classes)
    total_test_len = len(dataset_test_step1)+len(city_test_step2)+len(IDD_test_step3_rep)
    
    with open(test_path, "w") as myfile:
        myfile.write("Total number of test files : "+str(total_test_len))
        myfile.write("\nEPOCH IoU on Test set: "+str((iouVal.numpy()*100))+" %\n")
        myfile.write("Class wise IoU\n")
        for i in range(len(iou_classes)):
            myfile.write(str(i)+" : "+str(iou_classes[i].numpy()*100)+" %\n")
    
    
    
    
    
    
   



def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    print("Saving model: ", filenameCheckpoint)
    if is_best:
        print("Saving model as best: ", filenameBest)
        torch.save(state, filenameBest)


def main(args):
    
    global current_task, device, savedir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    current_task = args.current_task
    
    print("\nNumber of classes : ",args.num_classes)
    print("Number of classes old : ",args.num_classes_old)
    print("Datasets old : ",args.dataset_old)
    print("Dataset new : ",args.dataset)
    print("Current task : ",args.current_task)
    print("nb_task : ",args.nb_tasks)
    print("Few shots for Step 2 : ",args.nshot)
    print("Batch size : ",args.batch_size,"\n")
    print("pseudo_label : ",args.pseudo_label)
    print("num_epochs : ",args.num_epochs)
    
    os.makedirs('save/',exist_ok=True)
    savedir = 'save/{}_{}'.format(args.savedir,str(args.nshot))
    print("Model saved at ",savedir)
    
    if args.eval_type=='train':
    
        start_epoch, best_acc = 1,0
    
        print("\n\nTraining starting \n")
        
        if os.path.exists(savedir):
            print("Previous saved metrics deleted")
            #shutil.rmtree(savedir)
        os.makedirs(savedir,exist_ok=True)
        
        with open(savedir + '/opts.txt', "w") as myfile:
            myfile.write(str(args))

        # Load Model
        #assert os.path.exists("models/"+ args.model + ".py"), "Error: model definition not found"
        
        if args.model == 'unet':
            model = UNet(classes=args.num_classes[-1])
        
        if args.model == 'deeplab':
            print('Selected backbone : ',args.model)
            model = DeepLab(Xception(output_stride=16), num_classes=args.num_classes[-1])
            #model_old = DeepLab(Xception(output_stride=16), num_classes=args.num_classes[-2])
            
        
        if args.cuda:
            model = torch.nn.DataParallel(model).to(device)
            #model_old = torch.nn.DataParallel(model_old).to(device)
            #model = model.cuda()
        
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        
        if args.resume:
            savedir_curr_step = 'save/{}_{}/'.format(args.savedir,str(args.nshot))
            checkpoint_file = [fs for fs in os.listdir(savedir_curr_step) if fs.startswith("model_best")][0]
            checkpoint_file_path = savedir_curr_step+checkpoint_file
            
            resume_checkpoint = torch.load(checkpoint_file_path)
            print('\nResuming model from epoch ',resume_checkpoint['epoch'],'\n')
            print('\nBest acc of model ',resume_checkpoint['best_acc'],'\n')
            start_epoch = resume_checkpoint['epoch']
            best_acc = resume_checkpoint['best_acc']
            model.load_state_dict(resume_checkpoint['state_dict'], strict=True)
        
        if args.state:
            savedir_prev_step = 'save/step{}_{}/'.format(current_task,str(args.nshot))
            checkpoint_file = [fs for fs in os.listdir(savedir_prev_step) if fs.startswith("model_best")][0]
            checkpoint_file_path = savedir_prev_step+checkpoint_file
            
            print("Loaded model from previous step from, ",checkpoint_file_path)
            checkpoint_step = torch.load(checkpoint_file_path)
            
            #model_old.load_state_dict(checkpoint_step['state_dict'], strict=True)
            #print("Old model loaded from step 1 loaded")
            
            
            ## Except last layer copy other layers weights
            load_exp_outc = {}
            
            for k, v in checkpoint_step['state_dict'].items():
                if "conv_logit" not in k:  # take all the common params as it is
                    load_exp_outc[k] = v
                        
              
            model.load_state_dict(load_exp_outc, strict=False)
            print('loaded current model with required weights\n')
            
            
                    
        total_params = sum(p.numel() for p in model.parameters())
        print("Total number of parameters : ",total_params)
        
        total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("Total number of trainable parameters : ",total_params_train)
        
        
        model = train(args, model, start_epoch, best_acc)
        # print('\nMODEL:\n', model)
        del model
        
        now_time = datetime.now()
        s1 = now_time.strftime("%d/%m/%Y, %H:%M:%S")
        # mm/dd/YY H:M:S format
        print("Training finish time :", s1)
        
        print("========== TRAINING FINISHED ===========")
    
    if args.eval_type=='test':
        print("\n\n========== Testing best model on test data ==========")
        
        if args.model == 'unet':
            model_best = UNet(classes=args.num_classes[-1])
        
        if args.model == 'deeplab':
            print('Selected backbone : ',args.model)
            model_best = DeepLab(Xception(output_stride=16), num_classes=args.num_classes[-1])

        
        if args.cuda:
            model_best = torch.nn.DataParallel(model_best).to(device)
            #model = model.cuda()
        
        best_checkpoint_file = [fs for fs in os.listdir(savedir) if fs.startswith("model_best")][0]
        
        best_checkpoint_path = savedir+"/"+best_checkpoint_file
        #best_checkpoint_path = '/hdd4/cil/codes/save/step1/model_best_BDD_erfnet_RA_parallel_100_1RAP_FT_step1.pth.tar'
        
        print("Loaded from ",best_checkpoint_path)
        best_checkpoint = torch.load(best_checkpoint_path)
        model_best.load_state_dict(best_checkpoint['state_dict'], strict=True)
        
        print('\nBest model is from epoch ',best_checkpoint['epoch'],'\n')
        
        
            #model = model.cuda()
        test(args,model_best)
        del model_best
        print("\n\n========== Testing best model finished. ==========")
    
    


if __name__ == '__main__':
    parser = ArgumentParser()
    # NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--cuda', action='store_true', default=True)
    parser.add_argument('--model', default="deeplab")  # give erfnet_bn
    # parser.add_argument('--dataset', default="cityscapes")
    parser.add_argument('--dataset', default="step3_mix")
    parser.add_argument('--dataset_old', nargs="+", help='pass list of datasets in order',
                        required=False, default=['BDD_step1','Cityscape','IDD'])

    # 27 for level 3 of IDD, 20 for BDD and city
    # do type=int, nargs='+' when you want to pass as input a list of integers
    parser.add_argument('--num-classes', type=int, nargs="+", help='pass list with number of classes',
                        required=False, default=[11,13,15])  # send [20, 20] in IL-step2 (BDD), [20, 20, 27] in IL-step3 (IDD)
    parser.add_argument('--num-classes-old', type=int, nargs="+", help='pass list with number of classes in previous task model, t-1 model',
                        required=False, default=[11,13])  # send [20] in IL-step2 (BDD), [20, 20] in IL-step3 (IDD)

    parser.add_argument('--nb_tasks', type=int, default=3)
    # 0 for IL-step1 (CS), 1 for IL-step2 (BDD), 2 for IL-step3 (IDD)
    parser.add_argument('--current_task', type=int, default=2)
    parser.add_argument('--state',default=True)
    parser.add_argument('--pseudo_label',type=int, default=5)
    parser.add_argument('--subset_size',type=int, default=100)
    parser.add_argument('--fix_thresh',type=float, default=0.6)
    # to be tuned, for now based on ADVENT
    parser.add_argument('--lambdac', type=float, default=0.1)
    parser.add_argument('--nshot', type=int, default=40)
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    # You can use this value to save model every X epochs
    parser.add_argument('--epochs-save', type=int, default=0)
    parser.add_argument('--savedir',default="step3", required=False)
    parser.add_argument('--decoder', action='store_true')
    # , default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--pretrainedEncoder')
    parser.add_argument("--alpha", default=1., type=float,
                        help="The parameter to hard-ify the soft-labels. Def is 1.")
    # recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouTrain', action='store_true', default=False)
    parser.add_argument('--iouVal', action='store_true', default=True)
    # Use this flag to load last checkpoint for training
    parser.add_argument('--resume',default=False,action='store_true')
    parser.add_argument('--model-name-suffix', default="deeplab")
    parser.add_argument('--eval-type', default='none')
    
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
    print("\n",os.getcwd())
    print("\n\nStep 3 Incremental : BDD->Cityscape->IDD Training\n")
    main(parser.parse_args())
    
    now_time = datetime.now()
    s1 = now_time.strftime("%m/%d/%Y, %H:%M:%S")
    # mm/dd/YY H:M:S format
    print("End time :", s1)
