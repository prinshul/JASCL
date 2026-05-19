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

from dataset import VOC12, cityscapes, IDD, BDD100k
from transform import Relabel, ToLabel, Colorize
import itertools
#import config_task
import pickle
import importlib
from iouEval import iouEval, getColorEntry
import torch.nn as nn
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
    def __init__(self, augment=True, height=512, width=1024):
        self.augment = augment
        self.height = height
        self.width = width
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
        #print('relabeling 255 as: ', NUM_CLASSES-1)
        target = Relabel(255, NUM_CLASSES - 1)(target)

        return input, target


class CrossEntropyLoss2d(torch.nn.Module):

    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


def train(args, model, start_epoch, best_acc):
    global NUM_CLASSES
    NUM_CLASSES = args.num_classes[args.current_task]
    print('NUM_CLASSES: ', NUM_CLASSES)

    

    tf_dir = 'runs_{}_{}_{}_{}{}_step{}'.format(
        args.dataset, args.model, args.num_epochs, args.batch_size, args.model_name_suffix, len(args.num_classes))
    writer = SummaryWriter('Adaptations/' + tf_dir)

    data_name = args.dataset

    

    co_transform = MyCoTransform(augment=True,height=args.height, width=args.width)  # 1024)
    co_transform_val = MyCoTransform(augment=False,height=args.height, width=args.width)  # 1024)

    
    with open(shots_datadir+'gt_train_step1.pkl', 'rb') as file: 
        gt_train=pickle.load(file)
        
    with open(shots_datadir+'gt_val_step1.pkl', 'rb') as file: 
        gt_val=pickle.load(file)
    
    print("Total files in train : ",len(gt_train))
    print("Total files in val : ",len(gt_val))
        
    if data_name == 'BDD':
        print('taking BDD')
        dataset_train = BDD100k(BDD_datadir,gt_train, co_transform,'train')
        dataset_val = BDD100k(BDD_datadir,gt_val,co_transform_val,'train')
        
    sys.stdout.flush()
    
    weight_BDD = torch.tensor([0.1000, 0.4896, 0.1000, 1.9219, 0.9045, 5.4740, 2.8373, 4.2810, 0.1196,
        1.0000, 0.0000])
    weight_BDD[NUM_CLASSES - 1] = 0


    #weight_BDD = torch.tensor([3.6525147,8.799815,4.781908,10.034828,9.556787,
    #             8.315293,8.163474,9.246903,6.0067043,9.606205,0])
        
    loader = DataLoader(dataset_train, num_workers=args.num_workers,
                        batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers,
                            batch_size=args.batch_size, shuffle=False)

    if args.cuda:
        class_weights = weight_BDD.cuda()
        class_weights = torch.clamp(class_weights, 0.1, 10.0)
        
    print("Class weights ",class_weights)
    

    print('\n\n\n')
    for name, m in model.named_parameters():
        print(name, m.requires_grad)
    
    
    #savedir = f'save/{args.savedir}'

    automated_log_path = savedir + "/automated_log.txt"
    modeltxtpath = savedir + "/model.txt"

    if (not os.path.exists(automated_log_path)):  # dont add first line if it exists
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))
    
    
    ############ Model hyperparameters ####################################
    
    #criterion_city = CrossEntropyLoss2d(weight_BDD)
    '''
    criterion = CrossEntropyLoss2d(weight=class_weights)
    
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),
                     eps=1e-08, weight_decay=1e-4)  # scheduler 2

    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    def lambda1(epoch): return pow((1-((epoch-1)/args.num_epochs)), 0.9)  # scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)  # scheduler 2
    
    '''
    
    # Deep lab hyperparameters
    #criterion = CrossEntropyLoss2d(weight=class_weights)
    criterion = nn.CrossEntropyLoss(ignore_index=NUM_CLASSES - 1, weight=class_weights)
    print(type(criterion))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=4e-5)

    lr_update = lambda epoch: (1 - epoch / args.num_epochs) ** 0.9
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_update)
    
    
    
    
    
    #start_epoch = 0
    #best_acc = 0
    print("\nstart_epoch = ",start_epoch,"\nbest_acc = ",best_acc)
    

    for epoch in range(start_epoch, args.num_epochs+1):
        print("-----TRAINING - EPOCH---", epoch, "-----")

        scheduler.step(epoch)  # scheduler 2

        epoch_loss = []
        time_train = []

        doIouTrain = args.iouTrain

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels) in enumerate(loader):
            if epoch == start_epoch  and step == 1:
                print('image size new: ', images.size())
                print('labels size new: ', labels.size())
                print('labels are: ', np.unique(labels.numpy()))
                # writer.add_graph(model(), images.cuda(), True) #not working (Segmentation fault (core dumped))
            
            sys.stdout.flush()
            
            start_time = time.time()
            if args.cuda:
                inputs = images.cuda()
                targets = labels.cuda()

            outputs = model(inputs, current_task)

            optimizer.zero_grad()
            loss = criterion(outputs, targets[:, 0])
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss.item())
            time_train.append(time.time() - start_time)

            if (doIouTrain):
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})',
                      "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        print('epoch took: ', sum(time_train))

        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print("EPOCH IoU on TRAIN set: ", iouStr, "%")
    
        print("----- VALIDATING - EPOCH", epoch, "-----")
        #if epoch % 10 == 0:
        average_loss_val, val_acc = eval(
            model, loader_val, criterion, current_task, args.num_classes, epoch)
        # every 10 epochs, check validation accuracy on other datasets:
        # if epoch % 10 == 0:
        #     average_loss_val_bdd, val_acc_bdd = eval(
        #         model, loader_val_cs, criterion_city, 0, args.num_classes, epoch)
        #     print('cityscapes loss and acc: ', average_loss_val_bdd, val_acc_bdd)
        # logging tensorboard plots - epoch wise loss and accuracy. Not calculating iouTrain as that will slow down training
        # info = {'train_loss': average_epoch_loss_train, 'val_loss_{}'.format(
        #     data_name): average_loss_val, 'val_acc_{}'.format(data_name): val_acc}

        info = {'train_loss': average_epoch_loss_train, 'val_loss_{}'.format(
            data_name): average_loss_val, 'val_acc_{}'.format(data_name): val_acc}

        for tag, value in info.items():
            writer.add_scalar(tag, value, epoch)

        # remember best valIoU and save checkpoint
        if val_acc == 0:
            current_acc = -average_epoch_loss_val
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
        
        sys.stdout.flush()
        
    return(model)


def eval(model, dataset_loader, criterion, task, num_classes, epoch):
    # Validate on 500 val images after each epoch of training
    
    model.eval()
    epoch_loss_val = []
    time_val = []
    num_cls = num_classes[task]
    print('number of classes in current task: ', num_cls)
    print('validating task: ', task)
    iouEvalVal = iouEval(num_cls, num_cls-1)

    with torch.no_grad():
        for step, (images, labels) in enumerate(dataset_loader):
            start_time = time.time()
            inputs = images.cuda()
            targets = labels.cuda()

            outputs = model(inputs, current_task)
            if step == 1:
                print('------------------', outputs.size(), targets.size())

            loss = criterion(outputs, targets[:, 0])
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
    
    test_gt_files = os.listdir(BDD_datadir+'test/labels/')
    
    print("Total number of test files ",len(test_gt_files))
    
    co_transform_test = MyCoTransform(augment=False, height=args.height, width=args.width)  # 1024)
    
    dataset_test = BDD100k(BDD_datadir,test_gt_files,co_transform_test,'test')
        
    test_loader = DataLoader(dataset_test, num_workers=args.num_workers,
                        batch_size=args.batch_size, shuffle=False)
    
    
    model.eval()
    task = args.current_task
    num_cls = args.num_classes[task]
    print('number of classes in current task: ', num_cls)
    print('Testing task: ', task)
    iouEvalVal = iouEval(num_cls, num_cls-1)

    with torch.no_grad():
        for step, (images, labels) in enumerate(test_loader):
            start_time = time.time()
            inputs = images.cuda()
            targets = labels.cuda()
            
            outputs = model(inputs, current_task)
            iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

    iouVal = 0
    iouVal, iou_classes = iouEvalVal.getIoU()
    iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
    print("EPOCH IoU on Test set : ", iouVal.numpy()*100, "%")
    print("IoU on test set ",iouStr)
    for i in range(len(iou_classes)):
        print(str(i)+" : "+str(iou_classes[i].numpy()*100)+" %\n")
        
    #savedir = f'save/{args.savedir}'
    test_path = savedir + "/test_data_best_model.txt"
    
    #print(iou_classes)

    with open(test_path, "w") as myfile:
        myfile.write("Total number of test files : "+str(len(test_gt_files)))
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
    
    global current_task, savedir
    current_task = args.current_task
    
    print("\nNumber of classes : ",args.num_classes)
    print("Datasets : ",args.dataset)
    print("Batch size : ",args.batch_size,"\n")
    print("\nNumber of classes : ",args.num_classes)
    print("Current task : ",args.current_task)
    print("nb_task : ",args.nb_tasks)
    print("epochs : ",args.num_epochs)
    
    os.makedirs('save/',exist_ok=True)
    savedir = f'save/{args.savedir}'
    print("Model saved at ",savedir)
    
    if os.path.exists(savedir):
        print("Previous saved metrics deleted")
        #shutil.rmtree(savedir)
    os.makedirs(savedir,exist_ok=True)
    
    if args.eval_type=='train':
        
        start_epoch, best_acc = 1,0
    
        with open(savedir + '/opts.txt', "w") as myfile:
            myfile.write(str(args))

        # Load Model
        #assert os.path.exists("models/"+ args.model + ".py"), "Error: model definition not found"
        
        if args.model == 'unet':
            print('Selected backbone : ',args.model)
            model = UNet(classes=args.num_classes[-1])
            
        if args.model == 'deeplab':
            print('Selected backbone : ',args.model)
            model = DeepLab(Xception(output_stride=16), current_task, num_classes=args.num_classes[-1])
        
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
        
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()
            #model = model.cuda()

        
        
        if args.resume:
            savedir_curr_step = 'save/step{}/'.format(args.nb_tasks)
            checkpoint_file = [fs for fs in os.listdir(savedir_curr_step) if fs.startswith("model_best")][0]
            checkpoint_file_path = savedir_curr_step+checkpoint_file
            
            resume_checkpoint = torch.load(checkpoint_file_path)
            print('\nResuming model from epoch ',resume_checkpoint['epoch'],'\n')
            print('\nBest acc of model ',resume_checkpoint['best_acc'],'\n')
            start_epoch = resume_checkpoint['epoch']
            best_acc = resume_checkpoint['best_acc']
            model.load_state_dict(resume_checkpoint['state_dict'], strict=True)
        
        
        print('loaded\n')
        
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
            print('Selected backbone : ',args.model)
            model_best = UNet(classes=args.num_classes[-1])
            
        if args.model == 'deeplab':
            print('Selected backbone : ',args.model)
            model_best = DeepLab(Xception(output_stride=16), current_task,  num_classes=args.num_classes[-1])


        if args.cuda:
            model_best = torch.nn.DataParallel(model_best).cuda()
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
    parser.add_argument('--dataset', default="BDD")

    # 27 for level 3 of IDD, 20 for BDD and city
    # do type=int, nargs='+' when you want to pass as input a list of integers
    parser.add_argument('--num-classes', type=int, nargs="+", help='pass list with number of classes',
                        required=False, default=[11])  # send [20, 20] in IL-step1, [20, 20, 27] in IL-step2
    parser.add_argument('--nb_tasks', type=int, default=1)  # 2 for IL-step1, 3 for IL-step2
    # 0 for IL-step1 (CS), 1 for IL-step2 (BDD), 2 for IL-step3 (IDD)
    parser.add_argument('--current_task', type=int, default=0)
    parser.add_argument('--state',default=False)

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    # You can use this value to save model every X epochs
    parser.add_argument('--epochs-save', type=int, default=0)
    parser.add_argument('--savedir',default="step1",required=False)
    parser.add_argument('--decoder', action='store_true')
    # , default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--pretrainedEncoder')

    # recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouTrain', action='store_true', default=False)
    parser.add_argument('--iouVal', action='store_true', default=True)
    # Use this flag to load last checkpoint for training
    parser.add_argument('--resume',default=False)
    parser.add_argument('--model-name-suffix', default="deeplab")
    parser.add_argument('--eval-type', default='test')
    
    # current date and time
    now_time = datetime.now()
    s1 = now_time.strftime("%d/%m/%Y, %H:%M:%S")
    # mm/dd/YY H:M:S format
    print("Start time :", s1)
    
    print("\nGPU Details : ")
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))
    
    torch.cuda.empty_cache()
    print("\n\nStep 1 : BDD Training\n\n")
    main(parser.parse_args())
    torch.cuda.empty_cache()
    
    now_time = datetime.now()
    s1 = now_time.strftime("%d/%m/%Y, %H:%M:%S")
    # mm/dd/YY H:M:S format
    print("End time :", s1)
