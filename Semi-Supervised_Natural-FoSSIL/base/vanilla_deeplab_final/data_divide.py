import pickle
from fix_seed import seed_current,BDD_datadir,IDD_datadir,step3_datadir
from sklearn.model_selection import train_test_split
from dataset import VOC12, cityscapes, IDD, BDD100k
from PIL import Image
import os
import random
import time
import sys
import numpy as np
import torch
import math
import shutil
import re


import warnings
warnings.filterwarnings('ignore')

random.seed(seed_current)
np.random.seed(seed_current)
torch.manual_seed(seed_current)
torch.cuda.manual_seed(seed_current)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def read_gts(gts_folder):
    class_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 255]
    label_dict = {}
    for cl in class_labels:
        label_dict[cl]=0
    print(label_dict)
    
    print(gts_folder)
    
    sys.stdout.flush()
    
    gts_files = os.listdir(gts_folder)

    for files in gts_files:
        #print(files)
        gt_read = np.array(Image.open(gts_folder+files))
        uniq=np.unique(gt_read)
        #print(uniq)
        for i in uniq:
            #labels.add(i)
            label_dict[i]+=1
        #print(gt_read.shape)
    
    for i in label_dict.keys():
        if label_dict[i]==0:
            label_dict.pop(i)
    
    print(label_dict)


#read_gts(step1_train)
#read_gts(step1_test)

step1_train = BDD_datadir+"train/labels/"
step1_test = BDD_datadir+"test/labels/"
step2_train = IDD_datadir+"train/labels/"
step2_test = IDD_datadir+"test/labels/"
step3_train = step3_datadir+"train/labels/"
step3_test = step3_datadir+"test/labels/"

total_files_curr = int((args.nshot*10)/8)
print("Files needed to be selected ",total_files_curr)

with open(IDD_datadir+'train/train_files_idd_step2.pkl', 'rb') as file: 
    step2_all = pickle.load(file) 

train_gt_files = []
val_gt_files = []

for f_sp in step2_all:
    curr_files = step2_all[f_sp]
    random.shuffle(curr_files)
    #print(curr_files)
    curr_train, curr_val = train_test_split(curr_files[:total_files_curr], test_size=0.2, random_state=seed_current)
    #print(len(curr_train),len(curr_val))
    train_gt_files.extend(curr_train)
    val_gt_files.extend(curr_val)

print("Few Shot per class in Step 2 ",args.nshot)
print("Total files in Step 2 IDD train : ",len(train_gt_files))
print("Total files in Step 2 IDD val : ",len(val_gt_files))

with open('data_used/IDD_train_step2.pkl', 'wb') as file: 
    pickle.dump(train_gt_files, file) 
    
with open('data_used/IDD_val_step2.pkl', 'wb') as file: 
    pickle.dump(val_gt_files, file)










