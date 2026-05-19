from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
    CenterSpatialCropd,
    Resized,
    SpatialPadd,
    apply_transform,
)

import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
from copy import copy, deepcopy
import cc3d
import argparse
import os
import h5py
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
import sys
torch.multiprocessing.set_sharing_strategy('file_system')
from datetime import datetime as date_and_time
from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor

from utils.utils import get_key


## full list
# TRANSFER_LIST = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10_03', '10_06', '10_07', '10_08', '10_09', '10_10', '12', '13', '14']


TEMPLATE={
    '01': [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    '02': [1,0,3,4,5,6,7,0,0,0,11,0,0,14],
    '03': [6],
    '04': [6,27],       # post process
    '05': [2,26,32],       # post process
    '07': [6,1,3,2,7,4,5,11,14,18,19,12,20,21,23,24],
    '08': [6, 2, 1, 11],
    '09': [1,2,3,4,5,6,7,8,9,11,12,13,14,21,22],
    '12': [6,21,16,2],  
    '13': [6,2,1,11,8,9,7,4,5,12,13,25], 
    '14': [11,11,28,28,28],     # Felix data, post process
    '10_03': [6, 27],   # post process
    '10_06': [30],
    '10_07': [11, 28],  # post process
    '10_08': [15, 29],  # post process
    '10_09': [1],
    '10_10': [31],
    'step0_test': [1,2,3,4,5,22,7,8,14,9],
    'step1_test': [1,2,3,4,5,22,7,8,14,9,11,33,10,13,12,6,15,29],
    'step1_train': [255,255,255,255,255,255,255,255,255,255, 255,33,255,255,255,255,255,255],
    'step2_test': [1,2,3,4,5,22,7,8,14,9,11,33,10,13,12,6,15,29,34,35,36,17,17,21],
    'step2_train': [255,2,3,255,255,255,7,255,255,255,255,255,255,255,255,255,255,255,34,35,36,255,255,255],
    'step3_test': [1,2,3,4,5,22,7,8,14,9,11,33,10,13,12,6,15,29,34,35,36,17,17,21,31,30,23,24],
    'step4_test': [1,2,3,4,5,22,7,8,14,9,11,33,10,13,12,6,15,29,34,35,36,17,17,21,31,30,23,24,37,38,39,40,41,42,43],
    'step4_train': [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,37,38,39,40,41,42,43]
}






POST_TUMOR_DICT = {
    '04': [(2,27)],
    '05': [(2,26), (3,32)],
    '10_03': [(2,27)], 
    '10_07': [(2,28)]
}

def rl_split(input_data, organ_index, right_index, left_index, name):
    '''
    input_data: 3-d tensor [w,h,d], after transform 'Orientationd(keys=["label"], axcodes="RAS")'
    oragn_index: the organ index of interest
    right_index and left_index: the corresponding index in template
    return [1, w, h, d]
    '''
    RIGHT_ORGAN = right_index
    LEFT_ORGAN = left_index
    label_raw = input_data.copy()
    label_in = np.zeros(label_raw.shape)
    label_in[label_raw == organ_index] = 1
    
    label_out = cc3d.connected_components(label_in, connectivity=26)
    # print('label_out', organ_index, np.unique(label_out), np.unique(label_in), label_out.shape, np.sum(label_raw == organ_index))
    # assert len(np.unique(label_out)) == 3, f'more than 2 component in this ct for {name} with {np.unique(label_out)} component'
    if len(np.unique(label_out)) > 3:
        count_sum = 0
        values, counts = np.unique(label_out, return_counts=True)
        num_list_sorted = sorted(values, key=lambda x: counts[x])[::-1]
        for i in num_list_sorted[3:]:
            label_out[label_out==i] = 0
            count_sum += counts[i]
        label_new = np.zeros(label_out.shape)
        for tgt, src in enumerate(num_list_sorted[:3]):
            label_new[label_out==src] = tgt
        label_out = label_new
        print(f'In {name}. Delete {len(num_list_sorted[3:])} small regions with {count_sum} voxels')
    a1,b1,c1 = np.where(label_out==1)
    a2,b2,c2 = np.where(label_out==2)
    
    label_new = np.zeros(label_out.shape)
    if np.mean(a1) < np.mean(a2):
        label_new[label_out==1] = LEFT_ORGAN
        label_new[label_out==2] = RIGHT_ORGAN
    else:
        label_new[label_out==1] = RIGHT_ORGAN
        label_new[label_out==2] = LEFT_ORGAN
    
    return label_new[None]

class ToTemplatelabel(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, lbl: NdarrayOrTensor, totemplate: List, tumor=False, tumor_list=None) -> NdarrayOrTensor:
        new_lbl = np.zeros(lbl.shape)
        for src, tgt in enumerate(totemplate):
            new_lbl[lbl == (src+1)] = tgt
        if tumor:
            for src, item in tumor_list:
                new_lbl[new_lbl == item] = totemplate[0]
        return new_lbl

class ToTemplatelabeld(MapTransform):
    '''
    Comment: spleen to 1
    '''
    backend = ToTemplatelabel.backend
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.totemplate = ToTemplatelabel()
        

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        dataset_index = d['template_key']
        TUMOR = False
        tumor_list = None
        if dataset_index == 1 or dataset_index == 2:
            template_key = d['name'][0:2]
            pass
        elif dataset_index == 10:
            template_key = d['name'][0:2] + '_' + d['name'][17:19]
        else:
            template_key = d['name'][0:2]
        if template_key in ['04', '05', '10_03', '10_07', '14']:
            TUMOR = True
            tumor_list = POST_TUMOR_DICT[template_key]
        d['label'] = self.totemplate(d['label'], TEMPLATE[dataset_index], tumor=TUMOR, tumor_list=tumor_list)
        return d

class RL_Split(Transform):
    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __call__(self, lbl: NdarrayOrTensor, organ_list: List, name) -> NdarrayOrTensor:
        lbl_new = lbl.copy()
        for organ in organ_list:
            organ_index = organ
            right_index = organ
            left_index = organ + 1
            lbl_post = rl_split(lbl_new[0], organ_index, right_index, left_index, name)
            lbl_new[lbl_post == left_index] = left_index
        return lbl_new

class RL_Splitd(MapTransform):
    backend = ToTemplatelabel.backend
    def __init__(self, keys: KeysCollection, allow_missing_keys: bool = False) -> None:
        super().__init__(keys, allow_missing_keys)
        self.spliter = RL_Split()

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        dataset_index = int(d['name'][0:2])
        if dataset_index in [5,8,13]:
            d['label'] = self.spliter(d['label'], [2], d['name'])
        elif dataset_index == 7:
            d['label'] = self.spliter(d['label'], [12], d['name'])
        elif dataset_index == 12:
            d['label'] = self.spliter(d['label'], [2, 16], d['name'])
        else:
            pass
        return d

def generate_label(input_lbl, num_classes, name, TEMPLATE, raw_lbl, template_key):
    """
    Convert class index tensor to one hot encoding tensor with -1 (ignored).
    Args:
         input: A tensor of shape [bs, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [bs, num_classes, *]
    Comment: spleen to 0
    """
    shape = np.array(input_lbl.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    input_lbl = input_lbl.long()

    ## generate binary cross entropy label and assign -1 to ignored organ
    B = result.shape[0]
    for b in range(B):
        dataset_index = template_key
        if dataset_index == 10:
            template_key = name[b][0:2] + '_' + name[b][17:19]
        else:
            template_key = name[b][0:2]
        
        # for organ split case
        if dataset_index == 5:
            organ_list = [2,3,26,32]
        elif dataset_index == 7:
            organ_list = [6,1,3,2,7,4,5,11,14,18,19,12,13,20,21,23,24]
        elif dataset_index == 8:
            organ_list = [6, 2, 3, 1, 11]
        elif dataset_index == 12:
            organ_list = [6,21,16,17,2,3]
        elif dataset_index == 13:
            organ_list = [6,2,3,1,11,8,9,7,4,5,12,13,25]
        else:
            organ_list = TEMPLATE[dataset_index]
        
        # -1 for organ not labeled
        for i in range(num_classes):
            if (i+1) not in organ_list:
                result[b, i] = -1
            else:
                result[b, i] = (input_lbl[b][0] ==  (i+1))
        
        # for tumor case
        if template_key in ['04', '05', '10_03', '10_07']:
            tumor_list = POST_TUMOR_DICT[template_key]
            for src, item in tumor_list:
                result[b, item - 1] = (raw_lbl[b][0] == src)

        if template_key in ['14']:
            tumor_lbl = torch.zeros(raw_lbl.shape)
            tumor_lbl[raw_lbl == 3] = 1
            tumor_lbl[raw_lbl == 4] = 1
            tumor_lbl[raw_lbl == 5] = 1
            result[b, organ_list[-1] - 1] = tumor_lbl[b][0]
    return result


now_time = date_and_time.now()
s1 = now_time.strftime("%d/%m/%Y, %H:%M:%S")
# mm/dd/YY H:M:S format
print("\n\nStart time :", s1)

NUM_WORKER = 4
NUM_CLASS = 43

set2_path = "med_3D/dataset/set2/set2_full/merged/"
data_path = set2_path+"dataset/"
image_path = data_path+"images/"
ann_path = data_path+"annotations/"


splitwise_path = set2_path+"split_wise/splitwise.pkl"
with open(splitwise_path, 'rb') as file: 
    splitwise = pickle.load(file) 

'''
### Specify data to process   
process_data = ['step0','test']
template_key = process_data[0]+"_"+process_data[1]
  
req_files = splitwise[process_data[0]][process_data[1]]
post_label_path = "set2_data/"+process_data[1]+"/"
'''

label_process = Compose(
    [
        LoadImaged(keys=["image", "label", "label_raw"]),
        AddChanneld(keys=["image", "label", "label_raw"]),
        Orientationd(keys=["image", "label", "label_raw"], axcodes="RAS"),
        ToTemplatelabeld(keys=['label']),
        #RL_Splitd(keys=['label']),
        Spacingd(
            keys=["image", "label", "label_raw"], 
            pixdim=(1.5, 1.5, 1.5), 
            mode=("bilinear", "nearest", "nearest"),), # process h5 to here
    ]
)

to_process = ['step4_train','step4_test']

for pro_data in to_process:
    print(pro_data)
    ### Specify data to process   
    process_data = []
    process_data.append(pro_data.split("_")[0])
    process_data.append(pro_data.split("_")[1])
    template_key = process_data[0]+"_"+process_data[1]
    
    req_files=[]
    if process_data[1]=='test':
        if process_data[0]=='step3':
            req_files.extend(splitwise['step{}'.format(process_data[0][4])][process_data[1]])
        else:
            for itest in range(int(process_data[0][4])+1):
                req_files.extend(splitwise['step{}'.format(itest)][process_data[1]])
            '''
            hepv_files=[]
            for files in req_files:
                if "hepaticvessel" in files:
                    hepv_files.append(files)
            req_files=hepv_files
            '''
    else:
        req_files.extend(splitwise['step{}'.format(process_data[0][4])][process_data[1]])
        
        '''
        alread_proc = ['amos_0034.nii.gz', 'amos_0038.nii.gz', 'amos_0113.nii.gz', 'amos_0195.nii.gz', 'amos_0216.nii.gz', 'amos_0276.nii.gz', 'amos_0321.nii.gz', 'amos_0332.nii.gz', 'amos_0334.nii.gz', 's0013_fused.nii.gz', 's0014_fused.nii.gz', 's0019_fused.nii.gz', 's0054_fused.nii.gz', 's0068_fused.nii.gz', 's0076_fused.nii.gz', 's0082_fused.nii.gz', 's0086_fused.nii.gz', 's0088_fused.nii.gz', 's0090_fused.nii.gz', 's0120_fused.nii.gz', 's0123_fused.nii.gz', 's0139_fused.nii.gz', 's0188_fused.nii.gz', 's0189_fused.nii.gz', 's0221_fused.nii.gz', 's0222_fused.nii.gz', 's0243_fused.nii.gz', 's0255_fused.nii.gz', 's0262_fused.nii.gz', 's0275_fused.nii.gz', 's0283_fused.nii.gz', 's0290_fused.nii.gz', 's0298_fused.nii.gz', 's0307_fused.nii.gz', 's0315_fused.nii.gz', 's0319_fused.nii.gz', 's0334_fused.nii.gz', 's0344_fused.nii.gz', 's0354_fused.nii.gz', 's0366_fused.nii.gz', 's0390_fused.nii.gz', 's0406_fused.nii.gz', 's0416_fused.nii.gz', 's0433_fused.nii.gz', 's0436_fused.nii.gz', 's0450_fused.nii.gz', 's0458_fused.nii.gz', 's0459_fused.nii.gz', 's0485_fused.nii.gz', 's0509_fused.nii.gz', 's0513_fused.nii.gz', 's0589_fused.nii.gz', 's0591_fused.nii.gz', 's0593_fused.nii.gz', 's0606_fused.nii.gz', 's0607_fused.nii.gz', 's0616_fused.nii.gz', 's0664_fused.nii.gz', 's0665_fused.nii.gz', 's0670_fused.nii.gz', 's0732_fused.nii.gz', 's0785_fused.nii.gz', 's0842_fused.nii.gz', 's0863_fused.nii.gz']
        not_to_use_data = ['s0864_fused.nii.gz']
        req_files = list(set(req_files)-set(alread_proc))
        req_files = list(set(req_files)-set(not_to_use_data))
        '''
    post_label_path = "set2_data/"+process_data[1]+"/"
    req_files.sort()
    
    print("Processesing "+process_data[0]+" "+process_data[1]+" data ")
    print(req_files)
    print(len(req_files))
    train_img = []
    train_lbl = []
    train_name = []
    
    post_temp = TEMPLATE[template_key]
    for i in range(len(post_temp)):
        print(i+1,post_temp[i])
    
    for files in req_files:
        #key = get_key(line.strip().split()[0])
        train_img.append(image_path + process_data[1] + "/" + files)
        train_lbl.append(ann_path + process_data[1] + "/" + files)
        train_name.append(files)
        
    data_dicts_train = [{'image': image, 'label': label, 'label_raw': label, 'name': name,
                'template_key':template_key}
                for image, label, name in zip(train_img, train_lbl, train_name)]
    print('train len {}'.format(len(data_dicts_train)))
    sys.stdout.flush()

    train_dataset = Dataset(data=data_dicts_train, transform=label_process)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKER, 
                                collate_fn=list_data_collate)

    for index, batch in enumerate(train_loader):
        x, y, y_raw, name = batch["image"], batch["label"], batch['label_raw'], batch['name']
        print(name, x.shape)
        print(len(np.unique(x))>5)
        print("old labels : ",np.unique(y),y.shape)
        
        
        y = generate_label(y, NUM_CLASS, name, TEMPLATE, y_raw, template_key)
        
        #name = batch['name'][0].replace('label', 'post_label')
        #print(name)
        post_dir = post_label_path + name[0]
        
        store_y = y.numpy().astype(np.uint8)
        
        print("new labels : ",np.unique(store_y),store_y.shape)
        sys.stdout.flush()
        
        if not os.path.exists(post_dir):
            os.makedirs(post_dir)
            
        with h5py.File(post_label_path + name[0] + '.h5', 'w') as f:
            f.create_dataset('post_label', data=store_y, compression='gzip', compression_opts=9)
            f.close()
    

# current date and time
now_time = date_and_time.now()
s1 = now_time.strftime("%d/%m/%Y, %H:%M:%S")
# mm/dd/YY H:M:S format
print("\n\nEnd time :", s1)