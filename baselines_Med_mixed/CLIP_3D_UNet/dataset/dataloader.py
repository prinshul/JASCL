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
    RandZoomd,
    RandCropByLabelClassesd,
)

import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
import h5py
import glob
import os
import random
import numpy as np
import torch
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union

sys.path.append("..") 
from utils.utils import get_key

from torch.utils.data import Subset

from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import Transform, MapTransform
from monai.utils.enums import TransformBackends
from monai.config.type_definitions import NdarrayOrTensor
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
DEFAULT_POST_FIX = PostFix.meta()

class UniformDataset(Dataset):
    def __init__(self, data, transform, datasetkey):
        super().__init__(data=data, transform=transform)
        self.dataset_split(data, datasetkey)
        self.datasetkey = datasetkey
    
    def dataset_split(self, data, datasetkey):
        self.data_dic = {}
        for key in datasetkey:
            self.data_dic[key] = []
        for img in data:
            key = get_key(img['name'])
            self.data_dic[key].append(img)
        
        self.datasetnum = []
        for key, item in self.data_dic.items():
            assert len(item) != 0, f'the dataset {key} has no data'
            self.datasetnum.append(len(item))
        self.datasetlen = len(datasetkey)
    
    def _transform(self, set_key, data_index):
        data_i = self.data_dic[set_key][data_index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i
    
    def __getitem__(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        set_key = self.datasetkey[set_index]
        # data_index = int(index / self.__len__() * self.datasetnum[set_index])
        data_index = np.random.randint(self.datasetnum[set_index], size=1)[0]
        return self._transform(set_key, data_index)


class UniformCacheDataset(CacheDataset):
    def __init__(self, data, transform, cache_rate, datasetkey):
        super().__init__(data=data, transform=transform, cache_rate=cache_rate)
        self.datasetkey = datasetkey
        self.data_statis()
    
    def data_statis(self):
        data_num_dic = {}
        for key in self.datasetkey:
            data_num_dic[key] = 0

        for img in self.data:
            key = get_key(img['name'])
            data_num_dic[key] += 1

        self.data_num = []
        for key, item in data_num_dic.items():
            assert item != 0, f'the dataset {key} has no data'
            self.data_num.append(item)
        
        self.datasetlen = len(self.datasetkey)
    
    def index_uniform(self, index):
        ## the index generated outside is only used to select the dataset
        ## the corresponding data in each dataset is selelcted by the np.random.randint function
        set_index = index % self.datasetlen
        data_index = np.random.randint(self.data_num[set_index], size=1)[0]
        post_index = int(sum(self.data_num[:set_index]) + data_index)
        return post_index

    def __getitem__(self, index):
        post_index = self.index_uniform(index)
        # print(post_index, self.__len__())
        return self._transform(post_index)

class LoadImageh5d(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        reader: Optional[Union[ImageReader, str]] = None,
        dtype: DtypeLike = np.float32,
        meta_keys: Optional[KeysCollection] = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        overwriting: bool = False,
        image_only: bool = False,
        ensure_channel_first: bool = False,
        simple_keys: bool = False,
        allow_missing_keys: bool = False,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting


    def register(self, reader: ImageReader):
        self._loader.register(reader)


    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]
        post_label_pth = d['post_label']
        with h5py.File(post_label_pth, 'r') as hf:
            data = hf['post_label'][()]
        d['post_label'] = data[0]
        return d

class RandZoomd_select(RandZoomd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = curr_temp_key #get_key(name)
        if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']):
            return d
        d = super().__call__(d)
        return d


class RandCropByPosNegLabeld_select(RandCropByPosNegLabeld):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = curr_temp_key #get_key(name)
        if key in ['10_03', '10_07', '10_08', '04']:
            return d
        d = super().__call__(d)
        return d

class RandCropByLabelClassesd_select(RandCropByLabelClassesd):
    def __call__(self, data):
        d = dict(data)
        name = d['name']
        key = curr_temp_key #get_key(name)
        if key not in ['10_03', '10_07', '10_08', '04']:
            return d
        d = super().__call__(d)
        return d

class Compose_Select(Compose):
    def __call__(self, input_):
        name = input_['name']
        key = curr_temp_key #get_key(name)
        for index, _transform in enumerate(self.transforms):
            # for RandCropByPosNegLabeld and RandCropByLabelClassesd case
            if (key in ['10_03', '10_07', '10_08', '04']) and (index == 8):
                continue
            elif (key not in ['10_03', '10_07', '10_08', '04']) and (index == 9):
                continue
            # for RandZoomd case
            if (key not in ['10_03', '10_06', '10_07', '10_08', '10_09', '10_10']) and (index == 7):
                continue
            input_ = apply_transform(_transform, input_, self.map_items, self.unpack_items, self.log_stats)
        return input_

def get_loader(args):
    train_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]), #0
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            SpatialPadd(keys=["image", "label", "post_label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z), mode='constant'),
            RandZoomd_select(keys=["image", "label", "post_label"], prob=0.3, min_zoom=1.3, max_zoom=1.5, mode=['area', 'nearest', 'nearest']), # 7
            RandCropByPosNegLabeld_select(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                pos=2,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 8
            RandCropByLabelClassesd_select(
                keys=["image", "label", "post_label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z), #192, 192, 64
                ratios=[1, 1, 5],
                num_classes=3,
                num_samples=args.num_samples,
                image_key="image",
                image_threshold=0,
            ), # 9
            RandRotate90d(
                keys=["image", "label", "post_label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImageh5d(keys=["image", "label"]),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image", "label"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear", "nearest"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label", "post_label"], source_key="image"),
            ToTensord(keys=["image", "label", "post_label"]),
        ]
    )
    
    data_path = args.data_root_path+"dataset/"
    image_path = data_path+"images/"
    ann_path = data_path+"annotations/"
    post_label_path = "med_3D/baselines/large_model/CLIP_3D/set2_data/"
    
    #print("args.data_root_path : ",args.data_root_path)
    splitwise_path = args.data_root_path+"split_wise/splitwise.pkl"
    with open(splitwise_path, 'rb') as file: 
        splitwise = pickle.load(file)
        
    global curr_temp_key 
    curr_temp_key = args.data_root_path
    
    curr_type=None
    if 'test' in args.data_txt_path:
        curr_type = 'test/'
        process_data=int(args.data_txt_path[4])
        req_files=[]
        for itest in range(process_data+1):
            req_files.extend(splitwise['step{}'.format(str(itest))]['test'])
        
        
        '''
        hepv_files=[]
        for files in req_files:
            if "hepaticvessel" in files:
                hepv_files.append(files)
        hepv_files = list(set(hepv_files)-set(['hepaticvessel_275.nii.gz','hepaticvessel_279.nii.gz']))
        '''
        #req_files.extend(splitwise['step{}'.format(str(process_data))]['test'])
        
        exclude_files = ['hepaticvessel_275.nii.gz','hepaticvessel_279.nii.gz',
        's0773_fused.nii.gz','lung_062.nii.gz','amos_0009.nii.gz',
        'amos_0017.nii.gz']
        #req_files=list(set(req_files)-set(exclude_files))
        
        req_files.sort()
        
    elif 'train' in args.data_txt_path:  
        curr_type = 'train/'
        process_data=int(args.data_txt_path[4])
        req_files=[]
        curr_req_data = []
        step_data = splitwise['step{}'.format(str(process_data))]['train']
        
        if args.data_txt_path=='step1_train':
            for st_dt in step_data:
                if 'bcv' in st_dt:
                    curr_req_data.append(st_dt)
            random.seed(args.seed)
            req_files=random.sample(curr_req_data, 5)
        #req_files.extend()
        if args.data_txt_path=='step2_train':
            to_exclude = ['amos_0034.nii.gz', 'amos_0038.nii.gz', 'amos_0113.nii.gz', 'amos_0195.nii.gz', 'amos_0216.nii.gz', 'amos_0276.nii.gz', 'amos_0321.nii.gz', 'amos_0332.nii.gz', 'amos_0334.nii.gz']
            step_data = list(set(step_data)-set(to_exclude))
            
            not_to_use_data = ['s0864_fused.nii.gz']
            step_data = list(set(step_data)-set(not_to_use_data))
            
            step2_old = ['s0068_fused.nii.gz','s0088_fused.nii.gz','s0123_fused.nii.gz',
            's0188_fused.nii.gz','s0221_fused.nii.gz','s0222_fused.nii.gz',
            's0262_fused.nii.gz','s0283_fused.nii.gz','s0290_fused.nii.gz',
            's0354_fused.nii.gz','s0433_fused.nii.gz','s0450_fused.nii.gz',
            's0459_fused.nii.gz','s0485_fused.nii.gz','s0513_fused.nii.gz',
            's0606_fused.nii.gz','s0665_fused.nii.gz','s0785_fused.nii.gz',
            's0895_fused.nii.gz','s0946_fused.nii.gz','s0978_fused.nii.gz',
            's1066_fused.nii.gz','s1077_fused.nii.gz','s1162_fused.nii.gz',
            's1201_fused.nii.gz','s1236_fused.nii.gz','s1270_fused.nii.gz',
            's1292_fused.nii.gz','s1337_fused.nii.gz','s1401_fused.nii.gz']
            
            step2_new = list(set(step_data)-set(step2_old))
            
            random.seed(args.seed)        
            ## 5 samples for each class - 15 each
            req_files.extend(random.sample(step2_old, 15))
            req_files.extend(random.sample(step2_new, 15))
            
        if args.data_txt_path=='step4_train':
            req_files = ['verse208.nii.gz', 'verse214.nii.gz','tm_BraTS20_Training_119.nii', 'tm_BraTS20_Training_138.nii',
            'verse214.nii.gz', 'verse208.nii.gz','verse226.nii.gz', 'tm_BraTS20_Training_120.nii',
            'verse267.nii.gz', 'verse255.nii.gz','tm_BraTS20_Training_189.nii', 'verse226.nii.gz',
            'verse214.nii.gz', 'verse232.nii.gz','tm_BraTS20_Training_119.nii', 'verse223.nii.gz',
            'tm_BraTS20_Training_369.nii', 'verse208.nii.gz','tm_BraTS20_Training_369.nii', 'tm_BraTS20_Training_119.nii',
            'tm_BraTS20_Training_369.nii', 'tm_BraTS20_Training_138.nii','verse255.nii.gz', 'tm_BraTS20_Training_189.nii',
            'verse232.nii.gz', 'verse214.nii.gz','verse223.nii.gz', 'tm_BraTS20_Training_189.nii',
            'tm_BraTS20_Training_120.nii', 'verse226.nii.gz','verse208.nii.gz', 'tm_BraTS20_Training_138.nii',
            'verse226.nii.gz', 'verse267.nii.gz','tm_BraTS20_Training_120.nii']
            
            
    elif 'val' in args.data_txt_path:  
        curr_type = 'val/'
        process_data=int(args.data_txt_path[4])
        req_files=[]
        req_files.extend(splitwise['step{}'.format(str(process_data))]['val'])
    
    
    ## training dict part
    train_img = []
    train_lbl = []
    train_post_lbl = []
    train_name = []
    

    for files in req_files:
        name = files
        train_img.append(image_path + curr_type + files)
        train_lbl.append(ann_path + curr_type + files)
        train_post_lbl.append(post_label_path + curr_type + files + '.h5')
        train_name.append(name)
        
    data_dicts_train = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(train_img, train_lbl, train_post_lbl, train_name)]
    print('train len {}'.format(len(data_dicts_train)))

    
    '''
    ## validation dict part
    val_img = []
    val_lbl = []
    val_post_lbl = []
    val_name = []
    for item in args.dataset_list:
        for line in open(args.data_txt_path + item +'_val.txt'):
            name = line.strip().split()[1].split('.')[0]
            val_img.append(args.data_root_path + line.strip().split()[0])
            val_lbl.append(args.data_root_path + line.strip().split()[1])
            val_post_lbl.append(args.data_root_path + name.replace('label', 'post_label') + '.h5')
            val_name.append(name)
    data_dicts_val = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(val_img, val_lbl, val_post_lbl, val_name)]
    print('val len {}'.format(len(data_dicts_val)))
    '''

    ## test dict part
    test_img = []
    test_lbl = []
    test_post_lbl = []
    test_name = []
    for files in req_files:
        name = files
        test_img.append(image_path + curr_type + files)
        test_lbl.append(ann_path + curr_type + files)
        test_post_lbl.append(post_label_path + curr_type + files + '.h5')
        test_name.append(name)
        
    data_dicts_test = [{'image': image, 'label': label, 'post_label': post_label, 'name': name}
                for image, label, post_label, name in zip(test_img, test_lbl, test_post_lbl, test_name)]
    print('test len {}'.format(len(data_dicts_test)))

    if args.phase == 'train':
        if args.cache_dataset:
            if args.uniform_sample:
                train_dataset = UniformCacheDataset(data=data_dicts_train, transform=train_transforms, cache_rate=args.cache_rate, datasetkey=args.datasetkey)
            else:
                train_dataset = CacheDataset(data=data_dicts_train, transform=train_transforms, cache_rate=args.cache_rate)
        else:
            if args.uniform_sample:
                train_dataset = UniformDataset(data=data_dicts_train, transform=train_transforms, datasetkey=args.datasetkey)
            else:
                train_dataset = Dataset(data=data_dicts_train, transform=train_transforms)
        train_sampler = DistributedSampler(dataset=train_dataset, even_divisible=True, shuffle=True) if args.dist else None
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, 
                                    collate_fn=list_data_collate, sampler=train_sampler)
        return train_loader, train_sampler
    
    
    if args.phase == 'validation':
        if args.cache_dataset:
            val_dataset = CacheDataset(data=data_dicts_val, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            val_dataset = Dataset(data=data_dicts_val, transform=val_transforms)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return val_loader, val_transforms
    
    
    if args.phase == 'test':
        if args.cache_dataset:
            test_dataset = CacheDataset(data=data_dicts_test, transform=val_transforms, cache_rate=args.cache_rate)
        else:
            test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
        return test_loader, val_transforms


def get_loader_without_gt(args):
    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            AddChanneld(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            # ToTemplatelabeld(keys=['label']),
            # RL_Splitd(keys=['label']),
            Spacingd(
                keys=["image"],
                pixdim=(args.space_x, args.space_y, args.space_z),
                mode=("bilinear"),
            ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=args.a_min,
                a_max=args.a_max,
                b_min=args.b_min,
                b_max=args.b_max,
                clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            ToTensord(keys=["image"]),
        ]
    )

    ## test dict part
    samples=['amos_0009.nii.gz','amos_0010.nii.gz','amos_0017.nii.gz','amos_0023.nii.gz']
    nii_files=[]
    for files in samples:
        nii_files.append(os.path.join(args.data_root_path,files))
    
    #nii_files = glob.glob(args.data_root_path + '/**.nii.gz')
    data_dicts_test = [{'image': image, 'name': image.split('/')[-1].split('.')[0]}
                for image in nii_files]
    print('test len {}'.format(len(data_dicts_test)))
    
    if args.cache_dataset:
        test_dataset = CacheDataset(data=data_dicts_test, transform=val_transforms, cache_rate=args.cache_rate)
    else:
        test_dataset = Dataset(data=data_dicts_test, transform=val_transforms)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=list_data_collate)
    return test_loader, val_transforms


if __name__ == "__main__":
    train_loader, test_loader = partial_label_dataloader()
    for index, item in enumerate(test_loader):
        print(item['image'].shape, item['label'].shape, item['task_id'])
        input()