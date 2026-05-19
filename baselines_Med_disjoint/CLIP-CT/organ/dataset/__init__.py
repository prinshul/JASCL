from .voc import VOCFSSDataset, VOCSegmentation
from .verse import VerSESDataset, VerSESegmentation
from .totalsegmentator import TotalSegmentatorSegmentation, TotalSegmentatorDataset
from .merged_data import MergedSegmentation, MergedDataset
from .cityscapes import CityscapesFSSDataset
from .coco import COCOFSS, COCO, COCOStuffFSS
from .ade import AdeSegmentation
# from .transform import Compose, RandomScale, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, \
#     CenterCrop, Resize, RandomResizedCrop, ColorJitter, PadCenterCrop, ToPILImage
from monai.transforms.transform import Transform, MapTransform
import random
from .utils import Subset, MyImageFolder, RandomDataset
from torchvision import transforms
from dataset.merged_data import RandomGenerator
import numpy as np
from monai.config import DtypeLike, KeysCollection

from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    SpatialCropd,
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
from monai.data.image_reader import ImageReader
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple, Union
from monai.utils.enums import PostFix
from monai.transforms.io.array import LoadImage, SaveImage
from monai.utils import GridSamplePadMode, ensure_tuple, ensure_tuple_rep

DEFAULT_POST_FIX = PostFix.meta()

TRAIN_CV = 0.8

def get_dataset(opts, task, train=True):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'cts':
        dataset = CityscapesFSSDataset
        train_transform = transform.Compose([
            transform.RandomScale((0.7, 2)),  # Using RRC should be (0.25, 0.75)
            transform.RandomCrop(opts.crop_size),
            transform.RandomHorizontalFlip(),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        val_transform = transform.Compose([
            transform.CenterCrop(size=opts.crop_size),
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        test_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
    elif opts.dataset == 'voc' or 'coco' in opts.dataset:
        if opts.dataset == 'voc':
            dataset = VOCFSSDataset
        else:
            if 'stuff' in opts.dataset:
                dataset = COCOStuffFSS
            else:
                dataset = COCOFSS

        train_transform = Compose([
            RandomScale((0.5, 2)),
            RandomCrop(opts.crop_size, pad_if_needed=True),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])
        val_transform = Compose([
            # Resize(size=opts.crop_size_test),
            PadCenterCrop(size=opts.crop_size_test),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])
        test_transform = Compose([
            # PadCenterCrop(size=opts.crop_size_test),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406],
                      std=[0.229, 0.224, 0.225]),
        ])
    elif opts.dataset == 'verse':
        dataset = VerSESDataset
    elif opts.dataset == 'totalsegmentator':
        dataset = TotalSegmentatorDataset
    elif opts.dataset == 'merged':
        dataset = MergedDataset
        train_transform = Compose([
            RandomGenerator(opts.step, output_size=opts.img_size, mode = 'train'),
            AddChanneld(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),

            Spacingd(
                    keys=["image", "label"],
                    pixdim=(opts.space_x, opts.space_y, opts.space_z),
                    mode=("bilinear", "nearest"),
                ), # process h5 to here
            ScaleIntensityRanged(
                keys=["image"],
                a_min=opts.a_min,
                a_max=opts.a_max,
                b_min=opts.b_min,
                b_max=opts.b_max,
                clip=True,
            ),
            SpatialPadd(keys=["image", "label"], spatial_size=(opts.roi_x, opts.roi_y, opts.roi_z), mode='constant'),

            RandZoomd(keys=["image", "label"], prob=0.3, min_zoom=1.3, max_zoom=1.5, mode=['area', 'nearest']), # 7
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(opts.roi_x, opts.roi_y, opts.roi_z), #192, 192, 64
                pos=2,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
             RandRotate90d(
                keys=["image", "label"],
                prob=0.10,
                max_k=3,
            ),
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.10,
                prob=0.20,
            ),

            ToTensord(keys=["image", "label"]),

            # RandomGenerator(opts.step, output_size=opts.img_size, mode = 'train'),
        ]
    )

        val_transform = Compose(
            [
                RandomGenerator(opts.step, output_size=opts.img_size, mode = 'train'),
                AddChanneld(keys=["image", "label"]),
                ScaleIntensityRanged(
                    keys=["image"],
                    a_min=opts.a_min,
                    a_max=opts.a_max,
                    b_min=opts.b_min,
                    b_max=opts.b_max,
                    clip=True,
                ),
                
                ToTensord(keys=["image", "label"]),
            ])
        
        transform_error= Compose(
            [
                RandomGenerator(opts.step, output_size=opts.img_size, mode = 'train'),
                AddChanneld(keys=["image", "label"]),
                SpatialCropd(keys=["image", "label"], roi_center = (0, 0, 0),
                roi_size=(opts.roi_x, opts.roi_y, opts.roi_z),
                ),
                ToTensord(keys=["image", "label"]),
            ]
        )
    else:
        raise NotImplementedError

    if train:
        if opts.cross_val:
            train_dst = dataset(root=opts.data_root, task=task, train=True, transform=None)
            train_len = int(TRAIN_CV * len(train_dst))
            idx = list(range(len(train_dst)))
            random.shuffle(idx)
            train_dst = Subset(train_dst, idx[:train_len], train_transform)
            val_dst = Subset(train_dst, idx[train_len:], val_transform)
            train_dst_noaug = Subset(train_dst, idx[:train_len], train_transform)
        else:
            # train_dst = dataset(root=opts.data_root, task=task, base_dir=opts.train_root_path, train="train", 
            #                 num_classes=opts.num_classes,
            #                    transform=transforms.Compose([RandomGenerator(opts.step, output_size=opts.img_size, mode = 'train')]))


            # val_dst = dataset(root=opts.data_root, task=task, base_dir=opts.val_root_path, train="val", num_classes=opts.num_classes,
            #                    transform=transforms.Compose(
            #                        [RandomGenerator(opts.step, output_size=opts.img_size, mode = 'val')]))
            train_dst = dataset(root=opts.data_root, task=task, base_dir=opts.train_root_path, train="train", 
                            num_classes=opts.num_classes,
                               transform_error=transform_error, transform=train_transform)


            val_dst = dataset(root=opts.data_root, task=task, base_dir=opts.val_root_path, train="val", num_classes=opts.num_classes,
                               transform_error=transform_error,transform=val_transform)

            
        return train_dst, val_dst
    else:
        print("FROM __INIT__ IN DATASET: ", opts.data_root, opts.train_root_path, opts.num_classes)
        test_dst_all = dataset(root=opts.data_root, task=task, base_dir=opts.train_root_path, train="test", 
                            num_classes=opts.num_classes, transform_error=transform_error, transform=val_transform)
        test_dst_novel = dataset(root=opts.data_root, task=task, base_dir=opts.train_root_path, train="test", 
                            num_classes=opts.num_classes, transform_error=transform_error, transform=val_transform , masking=True)
       
        # test_dst_all = dataset(root=opts.data_root, task=task, base_dir=opts.train_root_path, train="test", 
        #                     num_classes=opts.num_classes, )#transform=transforms.Compose([RandomGenerator(output_size=opts.img_size, mode = 'train')]))
        # test_dst_novel = dataset(root=opts.data_root, task=task, base_dir=opts.train_root_path, train="test", 
        #                     num_classes=opts.num_classes, )#transform=transforms.Compose([RandomGenerator(output_size=opts.img_size, mode = 'train')]) , masking=True)
        return test_dst_all, test_dst_novel
