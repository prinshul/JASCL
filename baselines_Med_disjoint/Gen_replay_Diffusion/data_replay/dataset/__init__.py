from .voc import VOCFSSDataset, VOCSegmentation
from .verse import VerSESDataset, VerSESegmentation
from .totalsegmentator import TotalSegmentatorSegmentation, TotalSegmentatorDataset
from .merged_data import MergedSegmentation, MergedDataset
from .cityscapes import CityscapesFSSDataset
from .coco import COCOFSS, COCO, COCOStuffFSS
from .ade import AdeSegmentation
# from .transform import Compose, RandomScale, RandomCrop, RandomHorizontalFlip, ToTensor, Normalize, \
#     CenterCrop, Resize, RandomResizedCrop, ColorJitter, PadCenterCrop, ToPILImage

import random
from .utils import Subset, MyImageFolder, RandomDataset
from torchvision import transforms
from dataset.merged_data import RandomGenerator
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
            train_dst_noaug = Subset(train_dst, idx[:train_len], test_transform)
        else:
            train_dst = dataset(root=opts.data_root, task=task, base_dir=opts.train_root_path, train="train", 
                            num_classes=opts.num_classes,
                               transform=transforms.Compose([RandomGenerator(output_size=opts.img_size, mode = 'train')]))


            val_dst = dataset(root=opts.data_root, task=task, base_dir=opts.val_root_path, train="val", num_classes=opts.num_classes,
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=opts.img_size, mode = 'val')]))
            
            exemp_dst = dataset(root=opts.data_root, task=task, base_dir=opts.train_root_path, train="exemplar",
                                num_classes=opts.num_classes,transform=transforms.Compose(
                                   [RandomGenerator(output_size=opts.img_size, mode = 'train')]))
            
        return train_dst, val_dst , exemp_dst
    else:
        print("FROM __INIT__ IN DATASET: ", opts.data_root, opts.train_root_path, opts.num_classes)
        test_dst_all = dataset(root=opts.data_root, task=task, base_dir=opts.train_root_path, train="test", 
                            num_classes=opts.num_classes, transform=transforms.Compose([RandomGenerator(output_size=opts.img_size, mode = 'test')]))
        test_dst_novel = dataset(root=opts.data_root, task=task, base_dir=opts.train_root_path, train="test", 
                            num_classes=opts.num_classes, transform=transforms.Compose([RandomGenerator(output_size=opts.img_size, mode = 'test')]) , masking=True)
       
        # test_dst_all = dataset(root=opts.data_root, task=task, base_dir=opts.train_root_path, train="test", 
        #                     num_classes=opts.num_classes, )#transform=transforms.Compose([RandomGenerator(output_size=opts.img_size, mode = 'train')]))
        # test_dst_novel = dataset(root=opts.data_root, task=task, base_dir=opts.train_root_path, train="test", 
        #                     num_classes=opts.num_classes, )#transform=transforms.Compose([RandomGenerator(output_size=opts.img_size, mode = 'train')]) , masking=True)
        return test_dst_all, test_dst_novel
