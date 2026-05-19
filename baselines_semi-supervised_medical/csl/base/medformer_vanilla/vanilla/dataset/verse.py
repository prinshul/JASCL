import os.path as osp
import torch.utils.data as data
from .dataset import FSSDataset
import numpy as np
import pickle5 as pkl

from PIL import Image

import os
import random
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import cv2
import argparse
from torchvision import transforms
from torch.utils.data import DataLoader
import SimpleITK as sitk

random.seed(1024)
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
torch.backends.cudnn.deterministic = True


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

def random_rot_flip(image, label):
    # k--> angle
    # i, j: axis
    k = np.random.randint(0, 4)
    axis = random.sample(range(0, 3), 2)
    image = np.rot90(image, k, axes=(axis[0], axis[1]))  # rot along z axis
    label = np.rot90(label, k, axes=(axis[0], axis[1]))

    flip_id = np.array([np.random.randint(2), np.random.randint(2), np.random.randint(2)]) * 2 - 1
    image = np.ascontiguousarray(image[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    label = np.ascontiguousarray(label[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    return image, label


def random_rotate(image, label, min_value):
    angle = np.random.randint(-15, 15)  # -20--20
    rotate_axes = [(0, 1), (1, 2), (0, 2)]
    k = np.random.randint(0, 3)
    image = ndimage.interpolation.rotate(image, angle, axes=rotate_axes[k], reshape=False, order=3, mode='constant',
                                         cval=min_value)
    label = ndimage.interpolation.rotate(label, angle, axes=rotate_axes[k], reshape=False, order=0, mode='constant',
                                         cval=0.0)

    return image, label


# z, y, x     0, 1, 2
def rot_from_y_x(image, label):
    # k = np.random.randint(0, 4)
    image = np.rot90(image, 2, axes=(1, 2))  # rot along z axis
    label = np.rot90(label, 2, axes=(1, 2))

    return image, label


def flip_xz_yz(image, label):
    flip_id = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
    image = np.ascontiguousarray(image[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    label = np.ascontiguousarray(label[::flip_id[0], ::flip_id[1], ::flip_id[2]])
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, mode):
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        min_value = np.min(image)
        # centercop
        # crop alongside with the ground truth

        index = np.nonzero(label)
        index = np.transpose(index)  # 转置后变成二维矩阵，每一行有三个索引元素，分别对应z,x,y三个方向


        z_min = np.min(index[:, 0])
        z_max = np.max(index[:, 0])
        y_min = np.min(index[:, 1])
        y_max = np.max(index[:, 1])
        x_min = np.min(index[:, 2])
        x_max = np.max(index[:, 2])

        # middle point
        z_middle = int((z_min + z_max) / 2)
        y_middle = int((y_min + y_max) / 2)
        x_middle = int((x_min + x_max) / 2)

        Delta_z = int((z_max - z_min) / 3)  # 3
        Delta_y = int((y_max - y_min) / 4)  # 8
        Delta_x = int((x_max - x_min) / 4)  # 8

        # random number of x, y, z
        # z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)
        y_random = random.randint(y_middle - Delta_y, y_middle + Delta_y)
        x_random = random.randint(x_middle - Delta_x, x_middle + Delta_x)
        
        thre = z_min + Delta_z + int(self.output_size[0] / 2)
        if z_middle > thre:          # 此时z_middle + Delta_z < z_max
            delta_Z = z_middle - z_min - int(self.output_size[0] / 4)                         # 正常 int(self.output_size[0] / 2)，此时再大点，保证可以超出现有的范围
            z_random = random.randint(z_middle - delta_Z, z_middle + delta_Z)
        else:
            z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)

        # crop patch
        crop_z_down = z_random - int(self.output_size[0] / 2)
        crop_z_up = z_random + int(self.output_size[0] / 2)
        crop_y_down = y_random - int(self.output_size[1] / 2)
        crop_y_up = y_random + int(self.output_size[1] / 2)
        crop_x_down = x_random - int(self.output_size[2] / 2)
        crop_x_up = x_random + int(self.output_size[2] / 2)

        # padding
        if crop_z_down < 0 or crop_z_up > image.shape[0]:
            delta_z = np.maximum(np.abs(crop_z_down), np.abs(crop_z_up - image.shape[0]))
            image = np.pad(image, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=min_value)
            label = np.pad(label, ((delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=0.0)
            crop_z_down = crop_z_down + delta_z
            crop_z_up = crop_z_up + delta_z

        if crop_y_down < 0 or crop_y_up > image.shape[1]:
            delta_y = np.maximum(np.abs(crop_y_down), np.abs(crop_y_up - image.shape[1]))
            image = np.pad(image, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=min_value)
            label = np.pad(label, ((0, 0), (delta_y, delta_y), (0, 0)), 'constant', constant_values=0.0)
            crop_y_down = crop_y_down + delta_y
            crop_y_up = crop_y_up + delta_y

        if crop_x_down < 0 or crop_x_up > image.shape[2]:
            delta_x = np.maximum(np.abs(crop_x_down), np.abs(crop_x_up - image.shape[2]))
            image = np.pad(image, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=min_value)
            label = np.pad(label, ((0, 0), (0, 0), (delta_x, delta_x)), 'constant', constant_values=0.0)
            crop_x_down = crop_x_down + delta_x
            crop_x_up = crop_x_up + delta_x

        label = label[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]
        image = image[crop_z_down: crop_z_up, crop_y_down: crop_y_up, crop_x_down: crop_x_up]

        label = np.round(label)

        # data augmentation
        if self.mode == 'train':
            if random.random() > 0.5:
                image, label = flip_xz_yz(image, label)
            if random.random() > 0.5:                      # elif random.random() > 0.5:
                image, label = random_rotate(image, label, min_value)
                label = np.round(label)

        image = torch.from_numpy(image.astype(float)).unsqueeze(0).float()
        label = torch.from_numpy(label.astype(np.float32)).float()

        sample = {'image': image, 'label': label.long(), 'case_name': sample['case_name']}

        return sample
    
class VerSESegmentation(data.Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Segmentation Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        train (bool): Use train (True) or test (False) split
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """

    def __init__(self, root, base_dir, split, num_classes, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        
        self.data_dir = base_dir
        self.num_classes = num_classes

        ds_root = osp.join(root, self.data_dir)
        print("ds_root:", ds_root)
        splits_dir = osp.join(ds_root, 'split')
        
        self.sample_list = open(os.path.join(splits_dir, self.split + '.txt')).readlines()
        
        if self.split == 'train':
            path = '/inverse_dict_new_train.pkl'
            self.class_to_images_ = pkl.load(open(splits_dir + path, 'rb'))
        elif self.split == 'val':
            path = '/inverse_dict_new_val.pkl'
            self.class_to_images_ = pkl.load(open(splits_dir + path, 'rb'))
        else:
            path = '/inverse_dict_new_test.pkl'
            self.class_to_images_ = pkl.load(open(splits_dir + path, 'rb'))
        file_idx = {}
        for i, file in enumerate(self.sample_list):
            file_idx[file.strip()] = i
       
        self.images_to_idx_ = file_idx
            
    def __len__(self):
        return len(self.sample_list)

    @property
    def class_to_images(self):
        return self.class_to_images_
    @property
    def images_to_idx(self):
        return self.images_to_idx_
    
    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip('\n')
            img_path = os.path.join(self.data_dir, 'dataset', 'images', 'train', slice_name + '.nii.gz')
            image = sitk.ReadImage(img_path)
            label_path = os.path.join(self.data_dir, 'dataset', 'annotations', 'train', slice_name + '.nii.gz')
            label = sitk.ReadImage(label_path)
            origin = np.array(image.GetOrigin())
            spacing = np.array(image.GetSpacing())
            image = sitk.GetArrayFromImage(image)
            label = sitk.GetArrayFromImage(label)
            # print("label path:", label_path)
            # print("labels1:", np.unique(label, return_index=True))
            # print("labels1:", np.unique(label))
    
            # print(slice_name)
            # # print(label[0, :, :])
            # #print(np.unique(label, return_index=True))
            # print("labels2:", np.unique(label, return_index=True))

        elif self.split == "val":
            slice_name = self.sample_list[idx].strip('\n')
            img_path = os.path.join(self.data_dir, 'dataset', 'images', 'val', slice_name + '.nii.gz')
            image = sitk.ReadImage(img_path)
            label_path = os.path.join(self.data_dir, 'dataset', 'annotations', 'val', slice_name + '.nii.gz')
            label = sitk.ReadImage(label_path)
            origin = np.array(image.GetOrigin())
            spacing = np.array(image.GetSpacing())
            image = sitk.GetArrayFromImage(image)
            label = sitk.GetArrayFromImage(label)
        else:
            slice_name = self.sample_list[idx].strip('\n')
            img_path = os.path.join(self.data_dir, 'dataset', 'images', 'test', slice_name + '.nii.gz')
            image = sitk.ReadImage(img_path)
            label_path = os.path.join(self.data_dir, 'dataset', 'annotations', 'test', slice_name + '.nii.gz')
            label = sitk.ReadImage(label_path)
            
            origin = np.array(image.GetOrigin())
            spacing = np.array(image.GetSpacing())
            
            image = sitk.GetArrayFromImage(image)
            label = sitk.GetArrayFromImage(label)

        label[label < 0.5] = 0.0  # maybe some voxels is a minus value
        label[label > 25.5] = 0.0
        
        # print(label[0,:,:])
        #print(np.unique(label, return_index=True))
        # print("labels3:", np.unique(label, return_index=True))
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        sample['case_name'] = self.sample_list[idx].strip('\n')

        sample['origin'] = origin
        sample['spacing'] = spacing
        return sample

class VerSESDataset(FSSDataset):
    def make_dataset(self, root, base_dir, split, num_classes):
        full_voc = VerSESegmentation(root, base_dir, split, num_classes, transform=None)
        return full_voc