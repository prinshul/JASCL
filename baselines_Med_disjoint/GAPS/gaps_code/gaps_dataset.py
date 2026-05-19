import pickle5 as pkl
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import shutil
import fnmatch
import zipfile
import torch
from scipy import ndimage

import torch.nn.functional as F
import random

import sys
import os

# [] [] [] [] []


def print_lbl(img):
    print(img[-20:])
    image_rd = sitk.ReadImage(img)
    image_rd = sitk.GetArrayFromImage(image_rd)
    print(image_rd.shape)
    print(np.unique(image_rd))

def read_img(img):
    print(img)
    image_rd = sitk.ReadImage(img)
    image_rd = sitk.GetArrayFromImage(image_rd)
    #print(image_rd.shape)
    return image_rd



def flip_xz_yz(image, label):
    flip_id = np.array([1, np.random.randint(2), np.random.randint(2)]) * 2 - 1
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

class RandomGenerator(object):
    def __init__(self, output_size, mode):
        self.output_size = output_size
        self.mode = mode

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        min_value = np.min(image)
        # centercop
        # crop alongside with the ground truth
        # print(".............", np.unique(label))
        
            
        index = np.nonzero(label)
        index = np.transpose(index)  # 转置后变成二维矩阵，每一行有三个索引元素，分别对应z,x,y三个方向
        
        z_min = np.min(index[:, 0])
        z_max = np.max(index[:, 0])
        y_min = np.min(index[:, 1])
        y_max = np.max(index[:, 1])
        x_min = np.min(index[:, 2])
        x_max = np.max(index[:, 2])

        # middle point
        z_middle = np.int((z_min + z_max) / 2)
        y_middle = np.int((y_min + y_max) / 2)
        x_middle = np.int((x_min + x_max) / 2)

        Delta_z = np.int((z_max - z_min) / 3)  # 3
        Delta_y = np.int((y_max - y_min) / 4)  # 8
        Delta_x = np.int((x_max - x_min) / 4)  # 8

        # random number of x, y, z
        # z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)
        y_random = random.randint(y_middle - Delta_y, y_middle + Delta_y)
        x_random = random.randint(x_middle - Delta_x, x_middle + Delta_x)
        
        thre = z_min + Delta_z + np.int(self.output_size[0] / 2)
        if z_middle > thre:          # 此时z_middle + Delta_z < z_max
            delta_Z = z_middle - z_min - np.int(self.output_size[0] / 4)                         # 正常 np.int(self.output_size[0] / 2)，此时再大点，保证可以超出现有的范围
            z_random = random.randint(z_middle - delta_Z, z_middle + delta_Z)
        else:
            z_random = random.randint(z_middle - Delta_z, z_middle + Delta_z)

        # crop patch
        crop_z_down = z_random - np.int(self.output_size[0] / 2)
        crop_z_up = z_random + np.int(self.output_size[0] / 2)
        crop_y_down = y_random - np.int(self.output_size[1] / 2)
        crop_y_up = y_random + np.int(self.output_size[1] / 2)
        crop_x_down = x_random - np.int(self.output_size[2] / 2)
        crop_x_up = x_random + np.int(self.output_size[2] / 2)

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
 
        image = torch.from_numpy(image.astype(np.float)).float()
        label = torch.from_numpy(label.astype(np.float32)).float()

        sample = {'image': image, 'label': label.long(), 'case_name': sample['case_name']}

        return sample
    
    
    
data_path = "cil/processed_common_organ_data/merged/"
splits_path = "cil/processed_common_organ_data/merged/split/"
data_train = "cil/processed_common_organ_data/merged/dataset/images/train/"
ann_train = "cil/processed_common_organ_data/merged/dataset/annotations/train/"


data_exmp = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/images/exemplar/"
ann_exmp = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/annotations/exemplar/"

data_exmp1 = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/images/exemplar1/"
data_exmp2 = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/images/exemplar2/"

ann_exmp1 = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/annotations/exemplar1/"
ann_exmp2 = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/annotations/exemplar2/"

data_exmp_step1 = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/images/exemplar_step1/"
ann_exmp_step1 = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/dataset/annotations/exemplar_step1/"



os.makedirs(data_exmp1, exist_ok=True)
os.makedirs(ann_exmp1, exist_ok=True)

shutil.rmtree(data_exmp2)
shutil.rmtree(ann_exmp2)
os.makedirs(data_exmp2, exist_ok=True)
os.makedirs(ann_exmp2, exist_ok=True)

train_split = splits_path+"train.txt"

sample_list = open(train_split).readlines()
#print(sample_list)

train_pickle = splits_path+'inverse_dict_new_train.pkl'
class_to_images = pkl.load(open(train_pickle, 'rb'))
print(class_to_images)

step1_class = [16,17,18,19,20]
step2_class = [21,22,23,24,25,26]


step0_samples = os.listdir(ann_exmp)
print("Total base samples ", len(step0_samples))


step1_samples = os.listdir(ann_exmp_step1)
print("Total Step 1 samples ", len(step1_samples))

step2_samples = set()
for cl in step2_class:
    for sm in class_to_images[cl]:
        step2_samples.add(sm)
step2_samples = list(step2_samples)

print(step2_samples)

for i in range(0,len(step2_samples)):
    step2_samples[i] = sample_list[step2_samples[i]][:-1]+".nii.gz"
print("Step 2 = ",step2_samples)
print("Total step 2 samples ", len(step2_samples))
# [] [] [] [] []


print("All files print")

'''
for files in step0_samples:
    print("Step 0")
    #print_lbl(ann_exmp+files)

for files in step1_samples:
    print("Step 1")
    #print_lbl(ann_train+files)
    
for files in step2_samples:
    print("Step 2")
    #print_lbl(ann_train+files)
'''

Rnd_G = RandomGenerator(output_size=[128,160,96], mode = 'train')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print(torch.cuda.current_device())

print(step0_samples)
print(step1_samples)
print(step2_samples)
    
def copy_paste2(sample_step, sample_base,  data_exmp, ann_exmp,labels, name):
    base_img, base_lbl = sample_base['image'].to(device), sample_base['label'].to(device)
    step_img, step_lbl = sample_step['image'].to(device), sample_step['label'].to(device)
    
    print("labels received = ", labels)
    print(labels[0])
    lbl_ch = torch.where(step_lbl==labels[0], step_lbl, base_lbl)
    img_ch = torch.where(step_lbl==labels[0], step_img, base_img)
    
    if len(labels)>=2:
        print(labels[1])
        lbl_ch = torch.where(step_lbl==labels[1], step_lbl, lbl_ch)
        img_ch = torch.where(step_lbl==labels[1], step_img, img_ch)
    
    if len(labels)==3:
        print(labels[2])
        lbl_ch = torch.where(step_lbl==labels[2], step_lbl, lbl_ch)
        img_ch = torch.where(step_lbl==labels[2], step_img, img_ch)
    print(torch.unique(lbl_ch))
    
    lbl_wr = sitk.GetImageFromArray(lbl_ch.cpu().numpy())
    sitk.WriteImage(lbl_wr, ann_exmp+str(name)+".nii.gz")

    img_wr = sitk.GetImageFromArray(img_ch.cpu().numpy())
    sitk.WriteImage(img_wr, data_exmp+str(name)+".nii.gz")

    del step_img, step_lbl, base_img, base_lbl, lbl_wr, img_wr, lbl_ch, img_ch
    
# 2 or 3 organs from step 1 pasted on base img
def copy_paste_step1(step_samples, base_samples,data_exmp, ann_exmp,path_img,path_ann):
    torch.cuda.empty_cache()
    base_classes=16
    num_classes=21
    base_list = np.arange(0,len(base_samples))
    np.random.shuffle(base_list)
    base_len=len(base_list)
    
    k=1
    for sm in range(0,len(step_samples)):
        ## Step 1 image, label
        print("Step 1 sample = ",step_samples[sm])
        step_img = read_img(path_img+step_samples[sm])#.to(device)
        step_lbl = read_img(path_ann+step_samples[sm])#.to(device)
        thresh_step = (int(num_classes) - 1) + 0.5
        step_lbl[step_lbl < 0.5] = 0.0  # maybe some voxels is a minus value
        step_lbl[step_lbl > thresh_step] = 255
        sample_step_unedit = {'image': step_img, 'label': step_lbl, 'case_name':step_samples[sm][:-7]}
        
        ## Base step img, lbl
        print("Base step sample = ",base_samples[base_list[sm%base_len]])
        base_img = read_img(data_exmp+base_samples[base_list[sm%base_len]])#.to(device)
        base_lbl = read_img(ann_exmp+base_samples[base_list[sm%base_len]])#.to(device)
        thresh_base = (int(base_classes) - 1) + 0.5
        base_lbl[base_lbl < 0.5] = 0.0  # maybe some voxels is a minus value
        base_lbl[base_lbl > thresh_base] = 255
        sample_base_unedit = {'image': base_img, 'label': base_lbl, 'case_name':base_samples[base_list[sm%base_len]][:-7]}
        
        while True:
            sample_base = Rnd_G(sample_base_unedit)
            sample_step = Rnd_G(sample_step_unedit)
        
            step_labels = torch.unique(sample_step['label'])[1:]
            print(step_labels)

            base_labels = torch.unique(sample_base['label'])
            print(base_labels)
            
            if len(step_labels)>0:
                break
        
        
        len_step_labels = len(step_labels)
        name = str(sample_step['case_name'])+"_"+str(sample_base['case_name'])+"_"+str(k)
        if len_step_labels==1:
            copy_paste2(sample_step, sample_base, data_exmp1, ann_exmp1, labels=step_labels, name=name+"_a")
        elif len_step_labels==2:
            copy_paste2(sample_step, sample_base, data_exmp1, ann_exmp1, labels=step_labels, name=name+"_b")
        elif len_step_labels==3:
            copy_paste2(sample_step, sample_base, data_exmp1, ann_exmp1, labels=step_labels[:2], name=name+"_c")
            copy_paste2(sample_step, sample_base, data_exmp1, ann_exmp1, labels=step_labels[2:], name=name+"_d")
        elif len_step_labels==4:
            copy_paste2(sample_step, sample_base, data_exmp1, ann_exmp1, labels=step_labels[:2], name=name+"_e")
            copy_paste2(sample_step, sample_base, data_exmp1, ann_exmp1, labels=step_labels[2:], name=name+"_f")
        else:
            copy_paste2(sample_step, sample_base, data_exmp1, ann_exmp1, labels=step_labels[:2], name=name+"_g")
            copy_paste2(sample_step, sample_base, data_exmp1, ann_exmp1, labels=step_labels[2:], name=name+"_h")
            
        del sample_step['image'],sample_step['label'],sample_base['image'],sample_base['label']
        k+=1
        sys.stdout.flush()
    print("\n\nStep 2 data for GAPS generated")
        
# [] [] [] [] []




##############################################################################################################################
##############################################################################################################################



def copy_paste_double(sample_jump, sample_base, step_list, sm,  data_exmp, ann_exmp,labels, name):
    base_img, base_lbl = sample_base['image'].to(device), sample_base['label'].to(device)
    jump_img, jump_lbl = sample_jump['image'].to(device), sample_jump['label'].to(device)
    
    
    ## Step 1 image, label
    step_classes=21
    print("Step 1 sample = ",step1_samples[step_list[sm]])
    step_img = read_img(data_exmp_step1+step1_samples[step_list[sm]])#.to(device)
    step_lbl = read_img(ann_exmp_step1+step1_samples[step_list[sm]])#.to(device)
    thresh_step = (int(step_classes) - 1) + 0.5
    step_lbl[step_lbl < 0.5] = 0.0  # maybe some voxels is a minus value
    step_lbl[step_lbl > thresh_step] = 255
    sample_step_unedit = {'image': step_img, 'label': step_lbl, 'case_name':step1_samples[step_list[sm]][:-7]}
    
    while True:
        sample_step = Rnd_G(sample_step_unedit)

        step_labels = torch.unique(sample_step['label'])[1:]
        print("Step 1 labels = ",step_labels)
        
        if len(step_labels)>1:
            break
    
    print("labels received = ", labels)
    step1_img, step1_lbl = sample_step['image'].to(device), sample_step['label'].to(device)
    
    
    step1_paste_arr = np.arange(0,len(step_labels))
    step1_paste_list=np.random.choice(step1_paste_arr,size=2,replace=False)
    
    
    lbl_ch = torch.where(step1_lbl==step_labels[step1_paste_list[0]], step1_lbl, base_lbl)
    img_ch = torch.where(step1_lbl==step_labels[step1_paste_list[0]], step1_img, base_img)
    
    lbl_ch = torch.where(step1_lbl==step_labels[step1_paste_list[1]], step1_lbl, lbl_ch)
    img_ch = torch.where(step1_lbl==step_labels[step1_paste_list[1]], step1_img, img_ch)
    
    lbl_ch = torch.where(jump_lbl==labels[0], jump_lbl, lbl_ch)
    img_ch = torch.where(jump_lbl==labels[0], jump_img, img_ch)
    
    
    if len(labels)>=2:
        print(labels[1])
        lbl_ch = torch.where(jump_lbl==labels[1], jump_lbl, lbl_ch)
        img_ch = torch.where(jump_lbl==labels[1], jump_img, img_ch)
    
    print("Classes attached = ",torch.unique(lbl_ch))
    print("\n\n")
    lbl_wr = sitk.GetImageFromArray(lbl_ch.cpu().numpy())
    sitk.WriteImage(lbl_wr, ann_exmp+str(name)+"_"+str(step1_samples[step_list[sm]])+".nii.gz")

    img_wr = sitk.GetImageFromArray(img_ch.cpu().numpy())
    sitk.WriteImage(img_wr, data_exmp+str(name)+"_"+str(step1_samples[step_list[sm]])+".nii.gz")

    del step_img, step_lbl, base_img, base_lbl, lbl_wr, img_wr, lbl_ch, img_ch, sample_step['label'], sample_step['image']

    
## Step 1 is sampled here, only 1 organ from step 1 and 2 from step 2
def copy_paste_step2(jump_samples,step_samples,base_samples,data_exmp,ann_exmp,path_img,path_ann):
    base_classes=16
    num_classes=27
    
    base_list = np.arange(0,len(base_samples))
    np.random.shuffle(base_list)
    base_len=len(base_list)
    
    step_arr = np.arange(0,len(step_samples))
    step_list=np.random.choice(step_arr,size=len(jump_samples),replace=False)
    step_len=len(step_list)
    
    k=1
    for sm in range(0,len(jump_samples)):
        ## Step 2 image, label
        print("Step 2 sample = ",jump_samples[sm])
        jump_img = read_img(path_img+jump_samples[sm])#.to(device)
        jump_lbl = read_img(path_ann+jump_samples[sm])#.to(device)
        thresh_jump = (int(num_classes) - 1) + 0.5
        jump_lbl[jump_lbl < 0.5] = 0.0  # maybe some voxels is a minus value
        jump_lbl[jump_lbl > thresh_jump] = 255
        sample_jump_unedit = {'image': jump_img, 'label': jump_lbl, 'case_name':jump_samples[sm][:-7]}
        
        
        ## Base step img, lbl
        print("Base step sample = ",base_samples[base_list[sm%base_len]])
        base_img = read_img(data_exmp+base_samples[base_list[sm%base_len]])#.to(device)
        base_lbl = read_img(ann_exmp+base_samples[base_list[sm%base_len]])#.to(device)
        thresh_base = (int(base_classes) - 1) + 0.5
        base_lbl[base_lbl < 0.5] = 0.0  # maybe some voxels is a minus value
        base_lbl[base_lbl > thresh_base] = 255
        sample_base_unedit = {'image': base_img, 'label': base_lbl, 'case_name':base_samples[base_list[sm%base_len]][:-7]}
        
        
        while True:
            sample_base = Rnd_G(sample_base_unedit)
            sample_jump = Rnd_G(sample_jump_unedit)
        
            jump_labels = torch.unique(sample_jump['label'])[1:]
            print(jump_labels)

            base_labels = torch.unique(sample_base['label'])
            print(base_labels)
            
            if len(jump_labels)>0:
                break
        
    
        len_jump_labels = len(jump_labels)
        name = str(sample_jump['case_name'])+"_"+str(sample_base['case_name'])+"_"+str(k)
        if len_jump_labels==1:
            copy_paste_double(sample_jump, sample_base,step_list,sm, data_exmp2, ann_exmp2, labels=jump_labels, name=name+"_a")
        elif len_jump_labels==2:
            copy_paste_double(sample_jump, sample_base,step_list,sm, data_exmp2, ann_exmp2, labels=jump_labels, name=name+"_b")
        elif len_jump_labels==3:
            copy_paste_double(sample_jump, sample_base,step_list,sm, data_exmp2, ann_exmp2, labels=jump_labels[:2], name=name+"_c")
            copy_paste_double(sample_jump, sample_base,step_list,sm, data_exmp2, ann_exmp2, labels=jump_labels[2:], name=name+"_d")
        elif len_jump_labels==4:
            copy_paste_double(sample_jump, sample_base,step_list,sm, data_exmp2, ann_exmp2, labels=jump_labels[:2], name=name+"_e")
            copy_paste_double(sample_jump, sample_base,step_list,sm, data_exmp2, ann_exmp2, labels=jump_labels[2:], name=name+"_f")
        elif len_jump_labels==5:
            copy_paste_double(sample_jump, sample_base,step_list,sm, data_exmp2, ann_exmp2, labels=jump_labels[:2], name=name+"_g")
            copy_paste_double(sample_jump, sample_base,step_list,sm, data_exmp2, ann_exmp2, labels=jump_labels[2:4], name=name+"_h")
            copy_paste_double(sample_jump, sample_base,step_list,sm, data_exmp2, ann_exmp2, labels=jump_labels[4:], name=name+"_i")
        else:
            copy_paste_double(sample_jump, sample_base,step_list,sm, data_exmp2, ann_exmp2, labels=jump_labels[:2], name=name+"_j")
            copy_paste_double(sample_jump, sample_base,step_list,sm, data_exmp2, ann_exmp2, labels=jump_labels[2:4], name=name+"_k")
            copy_paste_double(sample_jump, sample_base,step_list,sm, data_exmp2, ann_exmp2, labels=jump_labels[4:], name=name+"_l")
            
        k+=1    
        del sample_jump['image'],sample_jump['label'],sample_base['image'],sample_base['label']
        #break
        #sys.stdout.flush()
    print("\n\nStep 2 data for GAPS generated")
# [] [] [] [] []






def main(argv, arc):
    args = argv[1:]
    print(args)
    ## Prepare data for step 1 GAPS
    if args[0]=="step1":
        print("Step 1 data making ")
        copy_paste_step1(step1_samples, step0_samples,data_exmp, ann_exmp, path_img=data_train, path_ann=ann_train)
    else:
        print("Step 2 data making ")
        ## Prepare data for step 2 GAPS
        copy_paste_step2(step2_samples,step1_samples,step0_samples,data_exmp,ann_exmp,path_img=data_train,path_ann=ann_train)


if __name__ == "__main__":
    main(sys.argv, len(sys.argv))