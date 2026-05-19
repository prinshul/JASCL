import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset
import argparse
from torch.utils.data import DataLoader
import SimpleITK as sitk
join = os.path.join
from tqdm import tqdm
import json
import pickle5 as pkl
import itertools
from sklearn.model_selection import train_test_split
import shutil
import random
import numpy as np

random.seed(1024)
np.random.seed(1024)
torch.manual_seed(1024)
torch.cuda.manual_seed(1024)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True


def dict_items_flat(pdict):
    x = list(pdict.values())
    flat_list = list(itertools.chain(*x))
    
    return list(set(flat_list))

def class_map(pdict, class_map_dict):
    pdict_map = {}
    
    for key, value in pdict.items():
        pdict_map[class_map_dict[key]] = value
    return pdict_map

def seg_func(img_pid, img_path, seg_path, path_img_comb, path_anno_comb, filename):
    print(filename)
    img_path = os.path.join(img_path)
    image = sitk.ReadImage(img_path)
    
    label_path = os.path.join(seg_path)
    label = sitk.ReadImage(label_path)
    
    label = sitk.GetArrayFromImage(label)

    seg_class = np.unique(label)
    fname = filename.split(".")[0]
    new_nm = fname.split("_")[0][-3:]
    # print(new_nm)

    # for seg_cl in seg_class:
    #     if seg_cl != 0:
    #         if manuf.strip() == 'Philips':
    #             label[label == seg_cl] = seg_cl + 28
    #         elif manuf.strip() == 'SIEMENS':
    #             label[label == seg_cl] = seg_cl + 2*28
    #         elif manuf.strip() == 'TOSHIBA':
    #             label[label == seg_cl] = seg_cl + 3*28
    
    new_seg_class = np.unique(label)
    
    label_new = sitk.GetImageFromArray(label)
    
    sitk.WriteImage(image, join(path_img_comb, img_pid + '.nii.gz'))
    sitk.WriteImage(label_new, join(path_anno_comb, filename))
    return new_seg_class, new_nm

def preprocess(pdict, pdict_path, path_img_comb, path_anno_comb, data_path):
    count = 0
    for root, dirs, files in tqdm(os.walk(data_path)):
        count += 1
        # if i > 54:
        for filename_idx in range(len(files)):
            if '.nii.gz' in files[filename_idx] and 'gl' not in files[filename_idx]:
                seg_pid = files[filename_idx].split('.')[0]
                img_pid = seg_pid.replace("seg-vert_msk", "ct")

                seg_path = join(root, seg_pid)
                img_path = join(root.replace("derivatives", "rawdata"), img_pid)


                f = open(img_path+'.json')
                dict_ = json.load(f)
                seg_class, idx = seg_func(img_pid, img_path, seg_path, path_img_comb, path_anno_comb, files[filename_idx])
                class_names = [str(int(classes)) +"_"+ "class" for classes in list(seg_class) if classes != 0]

                for classes in class_names:
                    if classes not in pdict.keys():
                        pdict[classes] = [idx]
                    else:
                        if (pdict[classes] is not None) and (root[-3:] not in pdict[classes]):
                            pdict[classes] += [idx]

                    with open(pdict_path, 'wb') as handle:
                        pkl.dump(pdict, handle, protocol=pkl.HIGHEST_PROTOCOL)
        print(count)
        
def seg_func_test(img_pid, img_path, seg_path, path_img_test, path_anno_test, filename):
    print(filename)
    img_path = os.path.join(img_path)
    image = sitk.ReadImage(img_path)
    
    label_path = os.path.join(seg_path)
    label = sitk.ReadImage(label_path)
    
    label = sitk.GetArrayFromImage(label)

    seg_class = np.unique(label)
    fname = filename.split(".")[0]
    new_nm = fname.split("_")[0][-3:]

    # for seg_cl in seg_class:
    #     if seg_cl != 0:
    #         if manuf.strip() == 'Philips':
    #             label[label == seg_cl] = seg_cl + 28
    #         elif manuf.strip() == 'SIEMENS':
    #             label[label == seg_cl] = seg_cl + 2*28
    #         elif manuf.strip() == 'TOSHIBA':
    #             label[label == seg_cl] = seg_cl + 3*28
    
    new_seg_class = np.unique(label)
    
    label_new = sitk.GetImageFromArray(label)
    sitk.WriteImage(image, join(path_img_test, new_nm + '.nii.gz'))
    sitk.WriteImage(label_new, join(path_anno_test, new_nm + '.nii.gz'))
    return new_seg_class, new_nm

def preprocess_test(pdict, pdict_path, path_img_test, path_anno_test, data_path):
    count = 0
    
    for root, dirs, files in tqdm(os.walk(data_path)):
        count += 1
        # if i > 54:
        for filename_idx in range(len(files)):
            if '.nii.gz' in files[filename_idx] and 'gl' not in files[filename_idx]:
                seg_pid = files[filename_idx].split('.')[0]
                img_pid = seg_pid.replace("seg-vert_msk", "ct")

                seg_path = join(root, seg_pid)
                img_path = join(root.replace("derivatives", "rawdata"), img_pid)


                f = open(img_path+'.json')
                dict_ = json.load(f)
                seg_class, idx = seg_func_test(img_pid, img_path, seg_path, path_img_test, path_anno_test, files[filename_idx])
                class_names = [str(int(classes)) +"_"+ "class" for classes in list(seg_class) if classes != 0]

                for classes in class_names:
                    if classes not in pdict.keys():
                        pdict[classes] = [idx]
                    else:
                        if (pdict[classes] is not None) and (root[-3:] not in pdict[classes]):
                            pdict[classes] += [idx]

                with open(pdict_path, 'wb') as handle:
                    pkl.dump(pdict, handle, protocol=pkl.HIGHEST_PROTOCOL)
        print(count)

def main():
    data_tr_path = '/home/cil/totalsegmentator/dataset-01training/derivatives'
    data_val_path = '/home/cil/totalsegmentator/dataset-02validation/derivatives'
    # data_test_path = '/home/jupyter/MedSAM/VerSe2020/dataset-02test/derivatives'
    
    path_img_comb = '/home/cil/FSS_26_TS/data/combined/images'
    path_anno_comb = '/home/cil/FSS_26_TS/data/combined/annotations'
    
    dst_img_path_train = '/home/cil/FSS_26_TS/data/totalsegmentator/dataset/images/train'
    dst_img_path_test = '/home/cil/FSS_26_TS/data/totalsegmentator/dataset/images/val'

    dst_seg_path_train = '/home/cil/FSS_26_TS/data/totalsegmentator/dataset/annotations/train'
    dst_seg_path_test = '/home/cil/FSS_26_TS/data/totalsegmentator/dataset/annotations/val'

    dst_split_path = '/home/cil/FSS_26_TS/data/totalsegmentator/split'
    pdict_path = join(dst_split_path, 'inverse_dict.pkl')
                          
    # if not os.path.exists(dst_split_path):
    #     os.makedirs(dst_split_path)
    
    # if not os.path.exists(path_img_comb):
    #     os.makedirs(path_img_comb)
                          
    # if not os.path.exists(path_anno_comb):
    #     os.makedirs(path_anno_comb)
        
    # pdict = {}
    
    # # with open(pdict_path, 'wb') as handle:
    # #     pkl.dump(pdict, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    # preprocess(pdict, pdict_path, path_img_comb, path_anno_comb, data_tr_path)
    # print("Training done")
    
    # preprocess(pdict, pdict_path, path_img_comb, path_anno_comb, data_val_path)
    # print("Validation done")
    
    f = open(pdict_path, 'rb')
    inverse_dict_comb = pkl.load(f)
    
    x = list(inverse_dict_comb.values())
    flat_list = list(itertools.chain(*x))
    p_ids = list(set(flat_list))
    inverse_dict_comb['background'] = p_ids
    
    if not os.path.exists(dst_split_path):
        os.makedirs(dst_split_path)
    
    with open(join(dst_split_path, "label.txt"),"w") as f:
        for i in range(29):
            if i == 0:
                sname = "{}\tbackground".format(i)
            else:
                sname = "{}\t{}_{}".format(i, i, 'class')
            f.write(sname + "\n")
            

    # with open(join(dst_split_path, 'inverse_dict.pkl'), 'wb') as handle:
    #         pkl.dump(inverse_dict_comb, handle, protocol=pkl.HIGHEST_PROTOCOL)
            
    f = open(join(dst_split_path, 'inverse_dict.pkl'), 'rb')
    inverse_dict_comb = pkl.load(f)

    patient_slice_ids = dict_items_flat(inverse_dict_comb)

    pids = [idx.split("_")[0] for idx in patient_slice_ids]
    pid_unq = list(set(pids))
    pid_train, pid_test = train_test_split(pid_unq, test_size=0.3, random_state=42)

    class_map_dict = {}
    with open(join(dst_split_path, "label.txt"),"r") as f:
        for line in f:
            line = line.split('\t')
            if not line:  # empty line?
                continue
            class_map_dict[line[1].strip()] = int(line[0].strip())

    pdict_train = {}
    pdict_test = {}
    for key, value in inverse_dict_comb.items():
        val_unq = [value[i].split("_")[0] for i in range(len(value))] 
        train_value = list(set(val_unq).intersection(set(pid_train)))
        val_value = list(set(val_unq).intersection(set(pid_test)))

        pdict_train[key] = [value[i] for i in range(len(value)) if value[i].split("_")[0] in train_value]
        pdict_test[key] = [value[i] for i in range(len(value)) if value[i].split("_")[0] in val_value]

    pdict_train_map = class_map(pdict_train, class_map_dict)
    pdict_test_map = class_map(pdict_test, class_map_dict)

    # with open(join(dst_split_path, 'inverse_dict_train.pkl'), 'wb') as handle:
    #     pkl.dump(pdict_train_map, handle, protocol=pkl.HIGHEST_PROTOCOL)
    # with open(join(dst_split_path, 'inverse_dict_val.pkl'), 'wb') as handle:
    #     pkl.dump(pdict_test_map, handle, protocol=pkl.HIGHEST_PROTOCOL)

    patient_slice_ids_train = dict_items_flat(pdict_train_map)
    patient_slice_ids_test = dict_items_flat(pdict_test_map)

    with open(join(dst_split_path, "train.txt"),"w") as f:
        for elem in patient_slice_ids_train:
                f.write(elem + "\n")

    with open(join(dst_split_path, "val.txt"),"w") as f:
        for elem in patient_slice_ids_test:
                f.write(elem + "\n")
                
    if not os.path.exists(dst_img_path_train):
        os.makedirs(dst_img_path_train)

    if not os.path.exists(dst_img_path_test):
        os.makedirs(dst_img_path_test)

    if not os.path.exists(dst_seg_path_train):
        os.makedirs(dst_seg_path_train)

    if not os.path.exists(dst_seg_path_test):
        os.makedirs(dst_seg_path_test)
        
    files = os.listdir(path_anno_comb)
    for filename_idx in range(len(files)):
        if '.nii.gz' in files[filename_idx]:
            fname = files[filename_idx].split(".")[0]
            new_nm = fname.split("_")[0][-3:]
            fname_seg = files[filename_idx].split('.')[0] + '.nii.gz'
            fname_img = fname_seg.replace("seg-vert_msk", "ct")
            if new_nm in patient_slice_ids_train:
                shutil.copy(join(path_img_comb, fname_img), join(dst_img_path_train, new_nm + '.nii.gz'))
                shutil.copy(join(path_anno_comb, fname_seg), join(dst_seg_path_train, new_nm + '.nii.gz'))
            elif new_nm in patient_slice_ids_test:
                shutil.copy(join(path_img_comb, fname_img), join(dst_img_path_test, new_nm + '.nii.gz'))
                shutil.copy(join(path_anno_comb, fname_seg), join(dst_seg_path_test, new_nm + '.nii.gz'))
                
    data_test_path = '/home/cil/totalsegmentator/dataset-03test/derivatives'
   
    dst_img_path_test_final = '/home/cil/FSS_26_TS/data/totalsegmentator/dataset/images/test'
    dst_seg_path_test_final = '/home/cil/FSS_26_TS/data/totalsegmentator/dataset/annotations/test'

    dst_split_path = '/home/cil/FSS_26_TS/data/totalsegmentator/split'
    pdict_path_test = join(dst_split_path, 'inverse_dict_test.pkl')
   
    pdict = {}
   
    if not os.path.exists(dst_img_path_test_final):
        os.makedirs(dst_img_path_test_final)

    if not os.path.exists(dst_seg_path_test_final):
        os.makedirs(dst_seg_path_test_final)
        
    # with open(pdict_path_test, 'wb') as handle:
    #     pkl.dump(pdict, handle, protocol=pkl.HIGHEST_PROTOCOL)
   
    preprocess_test(pdict, pdict_path_test, dst_img_path_test_final, dst_seg_path_test_final, data_test_path)
    print("Test done")
       
    f = open(pdict_path_test, 'rb')
    inverse_dict_comb = pkl.load(f)
   
    x = list(inverse_dict_comb.values())
    flat_list = list(itertools.chain(*x))
    p_ids = list(set(flat_list))
    inverse_dict_comb['background'] = p_ids

    
    class_map_dict = {}
    with open(join(dst_split_path, "label.txt"),"r") as f:
        for line in f:
            line = line.split('\t')
            if not line:  # empty line?
                continue
            class_map_dict[line[1].strip()] = int(line[0].strip())
    
    pdict_test_final_map = class_map(inverse_dict_comb, class_map_dict)

    # with open(join(dst_split_path, 'inverse_dict_test.pkl'), 'wb') as handle:
    #         pkl.dump(pdict_test_final_map, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    patient_slice_ids_test_final = dict_items_flat(pdict_test_final_map)
    with open(join(dst_split_path, "test.txt"),"w") as f:
        for elem in patient_slice_ids_test_final:
                f.write(elem + "\n")
                
    dict_ = pkl.load(open(join(dst_split_path, "inverse_dict_test.pkl", "rb")))
    my_file = open(join(dst_split_path, "test.txt"), "r")
    
    # reading the file
    data = my_file.read()
    pids = data.split("\n")
    pids = pids[:-1]

    new_dict = {}
    for cl, img_set in dict_.items():
        l = []
        for val in img_set:
            l.append(pids.index(val))
        new_dict[cl] = l

    # with open(join(dst_split_path,"inverse_dict_new_test.pkl", 'wb')) as handle:
    #     pkl.dump(new_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    dict_ = pkl.load(open(join(dst_split_path, "inverse_dict_train.pkl", "rb")))
    my_file = open(join(dst_split_path, "train.txt"), "r")
    
    # reading the file
    data = my_file.read()
    pids = data.split("\n")
    pids = pids[:-1]

    new_dict = {}
    for cl, img_set in dict_.items():
        l = []
        for val in img_set:
            l.append(pids.index(val))
        new_dict[cl] = l

    # with open(join(dst_split_path,"inverse_dict_new_train.pkl", 'wb')) as handle:
    #     pkl.dump(new_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    dict_ = pkl.load(open(join(dst_split_path, "inverse_dict_val.pkl", "rb")))
    my_file = open(join(dst_split_path, "val.txt"), "r")
    
    # reading the file
    data = my_file.read()
    pids = data.split("\n")
    pids = pids[:-1]

    new_dict = {}
    for cl, img_set in dict_.items():
        l = []
        for val in img_set:
            l.append(pids.index(val))
        new_dict[cl] = l

    # with open(join(dst_split_path,"inverse_dict_new_val.pkl", 'wb')) as handle:
    #     pkl.dump(new_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)
        
if __name__ == '__main__':
    main()
