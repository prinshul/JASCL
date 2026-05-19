import numpy as np
import os
from PIL import Image
import shutil
import sys
import pickle


labels_to_name = {
  0 :'unlabeled',
  1 :'ego vehicle',
  2 :'rectification border',
  3 :'out of roi',
  4 :'static',
  5 :'dynamic',
  6 :'ground',
  7 :'road', ## Take from here
  8 :'sidewalk',
  9 :'parking',
 10 :'rail track',
 11 :'building',
 12 :'wall',
 13 :'fence',
 14 :'guard rail', ## not to take
 15 :'bridge', ## not to take
 16 :'tunnel', ## not to take
 17 :'pole',
 18 :'polegroup', ## not to take
 19 :'traffic light',
 20 :'traffic sign',
 21 :'vegetation',
 22 :'terrain',
 23 :'sky',
 24 :'person',
 25 :'rider',
 26 :'car',
 27 :'truck',
 28 :'bus',
 29 :'caravan', ## not to take
 30 :'trailer', ## not to take
 31 :'train',
 32 :'motorcycle',
 33 :'bicycle',
 -1 :'licenseplate' ## not to take
}

'''
train
0 832
1 2975
2 1098
3 2975
4 2945
5 1365
6 1073
7 2934
8 2811
9 720
10 101
11 2934
12 970
13 1296
14 19
15 233
16 25
17 2949
18 223
19 1658
20 2808
21 2891
22 1654
23 2686
24 2343
25 1023
26 2832
27 359
28 274
29 59
30 72
31 142
32 513
33 1646


test
0 93
1 500
2 267
3 500
4 498
5 264
6 189
7 484
8 466
9 111
10 15
11 491
12 202
13 189
14 1
15 14
16 0
17 491
18 37
19 291
20 473
21 486
22 241
23 442
24 402
25 253
26 479
27 80
28 75
29 8
30 13
31 22
32 93
33 345

https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
'''

idd_path = "semi_fscil/natural/natural_data/cityscape/"
img_path_train = idd_path+"leftImg8bit/train/"
img_path_test = idd_path+"leftImg8bit/val/"

lbl_path_train = idd_path+"gtFine/train/"
lbl_path_test = idd_path+"gtFine/val/"


train_files = []

for path, subdirs, files in os.walk(lbl_path_train):
    for name in files:
        if name.endswith("gtFine_labelIds.png"):
            #print(os.path.join(path, name))
            train_files.append(os.path.join(path, name))
            
print("Total train files ",len(train_files))



test_files = []

for path, subdirs, files in os.walk(lbl_path_test):
    for name in files:
        if name.endswith("gtFine_labelIds.png"):
        #if name.endswith(".png"):
            #print(os.path.join(path, name))
            test_files.append(os.path.join(path, name))

print("Total test files ",len(test_files))



bbd_save = "semi_fscil/natural/dataset/step2/"
bbd_save_train = bbd_save+"train/"
bbd_save_val = bbd_save+"test/"


class_labels = [10,11,255]

def class_label_dict(gts_folder,class_labels):
    label_dict={}
    for lbls in class_labels:
        label_dict[lbls]=[]
    
    print(label_dict)
    
    if type(gts_folder) is list:    
        gts_files = gts_folder
    else:
        gts_files=os.listdir(gts_folder)
    print(gts_folder, len(gts_files))
    for files in gts_files:
        #print(files)
        sys.stdout.flush()
        if type(gts_folder) is list:    
            gt_read = np.array(Image.open(files))
        else:
            gt_read = np.array(Image.open(gts_folder+files))
        uniq=np.unique(gt_read)
        #print(uniq)
        for i in uniq:
            label_dict[i].append(files.split("/")[-1])
    return label_dict 
    
label_dict_train=class_label_dict(bbd_save_train+"labels/",class_labels)
label_dict_val=class_label_dict(bbd_save_val+"labels/",class_labels)


for key in label_dict_train:
    print(key, len(label_dict_train[key]))

for key in label_dict_val:
    print(key, len(label_dict_val[key]))


with open(bbd_save+'class_wise_train.pkl', 'wb') as file: 
    pickle.dump(label_dict_train,file)
    
with open(bbd_save+'class_wise_val.pkl', 'wb') as file: 
    pickle.dump(label_dict_val,file)
    


## Step 2 
req_labels = [9,21]
train_file_names = {}
test_file_names = {}

# Define the label mapping
label_mapping = {
    9:10,
    21:11
}


def mask_and_relabel(image_path, file_save_path):
    
    gt_read = np.array(Image.open(image_path))
    
    # Set to 255 all values not in classes_to_keep
    mask = np.isin(gt_read, req_labels, invert=True)
    gt_read[mask] = 255
    
    unique_classes_after_masking = np.unique(gt_read)
    
    if len(unique_classes_after_masking)==1:
        return 0
     
    print(f"Unique classes after masking in {image_path.split('/')[-4:]}: {unique_classes_after_masking}")
    
    for original, new_label in sorted(label_mapping.items(), key=lambda x: x[1]):
        gt_read[gt_read == original] = new_label
    
    unique_classes_after_relabel = np.unique(gt_read)
    print(f"Unique classes after relabeling in {image_path.split('/')[-4:]}: {unique_classes_after_relabel}")
    
    image_name = file_save_path+image_path.split("/")[-1]
    result_img = Image.fromarray(gt_read.astype(np.uint8))
    result_img.save(image_name)
    print("File saved at ",image_name)
    return 1
    
    
bbd_save = "semi_fscil/natural/dataset/step2/"
bbd_save_train = bbd_save+"train/"
bbd_save_val = bbd_save+"test/"


os.makedirs(bbd_save_train,exist_ok=True)
os.makedirs(bbd_save_train+"images/",exist_ok=True)
os.makedirs(bbd_save_train+"labels/",exist_ok=True)

os.makedirs(bbd_save_val,exist_ok=True)
os.makedirs(bbd_save_val+"images/",exist_ok=True)
os.makedirs(bbd_save_val+"labels/",exist_ok=True)


print("Required Classes")
for lbls in req_labels:
    print(labels_to_name[lbls]," : ",label_mapping[lbls])
    train_file_names[lbls]=[]
    test_file_names[lbls]=[]


print("\n\nPreparing Test files ")
print("Files taken from \n",lbl_path_test,"\n",img_path_test)
print("Files saved at \n",bbd_save_val+"labels/","\n",bbd_save_val+"images/")


train_count=0
for files in test_files:
    print(files)
    mask_sel = mask_and_relabel(files,bbd_save_val+"labels/")
    sys.stdout.flush()
    if mask_sel: 
        img_name = "/".join(files.replace("gtFine_labelIds","leftImg8bit").split("/")[-2:])
        shutil.copy(img_path_test+img_name,bbd_save_val+"images/")
        print("Copied ",img_name," to ",bbd_save_val+"images/")
        sys.stdout.flush()
        train_count+=1
    if train_count%100==0:
        print("Processed ", train_count," files.")
  
 

'''
print("Preparing train files ")
print("Files taken from \n",lbl_path_train,"\n",img_path_train)
print("Files saved at \n",bbd_save_train+"labels/","\n",bbd_save_train+"images/")
#print(train_file_names)

#print(len(os.listdir(bbd_save_train+"labels/")))

train_count=0
for files in train_files:
    mask_sel = mask_and_relabel(files,bbd_save_train+"labels/")
    if mask_sel:
        img_name = "/".join(files.replace("gtFine_labelIds","leftImg8bit").split("/")[-2:])
        #if "train" in files:
        shutil.copy(img_path_train+img_name,bbd_save_train+"images/")
        #if "val" in files:
            #shutil.copy(img_path_val+img_name,bbd_save_train+"images/")
        print("Copied ",img_name," to ",bbd_save_train+"images/")
        sys.stdout.flush()
        train_count+=1
        
    if train_count%100==0:
        print("Processed ", train_count," files.")
'''    


print('Step 2 files from cityscape complete.')
print('All files done')
