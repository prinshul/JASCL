from PIL import Image
import numpy as np
import pickle 
import os.path as osp
import os 
import random
import shutil

BDD_datadir = '/home/hk_proj/natural_seg/final_dataset/step1/'
IDD_datadir = '/home/hk_proj/natural_seg/final_dataset/step2/'
step3_datadir = '/home/hk_proj/natural_seg/final_dataset/step3/'
shots_datadir = '/home/hk_proj/natural_seg/final_dataset/shots/'

img_path = BDD_datadir+"train/images/"
label_path = BDD_datadir+"train/labels/"

with open(shots_datadir+'gt_train_step1.pkl', 'rb') as file: 
    gt_train=pickle.load(file)
        
with open(shots_datadir+'gt_val_step1.pkl', 'rb') as file: 
    gt_val=pickle.load(file)

print("Total files ", len(os.listdir(img_path))," ",len(os.listdir(label_path)))

out_dir = "data/"
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
img_out_path = out_dir+"images/"
label_out_path = out_dir+"labels/"
os.makedirs(out_dir, exist_ok=True)
os.makedirs(img_out_path, exist_ok=True)
os.makedirs(label_out_path, exist_ok=True)

labels = set()
n = 30
 
samples = random.sample(gt_val, n)

for files in samples:
    print(files)
    shutil.copy(img_path+files[:-3]+"jpg",img_out_path)
    shutil.copy(label_path+files,label_out_path)
    for lbl in np.unique(Image.open(label_path+files).convert('P')):
        labels.add(lbl)


print(labels)


'''
#### Step 2    
    
nshot = str(args.nshot)
print("\nTaking files from path ",shots_datadir+'nshot_'+nshot)
with open(shots_datadir+'nshot_'+nshot+'/IDD_train_step2.pkl', 'rb') as file: 
    train_gt_files=pickle.load(file)
    
with open(shots_datadir+'nshot_'+nshot+'/IDD_val_step2.pkl', 'rb') as file: 
    val_gt_files=pickle.load(file)
    
    
    
    
#### Step 3

nshot = str(args.nshot)
print("\nTaking files from path ",shots_datadir+'nshot_'+nshot)

with open(shots_datadir+'nshot_'+nshot+'/IDD_repeat_train_step3.pkl', 'rb') as file: 
    step3_idd_rep_train_files=pickle.load(file)

with open(shots_datadir+'nshot_'+nshot+'/IDD_repeat_val_step3.pkl', 'rb') as file: 
    step3_idd_rep_val_files = pickle.load(file) 
    
'''