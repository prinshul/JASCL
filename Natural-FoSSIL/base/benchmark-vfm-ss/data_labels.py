import pickle 
import mmengine
import os.path as osp
import os 


BDD_datadir = '/home/hk_proj/natural_seg/final_dataset/step1/'
IDD_datadir = '/home/hk_proj/natural_seg/final_dataset/step2/'
step3_datadir = '/home/hk_proj/natural_seg/final_dataset/step3/'

shots_datadir = '/home/hk_proj/natural_seg/final_dataset/shots/'

with open(shots_datadir+'gt_train_step1.pkl', 'rb') as file: 
    gt_train=pickle.load(file)
        
with open(shots_datadir+'gt_val_step1.pkl', 'rb') as file: 
    gt_val=pickle.load(file)


out_dir = BDD_datadir+"splits/"
mmengine.mkdir_or_exist(out_dir)

split_names = ["train", "val", "test"]

for split in split_names:
    if split=='train':
        filenames = gt_train
    elif split=='val':
        filenames = gt_val
    elif split=='test':
        filenames = os.listdir(BDD_datadir+"test/labels/")
    
    with open(osp.join(out_dir, f"{split}.txt"), "w") as f:
        f.writelines(f + "\n" for f in filenames)

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