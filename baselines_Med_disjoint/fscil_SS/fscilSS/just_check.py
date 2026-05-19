import shutil
import os


fscil_ckpt = "/hdd2/cil/running_base/fscil_SS/fscilSS/checkpoints/step/15ss-merged/"
fscil_data = "/hdd2/cil/running_base/fscil_SS/fscilSS/data/merged/"


shutil.copy('cil/FSS_organ_vanilla_code/vanilla_organ/checkpoints/step/15ss-merged/FT_0_newUNET_dynamic.pth',
fscil_ckpt)



'''
merged
│       ├── dataset
│       │   ├── annotations
│       │   │   ├── test
│       │   │   ├── train
│       │   │   └── val
│       │   ├── images
│       │   │   ├── test
│       │   │   ├── train
│       │   │   └── val
│       │   └── prediction
│       │       ├── ground_truth
│       │       ├── predicted_mask
│       │       └── test
│       │           ├── ground_truth
│       │           ├── images
│       │           └── predicted_mask
'''


'''
print("All directory making ")
os.makedirs(fscil_data+"dataset/",exist_ok=True)
os.makedirs(fscil_data+"dataset/images/",exist_ok=True)
os.makedirs(fscil_data+"dataset/annotations/",exist_ok=True)
os.makedirs(fscil_data+"dataset/prediction/",exist_ok=True)
os.makedirs(fscil_data+"dataset/prediction/ground_truth/",exist_ok=True)
os.makedirs(fscil_data+"dataset/prediction/predicted_mask/",exist_ok=True)
os.makedirs(fscil_data+"dataset/prediction/test/",exist_ok=True)
os.makedirs(fscil_data+"dataset/prediction/test/images/",exist_ok=True)
os.makedirs(fscil_data+"dataset/prediction/test/predicted_mask/",exist_ok=True)
os.makedirs(fscil_data+"dataset/prediction/test/ground_truth/",exist_ok=True)
'''