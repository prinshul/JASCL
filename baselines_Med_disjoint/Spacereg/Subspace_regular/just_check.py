import os 
import shutil

subs_data = "/hdd2/cil/running_base/space_reg/Subspace_regular/data/merged/"


#shutil.rmtree(subs_data+"dataset/prediction/")

print("All directory making ")
os.makedirs(subs_data+"dataset/",exist_ok=True)
os.makedirs(subs_data+"dataset/images/",exist_ok=True)
os.makedirs(subs_data+"dataset/annotations/",exist_ok=True)
os.makedirs(subs_data+"dataset/prediction/",exist_ok=True)
os.makedirs(subs_data+"dataset/prediction/ground_truth/",exist_ok=True)
os.makedirs(subs_data+"dataset/prediction/predicted_mask/",exist_ok=True)
os.makedirs(subs_data+"dataset/prediction/test/",exist_ok=True)
os.makedirs(subs_data+"dataset/prediction/test/images/",exist_ok=True)
os.makedirs(subs_data+"dataset/prediction/test/predicted_mask/",exist_ok=True)
os.makedirs(subs_data+"dataset/prediction/test/ground_truth/",exist_ok=True)
