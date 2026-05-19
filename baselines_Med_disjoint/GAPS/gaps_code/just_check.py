import os 
import shutil 


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


print(len(os.listdir(ann_exmp2)))
#shutil.rmtree(data_exmp2)
#shutil.rmtree(ann_exmp2)