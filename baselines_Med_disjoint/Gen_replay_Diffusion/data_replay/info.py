import numpy as np 
import SimpleITK as sitk
import os
import shutil
import sys

def print_lbl(img):
    print(img[-40:])
    image_rd = sitk.ReadImage(img)
    image_rd = sitk.GetArrayFromImage(image_rd)
    print(image_rd.shape)
    print(np.unique(image_rd))

def read_img(img):
    print(img)
    image_rd = sitk.ReadImage(img)
    image_rd = sitk.GetArrayFromImage(image_rd)
    print(image_rd.shape)
    print(np.unique(image_rd))
    #return image_rd

'''
data_dir = "cil/processed_common_organ_data/merged/dataset/"
test_img = data_dir+"images/test/"
test_ann = data_dir+"images/test/"


print(os.listdir(test_img))


samp_ch = ['18', '21']

for files in samp_ch:
    print_lbl(test_img+files+'.nii.gz')
    print_lbl(test_ann+files+'.nii.gz')
'''


diff_replay = "/hdd2/cil/running_base/data_replay_Diffusion/data_replay/"
exemplar_img = diff_replay+"data/merged/dataset/images/exemplar/"
exemplar_ann = diff_replay+"data/merged/dataset/annotations/exemplar/"

'''
for files in os.listdir(exemplar_ann):
    print_lbl(exemplar_img+files)
    print_lbl(exemplar_ann+files)
    sys.stdout.flush()

'''

'''
gen_files = ['2_332.nii.gz', '1_352.nii.gz', '2_396.nii.gz', '2_224.nii.gz', '1_366.nii.gz', '1_362.nii.gz', 
'1_320.nii.gz', '2_47.nii.gz', '1_287.nii.gz', '2_87.nii.gz', '1_396.nii.gz', '1_120.nii.gz',
 '2_352.nii.gz', '2_216.nii.gz', '2_320.nii.gz', '2_112.nii.gz', '1_87.nii.gz', '2_366.nii.gz', 
 '1_112.nii.gz', '2_281.nii.gz', '2_287.nii.gz', '1_332.nii.gz', '2_129.nii.gz', '1_47.nii.gz', 
 '1_216.nii.gz', '2_120.nii.gz', '1_224.nii.gz', '2_362.nii.gz', '1_281.nii.gz', '1_129.nii.gz',
  '1_167.nii.gz', '2_167.nii.gz', '2_649.nii.gz', '1_1018.nii.gz', '2_1384.nii.gz', '2_1017.nii.gz',
   '2_1299.nii.gz', '1_638.nii.gz', '1_1371.nii.gz', '2_263.nii.gz', '1_769.nii.gz', '1_734.nii.gz', 
   '1_600.nii.gz', '1_636.nii.gz', '1_1017.nii.gz', '1_1112.nii.gz', '2_894.nii.gz', '1_797.nii.gz',
    '1_613.nii.gz', '2_970.nii.gz', '2_461.nii.gz', '2_1264.nii.gz', '2_1380.nii.gz', '1_649.nii.gz', 
    '1_865.nii.gz', '2_797.nii.gz', '2_1043.nii.gz', '2_865.nii.gz', '1_1152.nii.gz', '2_897.nii.gz', 
    '1_172.nii.gz', '2_1112.nii.gz', '2_769.nii.gz', '1_1299.nii.gz', '2_1371.nii.gz', '2_1093.nii.gz',
     '1_1384.nii.gz', '2_600.nii.gz', '1_897.nii.gz', '2_638.nii.gz', '2_1018.nii.gz', '2_777.nii.gz', 
     '1_1380.nii.gz', '2_613.nii.gz', '1_518.nii.gz', '1_1043.nii.gz', '2_172.nii.gz', '1_970.nii.gz', 
     '2_1152.nii.gz', '1_1093.nii.gz', '1_894.nii.gz', '1_292.nii.gz', '1_777.nii.gz', '2_734.nii.gz',
     '2_292.nii.gz', '2_636.nii.gz', '1_263.nii.gz', '1_1264.nii.gz', '2_518.nii.gz', '1_461.nii.gz']

diff_replay = "/hdd2/cil/running_base/data_replay_Diffusion/data_replay/"
exemplar_path = diff_replay+"data/merged/dataset/annotations/exemplar/"

gen_files.sort()
copied_files = os.listdir(exemplar_path)
copied_files.sort()

if gen_files==copied_files:
    print("Files match")
'''