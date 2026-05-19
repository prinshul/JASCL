env name for object detection : ood_det




Incremental steps - https://github.com/microsoft/SoftTeacher/


# 1. Prepare splits - divide classes and ratio, update configs with class names
nohup python tools/prepare_split.py > data_split.log &

# 2. Base training
nohup python tools/train.py configs/soft_incremental/base.py > base_train.log &


# 3. Incremental Step 1
## Make model path changes in file or give as arguments
nohup python tools/train.py configs/soft_incremental/step1.py > step1_train.log &

## --cfg-options load_from='work_dirs/base/epoch.pth'

# 4. Incremental Step 2
## Make model path changes in file or give as arguments
nohup python tools/train.py configs/soft_incremental/step2.py > step2_train.log &

# Testing - make path changes in file or give as arguments
nohup python tools/eval_incremetal.py > results.log &

## Before testing, make sure to combine json files for current and previous steps.
nohup python tools/merge_coco_jsons.py --ann ood_coco_splits/base/cartoon_test.json ood_coco_splits/step_1/sketch_test.json --out ood_coco_splits/step_1/upto_step1_test.json > merge_step1.log &

nohup python tools/merge_coco_jsons.py --ann ood_coco_splits/base/cartoon_test.json ood_coco_splits/step_1/sketch_test.json ood_coco_splits/step_2/painting_test.json --out ood_coco_splits/step_2/upto_step2_test.json > merge_step2.log &


mmdetection/
│
├── mmdet/
│   configs/
│   ├── soft_incremental/
│   │   ├── base.py
│   │   ├── step1.py
│   │   ├── step2.py
│   model/detectors/soft_teacher.py
├── tools/
│   ├── prepare_split.py
│   ├── train.py
│   ├── eval_incremental.py
│   ├── merge_coco_jsons.py


'''
#nohup python tools/train.py configs/ssl_incremental/coco_o_base.py --work-dir work_dirs/ssl_coco_o_base > base_train.log &

#nohup python tools/train.py configs/ssl_incremental/coco_o_step1.py --cfg-options load_from=work_dirs/ssl_coco_o_base/best_coco_bbox_mAP_epoch_10.pth > step1_train.log &
'''




train:
verse074.nii.gz
verse082.nii.gz
verse091.nii.gz
verse096.nii.gz
verse097.nii.gz
verse112.nii.gz
verse127.nii.gz
verse135.nii.gz
verse145.nii.gz
verse151.nii.gz
verse201.nii.gz
verse202.nii.gz
verse207.nii.gz
verse208.nii.gz
verse212.nii.gz
verse214.nii.gz
verse215.nii.gz
verse223.nii.gz
verse226.nii.gz
verse232.nii.gz
verse239.nii.gz
verse243.nii.gz
verse251.nii.gz
verse254.nii.gz
verse255.nii.gz
verse258.nii.gz
verse265.nii.gz
verse266.nii.gz
verse267.nii.gz
verse270.nii.gz
verse272.nii.gz
verse275.nii.gz

val:

verse209.nii.gz
verse221.nii.gz
verse225.nii.gz
verse230.nii.gz
verse235.nii.gz


now similarly verse the abopve files are present at below path you have to copy the _seg files rename as above and copy into same paths train files to "<Path_to_Data_Root>/Med_FoSSIL_Mixed_data/dataset/annotations/train"
and val files to "<Path_to_Data_Root>/Med_FoSSIL_Mixed_data/dataset/annotations/val"



rse113_snapshot.png  verse202.nii.gz        verse253_ctd.json      verse275_seg.nii.gz
verse046.nii.gz        verse082_ctd.json      verse122.nii.gz        verse202_ctd.json      verse253_seg.nii.gz    verse275_snapshot.png
