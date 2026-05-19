#!/bin/bash

# export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
echo ${port}
#alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --num_workers 4"
alias exp="python run.py --num_workers 4"
shopt -s expand_aliases

ds=merged
#num_classes=16
# num_classes=21
# num_classes=27
# num_classes=31
# num_classes=34
# num_classes=38

task='15ss'
gen_par="--task ${task} --dataset ${ds}" # --num_classes ${num_classes}"
lr=1e-3
iter=100
bs=2
path=../../../base/codu_run/codu/checkpoints/step/${task}-${ds}

ns=5
is=0
inc_par="--ishot ${is} --input_mix novel --val_interval 50 --ckpt_interval 5 " #--no_pooling"




## incremental commands (uncomment the current step to be run and comment remaining then run  "sh run/merged.sh")
exp --load_weight_strategy --vanila --num_classes 21 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --step_ckpt ${path}/FT_0_newUNET_dynamic.pth 
#exp --load_weight_strategy --vanila --num_classes 27 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 2 --step_ckpt ./checkpoints/step/15ss-merged/Experiment-s5-i0_1_newUNET_dynamic.pth
#exp --load_weight_strategy --vanila --num_classes 31 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 3 --step_ckpt ./checkpoints/step/15ss-merged/Experiment-s5-i0_2_newUNET_dynamic.pth 
#exp --load_weight_strategy --vanila --num_classes 34 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 4 --step_ckpt ./checkpoints/step/15ss-merged/Experiment-s5-i0_3_newUNET_dynamic.pth 
# exp --load_weight_strategy --vanila --num_classes 38 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 5 --step_ckpt ./checkpoints/step/15ss-merged/Experiment-s5-i0_4_newUNET_dynamic.pth 








