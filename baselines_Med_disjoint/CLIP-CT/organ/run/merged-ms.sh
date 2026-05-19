#!/bin/bash

# export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
echo ${port}
#alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --num_workers 4"
alias exp="python -m torch.distributed.launch --nproc_per_node=1 run.py --num_workers 8"
shopt -s expand_aliases

ds=merged
# num_classes=16
# num_classes=21
num_classes=27

task='15ss'
gen_par="--task ${task} --dataset ${ds} --num_classes ${num_classes}"
lr=1e-3
iter=100
bs=1
path=checkpoints/step/${task}-${ds}

# exp --method FT --name FT --lr ${lr} ${gen_par} --continue_ckpt --step 0 --debug --batch_size ${bs} --step_ckpt ${path}/FT_0_SwinUNETR_partial_dynamic.pth #--deeplab 'nonei
ns=5
is=0
inc_par="--ishot ${is} --input_mix both --val_interval 50 --ckpt_interval 5 --no_pooling"

# vanilla 
# exp --method FT --name FT --continue_ckpt --vanila --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --continue_ckpt --step 1 --debug --nshot ${ns} --step_ckpt ${path}/FT_0_SwinUNETR_partial_dynamic.pth --born_again --strict_IL --load_vanilla_base_model #--test_old
exp --method FT --name FT --vanila --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 2 --debug --nshot ${ns} --step_ckpt ${path}/FT-s5-i0_1_SwinUNETR_partial_dynamic.pth --born_again --strict_IL

 