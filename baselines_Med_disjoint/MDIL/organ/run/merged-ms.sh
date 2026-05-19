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

# exp --method FT --name FT --lr ${lr} ${gen_par} --step 0 --debug --batch_size ${bs} --test #--deeplab 'nonei
ns=5
is=0
inc_par="--ishot ${is} --input_mix novel --val_interval 50 --ckpt_interval 5 --no_pooling"

# vanilla 
# exp --method FT --name FT --vanila --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --debug --nshot ${ns} --step_ckpt ${path}/FT_0_erfnet_RA_parallel_dynamic.pth
exp --method FT --name FT --vanila --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 2 --debug --nshot ${ns} --step_ckpt ${path}/FT-s5-i0_1_erfnet_RA_parallel_dynamic.pth --test
