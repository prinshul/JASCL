#!/bin/bash

# export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
echo ${port}
#alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --num_workers 4"
alias exp="python -m torch.distributed.launch --nproc_per_node=1 run.py --num_workers 8"
shopt -s expand_aliases

ds=totalsegmentator

task='10ss'
gen_par="--task ${task} --dataset ${ds}"
lr=1e-3
iter=100
bs=2
path=checkpoints/step/${task}-${ds}

# exp --method FT --name FT --lr ${lr} ${gen_par} --step 0 --debug --batch_size ${bs} #--deeplab 'nonei
ns=5
is=0
inc_par="--ishot ${is} --input_mix novel --val_interval 50 --ckpt_interval 5 --no_pooling"

exp --method FT --name FT --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 0 --debug --nshot ${ns} --step_ckpt ${path}/FT_0.pth
# exp --method FT --name FT --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --debug --nshot ${ns} --step_ckpt ${path}/FT_0.pth
# exp --method FT --name FT --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 2 --debug --nshot ${ns} --step_ckpt ${path}/FT-s5-i0_1.pth
# exp --method FT --name FT --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 3 --debug --nshot ${ns} --step_ckpt ${path}/FT-s5-i0_2.pth
# exp --method FT --name FT --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 4 --debug --nshot ${ns} --step_ckpt ${path}/FT-s1-i0_3.pth
# exp --method FT --name FT --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 5 --debug --nshot ${ns} --step_ckpt ${path}/FT-s1-i0_4.pth
# exp --method FT --name FT --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 6 --debug --nshot ${ns} --step_ckpt ${path}/FT-s1-i0_5.pth

# for is in 0 1 2; do
#     inc_par="--ishot ${is} --input_mix novel --val_interval 50 --ckpt_interval 5 --no_pooling"
#     ns=3
   
#     exp --method FT --name FT --epochs 15 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --debug --nshot ${ns} --step_ckpt ${path}/FT_0.pth
# done
