#!/bin/bash

# export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
echo ${port}
#alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --num_workers 4"
#-m torch.distributed.launch --nproc_per_node=1 
alias exp="python run.py --num_workers 4"
shopt -s expand_aliases

ds=merged
#num_classes=11,19,25,29,36

task='15ss'
gen_par="--task ${task} --dataset ${ds}" #--num_classes ${num_classes}"
lr=1e-3
iter=100
bs=1
path=checkpoints/step/${task}-${ds}

#exp --method FT --name FT --lr ${lr} ${gen_par} --step 0 --debug --batch_size ${bs} #--deeplab 'nonei
ns=5
is=0
inc_par="--ishot ${is} --val_interval 50 --ckpt_interval 5 --no_pooling"

# vanilla 
#exp --method FT --name FT --num_classes 11 --lr ${lr} ${gen_par} --vanila --step 0 --debug --batch_size ${bs}
#exp --method FT --name FT --num_classes 19 --vanila --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --debug --input_mix novel --nshot ${ns} --step_ckpt ${path}/FT_0_swin_unetr_dynamic.pth
#exp --method FT --name FT --num_classes 25 --vanila --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 2 --debug --input_mix both --nshot ${ns} --step_ckpt ${path}/FT-s5-i0_1_swin_unetr_dynamic.pth
#exp --method FT --name FT --num_classes 29 --vanila --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 3 --debug --input_mix novel --nshot ${ns} --step_ckpt ${path}/FT-s5-i0_2_swin_unetr_dynamic.pth
exp --method FT --name FT --num_classes 36 --vanila --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 4 --debug --input_mix novel --nshot ${ns} --step_ckpt ${path}/FT-s5-i0_3_swin_unetr_dynamic.pth




# WI
# exp --method PIFS --name FT --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --debug --nshot ${ns} --step_ckpt ${path}/FT_0_newUNET_dynamic.pth

# with KD loss
# exp --method PIFS --name PIFS --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --debug --nshot ${ns} --step_ckpt ${path}/FT_0_3DUNET_dynamic.pth --born_again
# exp --method FT --name FT --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 2 --debug --nshot ${ns} --step_ckpt ${path}/FT-s5-i0_1_3DUNET_dynamic.pth --born_again --ckd --loss_kd 0.5

# with KD loss, feature loss and body distillation
# exp --method FT --name FT --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --debug --nshot ${ns} --step_ckpt ${path}/FT_0_3DUNET_dynamic.pth --born_again --loss_kd 0.1 --loss_de 0.1 --l1_loss 0.1 #--ckd
# exp --method FT --name FT --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 2 --debug --nshot ${ns} --step_ckpt ${path}/FT-s5-i0_1_3DUNET_dynamic.pth --born_again --loss_kd 0.1 --loss_de 0.1 --l1_loss 0.1 #--ckd

# Batch Renormalization
# exp --method PIFS --name PIFS --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --debug --nshot ${ns} --step_ckpt ${path}/FT_0_3DUNET_dynamic.pth --born_again --loss_kd 0.3 --loss_de 0.3 --l1_loss 0.3 --ckd

# for is in 0 1 2; do
#     inc_par="--ishot ${is} --input_mix novel --val_interval 50 --ckpt_interval 5 --no_pooling"
#     ns=3
   
#     exp --method FT --name FT --epochs 15 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --debug --nshot ${ns} --step_ckpt ${path}/FT_0.pth
# done
