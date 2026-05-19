#!/bin/bash

# export CUDA_VISIBLE_DEVICES=$1
#port=$(python get_free_port.py)
#echo ${port}
#alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --num_workers 4"
#alias exp="python -m torch.distributed.launch --nproc_per_node=1 run.py --num_workers 8"
#alias exp="python run.py"
alias exp="python -m torch.distributed.launch --nproc_per_node=1 run.py --num_workers 8"
alias diverse_samp="python diverse_samples.py"
alias gaps_data="python gaps_dataset.py"
shopt -s expand_aliases

ds=merged
#num_classes=6
#num_classes=21
#num_classes=16

task='15ss'
gen_par="--task ${task} --dataset ${ds} " #--num_classes ${num_classes}"
lr=1e-3
iter=100
bs=1
path=checkpoints/step/${task}-${ds}

# exp --method FT --name FT --lr ${lr} ${gen_par} --step 0 --debug --batch_size ${bs} #--deeplab 'nonei
ns=5
is=0
inc_par="--ishot ${is} --input_mix novel --val_interval 50 --ckpt_interval 5 --no_pooling"


##  GAPS
#diverse_samp --method FT --name FT --epochs 10 --num_classes 16 --lr ${lr} ${gen_par} --batch_size ${bs} --step 0 
#gaps_data step1
#exp --method FT --name FT --epochs 10 --num_classes 21 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --debug --nshot ${ns} --step_ckpt ${path}/FT_0_newUNET_dynamic.pth 

#diverse_samp --method FT --name FT --epochs 10 --num_classes 21 --lr ${lr} ${gen_par} --batch_size ${bs} --step 1 
#gaps_data step2
exp --method FT --name FT --epochs 10 --num_classes 27 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 2 --debug --nshot ${ns} --step_ckpt ${path}/FT-s5-i0_1_newUNET_dynamic.pth 







# Subspace reg 
#python run.py --step 1 --num_classes 21 --epochs 1 --max_iterations 1 --space-reg 0.2 --space_ft 0.1
# Exp 1 - --space_reg 0.2 --space_ft 0.1
#exp --method FT --name FT --epochs 10 num_classes=21 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --debug --nshot ${ns} --step_ckpt #${path}/FT_0_newUNET_dynamic.pth 

#exp --method FT --name FT --epochs 10 num_classes=27 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 2 --debug --nshot ${ns} --step_ckpt ${path}/FT-s5-i0_1_newUNET_dynamic.pth





# vanilla 
# exp --method FT --name FT --vanila --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --debug --nshot ${ns} --step_ckpt ${path}/FT_0_newUNET_dynamic.pth
# exp --method FT --name FT --vanila --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 2 --debug --nshot ${ns} --step_ckpt ${path}/FT-s5-i0_1_newUNET_dynamic.pth

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
