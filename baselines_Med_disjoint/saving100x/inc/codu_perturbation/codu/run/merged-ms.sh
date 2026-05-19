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

#exp --method FT --name FT --lr ${lr} ${gen_par} --step 0 --debug --batch_size ${bs} --test #--deeplab 'nonei
ns=5
is=0
inc_par="--ishot ${is} --input_mix novel --val_interval 50 --ckpt_interval 5 " #--no_pooling"

## Do in base not here Step 0 test 
#exp --lr ${lr} ${gen_par} --test --num_classes 16 --step 0 --debug --batch_size ${bs} --test


# inv grad 
exp --load_weight_strategy --vanila --num_classes 21 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --step_ckpt ${path}/FT_0_newUNET_dynamic.pth 
#exp --load_weight_strategy --vanila --num_classes 27 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 2 --step_ckpt ./checkpoints/step/15ss-merged/Experiment-s5-i0_1_newUNET_dynamic.pth --test
#exp --load_weight_strategy --vanila --num_classes 31 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 3 --step_ckpt ./checkpoints/step/15ss-merged/Experiment-s5-i0_2_newUNET_dynamic.pth --test
#exp --load_weight_strategy --vanila --num_classes 34 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 4 --step_ckpt ./checkpoints/step/15ss-merged/Experiment-s5-i0_3_newUNET_dynamic.pth --test
#exp --load_weight_strategy --vanila --num_classes 38 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 5 --step_ckpt ./checkpoints/step/15ss-merged/Experiment-s5-i0_4_newUNET_dynamic.pth --test









 
#python saved_proto.py --num_classes 16 --lr ${lr} ${gen_par} --vanila --step 0 --debug --batch_size ${bs} --step_ckpt ${path}/Experiment_0_newUNET_dynamic.pth
#python saved_proto.py --num_classes 21 --lr ${lr} ${gen_par} --vanila --step 1 --debug --batch_size ${bs} --step_ckpt ${path}/Experiment-s5-i0_1_newUNET_dynamic.pth
#python saved_proto.py --num_classes 27 --lr ${lr} ${gen_par} --vanila --step 2 --debug --batch_size ${bs} --step_ckpt ${path}/Experiment-s5-i0_2_newUNET_dynamic.pth
#exp --load_weight_strategy --vanila --num_classes 31 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 3 --step_ckpt ${path}/Experiment-s5-i0_2_newUNET_dynamic.pth
#exp --load_weight_strategy --vanila --num_classes 34 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 4 --step_ckpt ${path}/Experiment-s5-i0_3_newUNET_dynamic.pth
#exp --load_weight_strategy --vanila --num_classes 38 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 5 --step_ckpt ${path}/Experiment-s5-i0_4_newUNET_dynamic.pth

#python saved_proto.py --num_classes 31 --lr ${lr} ${gen_par} --vanila --step 3 --debug --batch_size ${bs} --step_ckpt ${path}/Experiment-s5-i0_3_newUNET_dynamic.pth
#exp --load_weight_strategy --test --vanila --num_classes 34 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 4 --step_ckpt ${path}/Experiment-s5-i0_4_newUNET_dynamic.pth
#exp --load_weight_strategy --vanila --num_classes 38 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 5 --step_ckpt ${path}/Experiment-s5-i0_4_newUNET_dynamic.pth

#exp --load_weight_strategy --vanila --num_classes 38 --epochs 10 ${gen_par} ${inc_par} --batch_size ${bs} --step 4 --step_ckpt ${path}/Experiment-s5-i0_3_newUNET_dynamic.pth



#exp --lr ${lr} ${gen_par} --test --num_classes 16 --step 0 --debug --batch_size ${bs}
#exp --method FT --name FT --vanila --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --debug --nshot ${ns} --step_ckpt ${path}/Experiment_0_newUNET_dynamic.pth
#exp --method FT --name FT --vanila --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 2 --debug --nshot ${ns} --step_ckpt ${path}/Experiment-s5-i0_1_newUNET_dynamic.pth


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
