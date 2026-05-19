#/bin/bash

# export CUDA_VISIBLE_DEVICES=$1
port=$(python get_free_port.py)
echo ${port}
#alias exp="python -m torch.distributed.launch --master_port ${port} --nproc_per_node=1 run.py --num_workers 4"
alias exp="python run.py --num_workers 4"
shopt -s expand_aliases

ds=merged
# num_classes=16
#num_classes=21
# num_classes=27
#num_classes=11,19,25,29,36

task='15ss'
gen_par="--task ${task} --dataset ${ds}  --n_gpu 2" # --num_classes ${num_classes}"
lr=1e-3
iter=100
bs=2
path=checkpoints/step/${task}-${ds}

#exp --lr ${lr} ${gen_par} --test --num_classes 11 --step 0 --debug --batch_size ${bs} --test #--deeplab 'nonei
ns=5
is=0
inc_par="--ishot ${is} --val_interval 50 --ckpt_interval 5 --no_pooling"



## medformer inv grad
exp --num_classes 19 --vanila --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 1 --input_mix novel --debug --nshot ${ns} --step_ckpt ${path}/Experiment_0_medformer_dynamic.pth 
#exp --num_classes 25 --vanila --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 2 --input_mix both --debug --nshot ${ns} --step_ckpt ${path}/Experiment-s5-i0_1_medformer_dynamic.pth
#exp --num_classes 29 --vanila --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 3 --input_mix novel --debug --nshot ${ns} --step_ckpt ${path}/Experiment-s5-i0_2_medformer_dynamic.pth
#exp --num_classes 36 --vanila --load_weight_strategy --epochs 10 --lr ${lr} ${gen_par} ${inc_par} --batch_size ${bs} --step 4 --input_mix novel --debug --nshot ${ns} --step_ckpt ${path}/Experiment-s5-i0_3_medformer_dynamic.pth

