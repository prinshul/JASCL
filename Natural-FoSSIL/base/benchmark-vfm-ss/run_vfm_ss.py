## Step 0 test

Train step calculation : n files total 
No of steps for 1 epoch = n/16
For X epochs, tot_steps_train = x * (n/16) steps 
max_steps = tot_steps_train

Val step calculation : n files total 
val_step_interval/max_steps

Step 0 - Train steps 15250(50 epochs), val steps 3050
nohup python main.py fit -c configs/step0.yaml --root results/step0 --model.network.encoder_name samvit_base_patch16.sa1b > sam_step0.log &
/results/step0/lightning_logs/version_0


###########################################################################


Step 1 - Train steps 200(50 epochs), val steps 50
nohup python main.py fit -c configs/step1.yaml --root results/step1 --model.network.encoder_name samvit_base_patch16.sa1b --model.network.ckpt_path results/step0/lightning_logs/version_0/checkpoints/epoch=4-step=1336.ckpt > sam_step1.log &
results/step1/lightning_logs/version_0

nohup python main.py fit -c configs/step1.yaml --root results/step1 --model.network.encoder_name samvit_base_patch16.sa1b --model.network.ckpt_path results/step0/lightning_logs/version_0/checkpoints/epoch=4-step=1336.ckpt --model.freeze_encoder True > sam_step1_freeze.log &
results/step1/lightning_logs/version_1
#nohup python main.py fit -c configs/step0.yaml --root results/step1 --model.network.encoder_name vit_base_patch16_siglip_512.webli > clip_step0.log &

Stochastic classifier
nohup python main.py fit -c configs/step1.yaml --root results/step1 --model.network.encoder_name samvit_base_patch16.sa1b --model.network.ckpt_path results/step0/lightning_logs/version_0/checkpoints/epoch=4-step=1336.ckpt --model.freeze_encoder True > sam_step1_stoch.log &

Proto only
nohup python main.py fit -c configs/step1.yaml --root results/step1 --model.network.encoder_name samvit_base_patch16.sa1b --model.network.ckpt_path results/step0/lightning_logs/version_0/checkpoints/epoch=4-step=1336.ckpt --model.freeze_encoder True > sam_step1_proto.log &


nohup python main.py fit -c configs/step1.yaml --root results/step1 --model.network.encoder_name samvit_base_patch16.sa1b --model.network.ckpt_path results/step0/lightning_logs/version_0/checkpoints/epoch=4-step=1336.ckpt --model.freeze_encoder True > sam_step1_no_proto.log &


nohup python main.py fit -c configs/step1.yaml --root results/step1 --model.network.encoder_name samvit_base_patch16.sa1b --model.network.ckpt_path results/step0/lightning_logs/version_0/checkpoints/epoch=4-step=1336.ckpt --model.freeze_encoder True > sam_step1_0_2.log &

nohup python main.py fit -c configs/step1.yaml --root results/step1 --model.network.encoder_name samvit_base_patch16.sa1b --model.network.ckpt_path results/step0/lightning_logs/version_0/checkpoints/epoch=4-step=1336.ckpt --model.freeze_encoder True > sam_step1_0_2_20_steps.log &

sam_step1_0_2_20_steps.log
step1_step_20.log -  version 4 - Step 18, Wall time 1746884346.913045: 0.326173335313797
###########################################################################



Step 2 - Train steps 200(50 epochs), val steps 50
nohup python main.py fit -c configs/step2.yaml --root results/step2 --model.network.encoder_name samvit_base_patch16.sa1b --model.network.ckpt_path results/step1/lightning_logs/version_1/checkpoints/epoch=9-step=40.ckpt > sam_step1.log &
#results/step2/lightning_logs/version_0

nohup python main.py fit -c configs/step2.yaml --root results/step2 --model.network.encoder_name samvit_base_patch16.sa1b --model.network.ckpt_path results/step1/lightning_logs/version_0/checkpoints/epoch=9-step=40.ckpt --model.freeze_encoder True > sam_step2_freeze.log &
results/step2/lightning_logs/version_0

Stochastic classifier
nohup python main.py fit -c configs/step2.yaml --root results/step2 --model.network.encoder_name samvit_base_patch16.sa1b --model.network.ckpt_path results/step1/lightning_logs/version_22/checkpoints/epoch=9-step=40.ckpt --model.freeze_encoder True > sam_step2_stoch_new.log &



