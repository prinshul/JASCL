import os

def execute_python_file(cmd):
   try:
      os.system(cmd)
   except FileNotFoundError:
      print(f"Error: The file '{file_path}' does not exist.")

'''
'python train_step1.py --eval-type train',
'python train_step1.py --eval-type test'
'python train_step2.py --eval-type train --nshot 10',
'python train_step2.py --eval-type test --nshot 10',
"python main.py fit -c configs/step1.yaml --root results/step1 --model.network.encoder_name samvit_base_patch16.sa1b --model.network.ckpt_path results/step0/lightning_logs/version_0/checkpoints/epoch=4-step=1336.ckpt --model.freeze_encoder True"
'''
    
all_cmds = [#'python train_step1.py --eval-type test',
            'python train_step2.py --eval-type train --nshot 10',
            'python train_step2.py --eval-type test --nshot 10',
            'python train_step3.py --eval-type train --nshot 10',
            'python train_step3.py --eval-type test --nshot 10'
            ]

print('Training')        
for cmds in all_cmds:
    print("\n\nCurrent command : ",cmds)
    execute_python_file(cmds)
    print("Moving to new command")
