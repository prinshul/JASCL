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
pseudo_label proto_use
'''
    
all_cmds = ['python train_step1.py --eval-type test',
            'python train_step2.py --eval-type train --nshot 10 --pseudo_label 5 --proto_use 25 --batch-size 2',
            'python train_step2.py --eval-type test --nshot 10',
            'python train_step3.py --eval-type train --nshot 10 --pseudo_label 5 --proto_use 25 --batch-size 2',
            'python train_step3.py --eval-type test --nshot 10',
            'python train_step4.py --eval-type train --nshot 10 --pseudo_label 5 --proto_use 25 --batch-size 2',
            'python train_step4.py --eval-type test --nshot 10',
]

print('Training')        
for cmds in all_cmds:
    print("\n\nCurrent command : ",cmds)
    execute_python_file(cmds)
    print("Moving to new command")
