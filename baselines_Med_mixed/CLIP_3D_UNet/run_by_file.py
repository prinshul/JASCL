import os

def execute_python_file(cmd):
   try:
      os.system(cmd)
   except FileNotFoundError:
      print(f"Error: The file '{file_path}' does not exist.")

'''
python test.py --data_txt_path step0_test 
python pretrained_weights/clip_embedding.py
python train.py --data_txt_path step1_train --resume pretrained_weights/unet.pth
python train.py --data_txt_path step1_test --resume trained_checkpoints/step1_train/epoch_200.pth
python pretrained_weights/clip_embedding.py
python train.py --data_txt_path step2_train --resume trained_checkpoints/step1_train/epoch_200.pth
"python label_transfer.py",
"python test.py --data_txt_path step2_test --resume trained_checkpoints/step2_train/epoch_200.pth",
"python test.py --data_txt_path step3_test --resume trained_checkpoints/step2_train/epoch_200.pth",
"python label_transfer.py",
"python pretrained_weights/clip_embedding.py", 
"python train.py --data_txt_path step4_train --resume trained_checkpoints/step2_train/epoch_200.pth",                
'''
  
all_cmds = ["python test.py --data_txt_path step4_test --resume trained_checkpoints/step4_train/epoch_200.pth"]
 
print('Training')        
for cmds in all_cmds:
    print("\n\nCurrent command : ",cmds)
    execute_python_file(cmds)
    print("Moving to new command")
