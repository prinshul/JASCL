map_size = [4, 4, 4]
in_chan=1
num_classes=10
base_chan=32
conv_block= 'BasicBlock'
down_scale= [[2,2,2], [2,2,2], [2,2,2], [2,2,2]]
kernel_size= [[3,3,3], [3,3,3], [3,3,3], [3,3,3], [3,3,3]]
chan_num= [64, 128, 256, 320, 256, 128, 64, 32]
norm= 'in'
act= 'relu'
map_size= [4, 4, 4]
conv_num= [2,1,0,0, 0,1,2,2]
trans_num= [0,1,4,6, 4,1,0,0]
num_heads= [1,4,8,10,8,4,1,1]
expansion= 4
fusion_depth= 2
fusion_dim= 320
fusion_heads= 10
attn_drop= 0.
proj_drop= 0.
proj_type= 'depthwise'

training_size=[128, 160, 96]

aux_loss= False
aux_weight= [0.5, 0.5]
curr_step=0