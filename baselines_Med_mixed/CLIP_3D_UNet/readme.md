# CLIP UNet 3D

[CLIP-Driven Universal Model](https://github.com/ljwztc/CLIP-Driven-Universal-Model)

## How to run :

- Run label_transfer.py for pre-processing dataset for each step. Additional instructions can be find in original repo.
- Download U-Net weights from original repo and place in pretrained_weights/ folder.
- Use run_by_file.py for running baseline. Label transfer, training and testing command is provided in this file. Additionaly, we need to generate text embeddings using CLIP. 
