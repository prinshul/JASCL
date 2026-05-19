import nibabel as nib
import os

import numpy as np

def process_folder(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.nii.gz'):
            filepath = os.path.join(directory, filename)
            
            # Load the NIFTI file and get its data
            img = nib.load(filepath)
            data = img.get_fdata()
            
            # Get the unique values and print them
            unique_values = np.unique(data)
            print(f"File: {filename}")
            print("Unique values:", unique_values)
            print('-' * 50)

# Process each folder
folders = ['./dataset/annotations/exemplar'] #, './dataset/annotations/test', './dataset/annotations/val']

for folder in folders:
    print(f"Processing folder: {folder}")
    print('=' * 60)
    process_folder(folder)
