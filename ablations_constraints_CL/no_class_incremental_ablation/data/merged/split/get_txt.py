import os

def generate_txt_file(directory):
    # Extract the base directory name to use as the filename
    filename = directory.split('/')[-1] + ".txt"
    
    with open(filename, 'w') as f:
        # Sort files based on their integer value
        for file in sorted(os.listdir(directory), key=lambda x: int(x.split('.')[0])):
            # Extracting patient number from filenames like '0.nii.gz'
            patient_number = file.split('.')[0]
            f.write(patient_number + '\n')

if __name__ == "__main__":
    for folder in ["../dataset/annotations/train", "../dataset/annotations/test", "../dataset/annotations/val"]:
        generate_txt_file(folder)
