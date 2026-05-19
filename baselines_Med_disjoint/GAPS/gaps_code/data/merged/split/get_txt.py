import os

def generate_txt_file(directory):
    # Extract the base directory name to use as the filename
    filename = directory.split('/')[-1] + ".txt"
    print(filename)
    with open(filename, 'w') as f:
        # Sort files based on their integer value
        for file in sorted(os.listdir(directory)): #, key=lambda x: int(x.split('.')[0])):
            # Extracting patient number from filenames like '0.nii.gz'
            print(file)
            patient_number = file.split('.')[0]
            f.write(patient_number + '\n')

if __name__ == "__main__":
    run_base = "/hdd2/cil/running_base/GAPS/gaps_code/data/merged/"
    for folder in ["dataset/annotations/exemplar2"]: 
        generate_txt_file(run_base+folder)

    
