import os
import pickle
import nibabel as nib
import numpy as np


string_base = "exemplar2"
def create_inverse_dict_train():
    # Step 1: Read the label.txt file to get the class mappings
    class_map = {}
    with open("label.txt", "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            class_map[int(parts[0])] = parts[1]

    inverse_dict = {key: [] for key in class_map.keys()}
    print(inverse_dict)

    # Create a list of patient numbers from val.txt
    with open(string_base+".txt", "r") as f:
        patient_numbers = [line.strip() for line in f]

    # Step 2: Traverse only the /val directory
    folder = "//hdd2/cil/running_base/fscil_SS/fscilSS/data/merged/dataset/annotations/"+string_base
    for filename in os.listdir(folder):
        print(f"{folder}/{filename}") 
        # Extracting patient number from filenames like '0.nii.gz'
        patient_number = filename.split('.')[0]

        # Get the index of the patient number from the list
        patient_index = patient_numbers.index(patient_number)

        # Load the file using nibabel
        img = nib.load(os.path.join(folder, filename))
        data = img.get_fdata()
        
        # Get the unique values (classes) for the patient
        unique_classes = np.unique(data)
        
        # Add the patient index to the corresponding classes in the inverse_dict
        for patient_class in unique_classes:
            if patient_class in inverse_dict:  # Check if the class exists in the class_map
                inverse_dict[patient_class].append(patient_index)

    # Step 3: Save the dictionary to a .pkl file named inverse_dict_train.pkl
    with open("inverse_dict_"+string_base+".pkl", "wb") as f:
        pickle.dump(inverse_dict, f)
    with open("inverse_dict_new_"+string_base+".pkl", "wb") as f:
        pickle.dump(inverse_dict, f)

    # Print the dictionary for verification
    for key, value in inverse_dict.items():
        value.sort()
        print(f"Key: {key}, Value: {value}")

if __name__ == "__main__":
    create_inverse_dict_train()

