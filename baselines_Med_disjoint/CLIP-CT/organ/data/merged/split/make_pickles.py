import os
import pickle
import nibabel as nib
import numpy as np

def create_inverse_dict():
    # Step 1: Read the label.txt file to get the class mappings
    class_map = {}
    with open("label.txt", "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            class_map[int(parts[0])] = parts[1]

    inverse_dict = {key: [] for key in class_map.keys()}
    print(inverse_dict)

    # Step 2: Traverse the /train, /test, and /val directories
    for folder in ["../dataset/annotations/train", "../dataset/annotations/test", "../dataset/annotations/val"]:
        for filename in os.listdir(folder):
            print(f"{folder}/{filename}")    
            # Extract the base name without the .nii.gz extension
            base_name = os.path.splitext(os.path.splitext(filename)[0])[0]
            
            # Load the file using nibabel
            img = nib.load(os.path.join(folder, filename))
            data = img.get_fdata()
            
            # Get the unique values (classes) for the patient
            unique_classes = np.unique(data)
            
            # Add the base_name to the corresponding classes in the inverse_dict
            for patient_class in unique_classes:
                if patient_class in inverse_dict:  # Check if the class exists in the class_map
                    inverse_dict[patient_class].append(base_name)

    # Step 3: Save the dictionary to a .pkl file
    with open("inverse_dict.pkl", "wb") as f:
        pickle.dump(inverse_dict, f)

    # Print the dictionary for verification
    for key, value in inverse_dict.items():
        # Sort the values (filenames) by converting them to integers for the sorting operation
        sorted_values = sorted(value, key=lambda x: int(x))
        print(f"Key: {key}, Value: {sorted_values}")

if __name__ == "__main__":
    create_inverse_dict()
