import os

cwd = os.getcwd()

results_dir = cwd+"/results/"

del_files = []

for files in os.listdir(results_dir):
    if "medformer" not in files:
        os.remove(results_dir+files)
        
print(del_files)