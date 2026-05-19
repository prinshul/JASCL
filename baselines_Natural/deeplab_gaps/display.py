import pickle

data_path = 'save/step1/'

with open(data_path+"val_sort_proto.pkl", "rb") as f:
    files = pickle.load(f)

#print(files)
print(type(files))

i=0
for af in range(0,len(files)):
    print(af, files[af])
    print(files[af][0][0])
    i+=1
    if i==10:
        break
#print(files)
