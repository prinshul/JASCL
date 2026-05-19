import pickle5 as pkl

# Load the pickle file
with open("./inverse_dict_new_exemplar.pkl", "rb") as f:
    data = pkl.load(f)

# Print type of the loaded object
print(f"Type of the loaded object: {type(data)}")

# If it's a dictionary, print its keys and the type of its values
if isinstance(data, dict):
    print("Keys in the dictionary:")
    # counter = 0
    for key, value in data.items():
        # print(f"Key: {key}, Type of value: {type(value)}")

        # If you want to print the actual values (might be too large)
        value.sort()
        print(f"Key: {key}, Value: {value}")
        print(len(value))
        # counter = max(max(value), counter)
    # print("COUNTER: ", counter)
else:
    # If it's not a dictionary, just print its content
    print(data)
