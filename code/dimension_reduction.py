import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

files = {"arousal":"arousal_features_200.csv",
         "dominance":"dominance_features_200.csv",
         "liking":"liking_features_200.csv",
         "valence":"valence_features_200.csv"}

data = {}

print("------------------------------------------------------------")
print("Reading files...")
for attribute in list(files.keys()):
    print(f"Reading {attribute}...")
    data[attribute] = pd.read_csv(f"features/{files[attribute]}")

tsne = TSNE(n_components=2, random_state=42)

data_reduced = {}

print("------------------------------------------------------------")
print("Starting dimensional reduction")

for feature in (list(data.keys())):
    print(f"Using t-sne for {feature}...")
    data_reduced[feature] = tsne.fit_transform(data[feature])

print("Finished dimension reduction")

print("------------------------------------------------------------")
print("Saving reduced data")

try:
    os.mkdir("reduced_data")
except:
    print("The file reduces_data already exist... skipping it")

for key in list(data_reduced.keys()):
    path = f"reduced_data/{key}.csv"
    print(f"Saving {key} into ")
    data_frame = pd.DataFrame(data_reduced[key])
    data_frame.to_csv(path)

print("------------------------------------------------------------")
print("Plotting reduced data...")

t = np.arange(0, data_reduced[feature].shape[0])
plt.figure(figsize=(16,9))
for position, feature in enumerate(list(data_reduced.keys())):
    plt.subplot(len(list(data_reduced.keys())), 1, position + 1)
    plt.title(feature, fontsize=14)
    plt.scatter(data_reduced[feature][:, 0], data_reduced[feature][:, 1], c=t, cmap=plt.cm.hot)
    plt.grid(True)

plt.savefig("reduced_data/tsne.png")
plt.show()