import os

import numpy as np
import glob

import pickle

files = glob.glob("/home/yirus/Datasets/Active_Learning/trinity/train/images/*.png")

print("#data: {}".format(len(files)))
index = np.arange(len(files))
print(files[index[0]])
np.random.shuffle(index)
print(files[index[0]])

data = []
for i in range(200):
    data.append(files[index[i]])

with open("trinity_train_200.pkl", "wb") as f:
    pickle.dump(data, f)


