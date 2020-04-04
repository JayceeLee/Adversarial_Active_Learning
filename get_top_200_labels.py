import os
import pickle
import numpy as np


with open("results/unity_to_trinity_UDA_semi_10/evaluation_trinity_test/confidence_map_UDA.pkl", "rb") as f:
    data = pickle.load(f)

filename = data[0]
score = np.array(data[1])

sorted_index = np.argsort(-score)
images_to_label = []

for i in range(200):
    images_to_label.append(
        os.path.join("/home/yirus/Datasets/Active_Learning/trinity/train/images/",
                     filename[sorted_index[i]])
    )
    print(score[sorted_index[i]])

with open("dataloaders/eye/trinity_top_200.pkl", "wb") as f:
    pickle.dump(images_to_label, f)


