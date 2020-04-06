import os
import numpy as np
import pickle
import glob

from PIL import Image

import matplotlib.pyplot as plt

from dataloaders.eye.eyeDataset import JointDataset, OpenEDSDataset_withLabels
from dataloaders.eye.photometric_transform import PhotometricTransform, photometric_transform_config
from utils import dual_transforms

# import torch
#
# transforms_shape_target = dual_transforms.Compose(
#         [
#             dual_transforms.CenterCrop((400, 400)),
#             dual_transforms.Scale(224),
#         ]
#     )
# photo_transformer = PhotometricTransform(photometric_transform_config)
#
# traindata = OpenEDSDataset_withLabels(
#     root=os.path.join("/home/yirus/Datasets/Active_Learning/trinity", "train"),
#     image_size=[224,224],
#     data_to_train="",
#     shape_transforms=transforms_shape_target,
#     photo_transforms=photo_transformer,
#     train_bool=False,
# )

images_list = glob.glob("/home/yirus/Datasets/Active_Learning/trinity/train/images/*.png")

uscore = np.load("results/uncertainty_scores.npy")
with (open("results/unity_to_trinity_UDA_semi_10/evaluation_trinity_test/confidence_map_UDA.pkl", "rb")) as f:
    adv_scores = pickle.load(f)
index = np.argsort(adv_scores[1])[::-1]
ascore = np.empty((2, 8916))
ascore[0] = index
ascore[1] = adv_scores[1][index]

distance = np.load("results/distance_maps_UDA.npy")
distance_ascore = np.load("results/distance_maps_UDA_ascore.npy")
distance_original_order = np.load("results/distance_maps_UDA_original_order.npy")

# index_ascore = np.argsort(ascore[1])[::-1]
# ascore = ascore[1][index_ascore]
# images_list_ascore = []
# for i in index_ascore:
#     images_list_ascore.append(images_list[i])
# ascore = ascore[1][index_ascore]

index_uncertain = np.argsort(uscore[1])[::-1]

selected_images = []
last_next_idx = -1
last_cur_idx = -1
image_index = 1

# ########## adversarial selection ##########
# fig = plt.figure(figsize=(14,8))
# similar_appearance_cnt = 0
# last_cur_idx = -1
# for uidx in range(8915):
#     if len(selected_images) >= 200:
#         break
#
#     similar_appearance_cnt = 0
#
#     curr_idx = uidx
#     if last_next_idx != -1:
#         curr_idx = last_next_idx+1
#     elif last_cur_idx != -1:
#         curr_idx = last_cur_idx+1
#
#     print("current uncertain: {}".format(ascore[1][curr_idx]))
#
#     if images_list[int(ascore[1][curr_idx])] in selected_images:
#         continue
#
#     for next_idx in range(curr_idx+1, 8916):
#         curr_dist = distance_ascore[curr_idx][next_idx]
#         if curr_dist > 0.535:
#             similar_appearance_cnt += 1
#             continue
#         if images_list[int(ascore[1][next_idx])] in selected_images:
#             continue
#         selected_images.append(images_list[int(ascore[0][curr_idx])])
#         selected_images.append(images_list[int(ascore[0][next_idx])])
#
#         print("{} ---> {}".format(ascore[1][curr_idx], ascore[1][next_idx]))
#
#         if image_index == 36:
#             c = 1
#
#         # ax = fig.add_subplot(6, 6, image_index)
#         # with Image.open(images_list[int(ascore[0][curr_idx])]) as f:
#         #     curr_img = f.convert("L").copy()
#         # with Image.open(images_list[int(ascore[0][next_idx])]) as f:
#         #     next_img = f.convert("L").copy()
#         # ax.imshow(curr_img, cmap="gray")
#         # ax.set_title("{:.5f}, {}".format(ascore[1][curr_idx], curr_idx))
#         # ax.set_xticks([])
#         # ax.set_yticks([])
#         # image_index += 1
#         # ax = fig.add_subplot(6, 6, image_index)
#         # ax.imshow(next_img, cmap="gray")
#         # ax.set_title("{:.5f}, {}".format(ascore[1][next_idx], next_idx))
#         # ax.set_xticks([])
#         # ax.set_yticks([])
#         image_index += 1
#
#         last_next_idx = next_idx
#         last_cur_idx = -1
#         break
#
#     if similar_appearance_cnt==(8915-curr_idx):
#         last_cur_idx = curr_idx
#         last_next_idx = -1
#
# plt.tight_layout()
# plt.show()
#
# for f in selected_images:
#     print(f)
#
# print(len(selected_images))
#
# with open("results/heuristic_200_adversarial.pkl", "wb") as f:
#     pickle.dump(selected_images, f)
# ########## adversarial selection ##########

# fig = plt.figure(figsize=(14,8))
# similar_appearance_cnt = 0
# last_cur_idx = -1
# for uidx in range(8915):
#     if len(selected_images) > 200:
#         break
#
#     similar_appearance_cnt = 0
#
#     curr_idx = index_ascore[uidx]
#     # if last_next_idx != -1:
#     #     uidx = last_next_idx+1
#     #     curr_idx = index_ascore[uidx]
#
#     if last_next_idx != -1:
#         uidx = last_next_idx + 1
#         curr_idx = index_ascore[uidx]
#     elif last_cur_idx != -1:
#         curr_idx = last_cur_idx+1
#
#     print("current uncertain: {}, {}".format(ascore[1][curr_idx], images_list[curr_idx]))
#
#     if images_list[curr_idx] in selected_images:
#         continue
#
#     for ux_idx in range(uidx+1, 8916):
#         next_idx = index_ascore[ux_idx]
#         curr_dist = distance_original_order[curr_idx][next_idx]
#         if curr_dist > 0.5:
#             continue
#         if images_list[next_idx] in selected_images:
#             continue
#         selected_images.append(images_list[curr_idx])
#         selected_images.append(images_list[next_idx])
#
#         print("{} ---> {}, {}".format(ascore[1][curr_idx], ascore[1][next_idx], images_list[next_idx]))
#
#         # fig = plt.figure()
#         ax = fig.add_subplot(6, 6, image_index)
#         with Image.open(images_list[curr_idx]) as f:
#             curr_img = f.convert("L").copy()
#         with Image.open(images_list[next_idx]) as f:
#             next_img = f.convert("L").copy()
#         ax.imshow(curr_img, cmap="gray")
#         ax.set_title("{:.5f}, {}, {}".format(ascore[1][curr_idx], curr_idx, uidx))
#         ax.set_xticks([])
#         ax.set_yticks([])
#         image_index += 1
#         ax = fig.add_subplot(6, 6, image_index)
#         ax.imshow(next_img, cmap="gray")
#         ax.set_title("{:.5f}, {}, {}".format(ascore[1][next_idx], next_idx, ux_idx))
#         ax.set_xticks([])
#         ax.set_yticks([])
#         image_index += 1
#
#         last_next_idx = ux_idx
#         last_cur_idx = next_idx
#         break
#
# plt.tight_layout()
# plt.show()

fig = plt.figure(figsize=(14,8))
similar_appearance_cnt = 0
last_cur_idx = -1
for uidx in range(8915):
    if len(selected_images) >= 200:
        break

    similar_appearance_cnt = 0

    curr_idx = uidx
    if last_next_idx != -1:
        curr_idx = last_next_idx+1
    elif last_cur_idx != -1:
        curr_idx = last_cur_idx+1

    print("current uncertain: {}".format(uscore[1][curr_idx]))

    if images_list[curr_idx] in selected_images:
        continue

    for next_idx in range(curr_idx+1, 8916):
        # next_idx = index_uncertain[nidx]
        # print("next uncertain: {}".format(uscore[1][next_idx]))
        curr_dist = distance[curr_idx][next_idx]
        if curr_dist > 0.55:
            similar_appearance_cnt += 1
            continue
        if images_list[next_idx] in selected_images:
            continue
        selected_images.append(images_list[int(uscore[0][curr_idx])])
        selected_images.append(images_list[int(uscore[0][next_idx])])

        print("{} ---> {}".format(uscore[1][curr_idx], uscore[1][next_idx]))

        # ax = fig.add_subplot(6, 6, image_index)
        # with Image.open(images_list[int(uscore[0][curr_idx])]) as f:
        #     curr_img = f.convert("L").copy()
        # with Image.open(images_list[int(uscore[0][next_idx])]) as f:
        #     next_img = f.convert("L").copy()
        # ax.imshow(curr_img, cmap="gray")
        # ax.set_title("{:.5f}, {}".format(uscore[1][curr_idx], curr_idx))
        # ax.set_xticks([])
        # ax.set_yticks([])
        # image_index += 1
        # ax = fig.add_subplot(6, 6, image_index)
        # ax.imshow(next_img, cmap="gray")
        # ax.set_title("{:.5f}, {}".format(uscore[1][next_idx], next_idx))
        # ax.set_xticks([])
        # ax.set_yticks([])
        image_index += 1

        last_next_idx = next_idx
        last_cur_idx = -1
        break

    if similar_appearance_cnt==(8915-curr_idx):
        last_cur_idx = curr_idx
        last_next_idx = -1

# plt.tight_layout()
# plt.show()

# for f in selected_images:
#     print(f)

print(len(selected_images))

with open("results/heuristic_200_uncertain.pkl", "wb") as f:
    pickle.dump(selected_images, f)


