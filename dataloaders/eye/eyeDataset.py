import os
import sys

import numpy as np
import glob
import pickle
from PIL import Image
import cv2
import random
import matplotlib.pyplot as plt

# from .photometric_transform import PhotometricTransform, photometric_transform_config
# photo_transformer = PhotometricTransform(photometric_transform_config)

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../" % file_path)

from utils.utils import rescale_image, convt_array_to_PIL

NUM_CLASSES = 4

import torch
import torch.utils.data

seed = 1
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

class OpenEDSDataset_withRegionLabels(torch.utils.data.Dataset):
    """
    OpenEDS dataset
    Note:for segmentation, target is numpy array of size (th, tw) with pixel
    values representing the label index
    :param root: Essential, path to data_root
    :param data_to_train: .pkl saving select images from last round
    :param ph_transform: list of transforms of photometric augmentation
    :param train_bool: Boolean true if the data loader is for training dataset; to generate
    class prob and mean image
    """

    def __init__(self,
                 root,
                 image_size,
                 shape_transforms=None,
                 photo_transforms=None,
                 train_bool=False,
        ):
        self.root = root
        self.shape_transforms = shape_transforms
        self.photo_transforms = photo_transforms
        self.train_bool = train_bool
        self.image_size = image_size

        self.train_data_list = glob.glob(self.root+"/images/*.png")

        print("OpenEDS: loading {} images ........".format(len(self.train_data_list)))

        self.all_images = np.empty((len(self.train_data_list), self.image_size[0], self.image_size[1]), dtype=np.float32)
        self.all_labels = np.empty((len(self.train_data_list), self.image_size[0], self.image_size[1]))

        # fig = plt.figure()
        for idx in range(len(self.train_data_list)):
            with Image.open(self.train_data_list[idx]) as f:
                im = f.convert("L").copy()
            lb_filename = self.train_data_list[idx].replace("train_pseudo/", "train/").replace("/images/", "/masks/")
            lb_filename = lb_filename.replace(".png", ".npy")
            lb = np.load(lb_filename)
            lb_mask_filename = self.train_data_list[idx].replace("/images/", "/masks/").replace(".png", ".npy")
            lb_mask = np.load(lb_mask_filename)

            if self.shape_transforms is not None:
                lb = Image.fromarray(lb)
                im, lb = self.shape_transforms(im, lb)
            if self.photo_transforms is not None:
                im = rescale_image(np.array(im))
                im = self.photo_transforms(im)
                im = im.astype(np.float32)

            lb = np.array(lb)
            lb[lb_mask] = 4

            self.all_images[idx,:] = np.array(im).astype(np.float16)
            self.all_labels[idx,:] = lb

        if self.train_bool:
            self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.train_data_list)

    def __getitem__(self, index):
        im = self.all_images[index]
        lb = self.all_labels[index]
        im = im[np.newaxis, :, :] / 255.0
        return np.concatenate((im, im, im), axis=0), lb

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))
        sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, label = self.__getitem__(i)
            if label is not -1:
                for j in range(NUM_CLASSES):
                    counts[j] += np.sum(label == j)
        return counts

    def __compute_image_mean(self):
        sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, target = self.__getitem__(i)
            if i == 0:
                img_mean = img
            else:
                img_mean += img
        return 1.0 * img_mean / self.__len__()

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / np.sum(values)
        return torch.Tensor(p_values)


class OpenEDSDataset_withLabels(torch.utils.data.Dataset):
    """
    OpenEDS dataset
    Note:for segmentation, target is numpy array of size (th, tw) with pixel
    values representing the label index
    :param root: Essential, path to data_root
    :param data_to_train: .pkl saving select images from last round
    :param ph_transform: list of transforms of photometric augmentation
    :param train_bool: Boolean true if the data loader is for training dataset; to generate
    class prob and mean image
    """

    def __init__(self,
                 root,
                 image_size,
                 data_to_train,
                 shape_transforms=None,
                 photo_transforms=None,
                 train_bool=False,
        ):
        self.root = root
        self.shape_transforms = shape_transforms
        self.photo_transforms = photo_transforms
        self.train_bool = train_bool
        self.image_size = image_size
        self.data_to_train = data_to_train

        if self.data_to_train != '':
            with open(self.data_to_train, "rb") as f:
                self.train_data_list = pickle.load(f)
        else:
            self.train_data_list = glob.glob(self.root+"/images/*.png")

        print("OpenEDS: loading {} images ........".format(len(self.train_data_list)))

        self.all_images = np.empty((len(self.train_data_list), self.image_size[0], self.image_size[1]), dtype=np.float32)
        self.all_labels = np.empty((len(self.train_data_list), self.image_size[0], self.image_size[1]))

        # fig = plt.figure()
        for idx in range(len(self.train_data_list)):
            with Image.open(self.train_data_list[idx]) as f:
                im = f.convert("L").copy()
            lb_filename = self.train_data_list[idx].replace("/images/", "/masks/").replace(".png", ".npy")
            lb = np.load(lb_filename)

            # ### opencv read images ###
            # im = cv2.imread(self.train_data_list[idx], cv2.IMREAD_GRAYSCALE)
            # lb_filename = self.train_data_list[idx].replace("/images/", "/masks/").replace(".png", ".npy")
            # lb = np.load(lb_filename)
            # ## pad into 400x400 ##
            # # im = np.concatenate(
            # #     (np.zeros((75, 400), dtype=np.unit8),
            # #      np.array(im),
            # #      np.zeros((75, 400), dtype=np.unit8)), axis=0)
            # # lb = np.concatenate(
            # #     (np.zeros((75, 400), dtype=np.unit8),
            # #      np.array(lb),
            # #      np.zeros((75, 400), dtype=np.unit8)), axis=0)
            # im = cv2.copyMakeBorder(np.array(im), 75, 75, 0, 0, cv2.BORDER_CONSTANT)
            # lb = cv2.copyMakeBorder(lb, 75, 75, 0, 0, cv2.BORDER_CONSTANT)
            # ## pad into 400x400 ##
            # im = rescale_image(im)
            # im = convt_array_to_PIL(im)
            # ### opencv read images ###

            if self.shape_transforms is not None:
                lb = Image.fromarray(lb)
                im, lb = self.shape_transforms(im, lb)
            if self.photo_transforms is not None:
                im = rescale_image(np.array(im))
                im = self.photo_transforms(im)
                im = im.astype(np.float32)

                # ax = fig.add_subplot(121)
                # ax.imshow(im, cmap="gray")
                # ax = fig.add_subplot(122)
                # ax.imshow(im_ph, cmap="gray")
                # plt.show()

            self.all_images[idx,:] = np.array(im).astype(np.float16)
            self.all_labels[idx,:] = np.array(lb)

            # if idx >= 100:
            #     break

        if self.train_bool:
            self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.train_data_list)

    def __getitem__(self, index):
        im = self.all_images[index]
        lb = self.all_labels[index]
        im = im[np.newaxis, :, :] / 255.0
        return np.concatenate((im, im, im), axis=0), lb
        # return im, lb
        # if self.transforms:
        #     im, lb = self.apply_transform(
        #         im,
        #         lb,
        #     )
        #     return im, np.asarray(lb)
        #
        # return np.expand_dims(im, axis=0), np.asarray(lb)

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))
        sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, label = self.__getitem__(i)
            if label is not -1:
                for j in range(NUM_CLASSES):
                    counts[j] += np.sum(label == j)
        return counts

    def __compute_image_mean(self):
        sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, target = self.__getitem__(i)
            if i == 0:
                img_mean = img
            else:
                img_mean += img
        return 1.0 * img_mean / self.__len__()

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / np.sum(values)
        return torch.Tensor(p_values)

class OpenEDSDataset_withoutLabels(torch.utils.data.Dataset):
    """
    OpenEDS dataset
    Note:for segmentation, target is numpy array of size (th, tw) with pixel
    values representing the label index
    :param root: Essential, path to data_root
    :param ph_transform: list of transforms of photometric augmentation
    """

    def __init__(self,
                 root,
                 image_size,
                 shape_transforms=None,
                 photo_transforms=None,
        ):
        self.root = root
        self.shape_transforms = shape_transforms
        self.photo_transforms = photo_transforms
        self.image_size = image_size

        self.data_to_train = glob.glob(self.root+"/images/*.png")

        print("OpenEDS: Loading {} images ........".format(len(self.data_to_train)))

        self.all_images = np.empty(
            (len(self.data_to_train), self.image_size[0], self.image_size[1]),
            dtype=np.float32
        )

        # fig = plt.figure()
        for idx in range(len(self.data_to_train)):
            with Image.open(self.data_to_train[idx]) as f:
                im = f.convert("L").copy()

            ### opencv read images ###
            # im = cv2.imread(self.data_to_train[idx], cv2.IMREAD_GRAYSCALE)
            # ## pad into 400x400 ##
            # # im = np.concatenate(
            # #     (np.zeros((75, 400), dtype=np.unit8),
            # #      np.array(im),
            # #      np.zeros((75, 400), dtype=np.unit8)), axis=0)
            # im = cv2.copyMakeBorder(im, 75, 75, 0, 0, cv2.BORDER_CONSTANT)
            # #
            # im = convt_array_to_PIL(im)
            # ## pad into 400x400 ##
            ### opencv read images ###

            if self.shape_transforms is not None:
                im, _ = self.shape_transforms(im, None)

            if self.photo_transforms is not None:
                im = rescale_image(np.array(im))
                im = self.photo_transforms(im)
                im = im.astype(np.float32)

                # ax = fig.add_subplot(121)
                # ax.imshow(im, cmap="gray")
                # ax = fig.add_subplot(122)
                # ax.imshow(im_ph, cmap="gray")
                # plt.show()

            self.all_images[idx, :] = np.array(im).astype(np.float16)

            # if idx >= 100:
            #     break

    def __len__(self):
        return len(self.data_to_train)

    def __getitem__(self, index):
        im = self.all_images[index]
        im = im[np.newaxis, :, :] / 255.0
        return np.concatenate((im, im, im), axis=0)
        # return im

class EverestDataset(torch.utils.data.Dataset):
    """
    Everest dataset
    """
    def __init__(self,
                 root,
                 image_size,
                 shape_transforms=None,
                 photo_transforms=None,
                 train_bool=False,
        ):
        self.root = root
        self.image_size = image_size
        self.shape_transforms = shape_transforms
        self.photo_transforms = photo_transforms
        self.train_bool = train_bool

        self.img_list = glob.glob(os.path.join(self.root, "images")+"/*.png")
        self.label_list = glob.glob(os.path.join(self.root, "labels") + "/*.pkl")

        assert len(self.img_list) == len(self.label_list), \
            "Unmatched #images = {} with #labels = {}!".format(
                len(self.img_list),
                len(self.label_list)
            )

        print("Everest: loading {} images and {} labels ........".format(
            len(self.img_list), len(self.label_list))
        )

        self.all_images = np.empty(
            (len(self.img_list), self.image_size[0], self.image_size[1]),
            dtype=np.float32
        )
        self.all_labels = np.empty(
            (len(self.label_list), self.image_size[0], self.image_size[1])
        )

        # fig = plt.figure()
        for idx in range(len(self.img_list)):
            # with Image.open(self.img_list[idx]) as f:
            #     im = f.convert("L").copy()
            im = cv2.imread(self.img_list[idx], cv2.IMREAD_GRAYSCALE)
            lb_filename = self.img_list[idx].replace("/images/", "/labels/").replace(".png", ".pkl")
            with open(lb_filename, "rb") as f:
                dict = pickle.load(f)
            label = dict['mask']

            # im = cv2.resize(np.array(im), (self.image_size[0], self.image_size[1]), cv2.INTER_LINEAR)
            # label = cv2.resize(label, (self.image_size[0], self.image_size[1]), cv2.INTER_NEAREST)
            # ### resize into (400, 250) then pad into (400, 400) ###
            # im = cv2.resize(np.array(im), (400, 250), cv2.INTER_LINEAR)
            # label = cv2.resize(label, (400, 250), cv2.INTER_NEAREST)
            # im = cv2.copyMakeBorder(im, 75, 75, 0, 0, cv2.BORDER_CONSTANT)
            # label = cv2.copyMakeBorder(label, 75, 75, 0, 0, cv2.BORDER_CONSTANT)
            # ### resize into (400, 250) then pad into (400, 400) ###
            # im = rescale_image(im)

            im = convt_array_to_PIL(im)

            # fig = plt.figure()
            if self.shape_transforms is not None:
                label = Image.fromarray(label)
                im, label = self.shape_transforms(im, label)

            if self.photo_transforms is not None:
                im = rescale_image(np.array(im))
                im = self.photo_transforms(im)
                im = im.astype(np.float32)

                # ax = fig.add_subplot(121)
                # ax.imshow(im, cmap="gray")
                # ax = fig.add_subplot(122)
                # ax.imshow(im_ph, cmap="gray")
                # plt.show()

            self.all_images[idx,:] = np.array(im).astype(np.float16)
            self.all_labels[idx,:] = np.int64(np.array(label))

            # if idx >= 100:
            #     break

        if self.train_bool:
            self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        im = self.all_images[index]
        label = self.all_labels[index]
        im = im[np.newaxis, :, :] / 255.0
        return np.concatenate((im, im, im), axis=0), label
        # return im, label
        # if self.transforms is not None:
        #     im_t, lb_t = self.transforms(im, label)
        # else:
        #     im_t, lb_t = im.copy(), label.copy()
        # return np.expand_dims(im_t, axis=0), lb_t

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))
        sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, label = self.__getitem__(i)
            if label is not -1:
                for j in range(NUM_CLASSES):
                    counts[j] += np.sum(label == j)
        return counts

    def __compute_image_mean(self):
        sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, target = self.__getitem__(i)
            if i==0:
                img_mean=img
            else:
                img_mean+=img
        return 1.0*img_mean/self.__len__()


    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / np.sum(values)
        return torch.Tensor(p_values)

class UnityDataset(torch.utils.data.Dataset):
    """
    Everest dataset
    """
    def __init__(self,
                 root,
                 image_size,
                 shape_transforms=None,
                 photo_transforms=None,
                 train_bool=False,
        ):
        self.root = root
        self.image_size = image_size
        self.shape_transforms = shape_transforms
        self.photo_transforms = photo_transforms
        self.train_bool = train_bool

        self.img_list = glob.glob(os.path.join(self.root, "images")+"/*.png")
        self.label_list = glob.glob(os.path.join(self.root, "masks") + "/*.npy")

        assert len(self.img_list) == len(self.label_list), \
            "Unmatched #images = {} with #labels = {}!".format(
                len(self.img_list),
                len(self.label_list)
            )

        print("Unity: loading {} images and {} labels ........".format(
            len(self.img_list), len(self.label_list))
        )

        self.all_images = np.empty(
            (len(self.img_list), self.image_size[0], self.image_size[1]),
            dtype=np.float32
        )
        self.all_labels = np.empty(
            (len(self.label_list), self.image_size[0], self.image_size[1])
        )

        # fig = plt.figure()
        for idx in range(len(self.img_list)):
            # with Image.open(self.img_list[idx]) as f:
            #     im = f.convert("L").copy()
            im = cv2.imread(self.img_list[idx], cv2.IMREAD_GRAYSCALE)
            lb_filename = self.img_list[idx].replace("/images/", "/masks/").replace(".png", ".npy")
            label = np.load(lb_filename)

            # im = cv2.resize(np.array(im), (self.image_size[0], self.image_size[1]), cv2.INTER_LINEAR)
            # label = cv2.resize(label, (self.image_size[0], self.image_size[1]), cv2.INTER_NEAREST)
            # ### resize into (400, 250) then pad into (400, 400) ###
            # im = cv2.resize(np.array(im), (400, 250), cv2.INTER_LINEAR)
            # label = cv2.resize(label, (400, 250), cv2.INTER_NEAREST)
            # im = cv2.copyMakeBorder(im, 75, 75, 0, 0, cv2.BORDER_CONSTANT)
            # label = cv2.copyMakeBorder(label, 75, 75, 0, 0, cv2.BORDER_CONSTANT)
            # ### resize into (400, 250) then pad into (400, 400) ###
            # im = rescale_image(im)

            im = convt_array_to_PIL(im)

            # fig = plt.figure()
            if self.shape_transforms is not None:
                label = Image.fromarray(label)
                im, label = self.shape_transforms(im, label)

            if self.photo_transforms is not None:
                # ax = fig.add_subplot(131)
                # ax.imshow(im, cmap="gray")
                im = rescale_image(np.array(im))
                im = self.photo_transforms(im)
                im = im.astype(np.float32)

                # ax = fig.add_subplot(132)
                # ax.imshow(im, cmap="gray")
                # ax = fig.add_subplot(133)
                # ax.imshow(label)
                # plt.show()

            self.all_images[idx,:] = np.array(im).astype(np.float16)
            self.all_labels[idx,:] = np.int64(np.array(label))

            # if idx >= 100:
            #     break

        if self.train_bool:
            self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        im = self.all_images[index]
        label = self.all_labels[index]
        im = im[np.newaxis, :, :] / 255.0
        return np.concatenate((im, im, im), axis=0), label
        # return im, label
        # if self.transforms is not None:
        #     im_t, lb_t = self.transforms(im, label)
        # else:
        #     im_t, lb_t = im.copy(), label.copy()
        # return np.expand_dims(im_t, axis=0), lb_t

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))
        sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, label = self.__getitem__(i)
            if label is not -1:
                for j in range(NUM_CLASSES):
                    counts[j] += np.sum(label == j)
        return counts

    def __compute_image_mean(self):
        sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, target = self.__getitem__(i)
            if i==0:
                img_mean=img
            else:
                img_mean+=img
        return 1.0*img_mean/self.__len__()


    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / np.sum(values)
        return torch.Tensor(p_values)

class JointDataset(torch.utils.data.Dataset):
    """
    Everest dataset
    """
    def __init__(self,
                 root_source,
                 image_size,
                 data_to_train,
                 shape_transforms=None,
                 photo_transforms=None,
                 train_bool=False,
        ):
        self.root_source = root_source
        self.data_to_train = data_to_train
        self.image_size = image_size
        self.shape_transforms = shape_transforms
        self.photo_transforms = photo_transforms
        self.train_bool = train_bool

        self.img_list_source = glob.glob(os.path.join(self.root_source, "images")+"/*.png")
        self.label_list_source = glob.glob(os.path.join(self.root_source, "masks") + "/*.npy")

        with open(self.data_to_train, "rb") as f:
            self.train_data_list = pickle.load(f)

        tot_data = len(self.img_list_source) + len(self.train_data_list)

        self.all_images = np.empty((tot_data, self.image_size[0], self.image_size[1]),
                                   dtype=np.float32)
        self.all_labels = np.empty((tot_data, self.image_size[0], self.image_size[1]))

        for idx in range(len(self.img_list_source)):
            im = cv2.imread(self.img_list_source[idx], cv2.IMREAD_GRAYSCALE)
            lb_filename = self.img_list_source[idx].replace("/images/", "/masks/").replace(".png", ".npy")
            label = np.load(lb_filename)
            im = convt_array_to_PIL(im)

            if self.shape_transforms is not None:
                label = Image.fromarray(label)
                im, label = self.shape_transforms(im, label)

            if self.photo_transforms is not None:
                im = rescale_image(np.array(im))
                im = self.photo_transforms(im)
                im = im.astype(np.float32)

            self.all_images[idx,:] = np.array(im).astype(np.float16)
            self.all_labels[idx,:] = np.int64(np.array(label))

        for idx in range(len(self.train_data_list)):
            with Image.open(self.train_data_list[idx]) as f:
                im = f.convert("L").copy()
            lb_filename = self.train_data_list[idx].replace("/images/", "/masks/").replace(".png", ".npy")
            lb = np.load(lb_filename)
            if self.shape_transforms is not None:
                lb = Image.fromarray(lb)
                im, lb = self.shape_transforms(im, lb)

            if self.photo_transforms is not None:
                im = rescale_image(np.array(im))
                im = self.photo_transforms(im)
                im = im.astype(np.float32)

            self.all_images[len(self.img_list_source)+idx, :] = np.array(im).astype(np.float16)
            self.all_labels[len(self.img_list_source)+idx, :] = np.int64(np.array(lb))

        print("Done loading {} ..............".format(self.all_images.shape[0]))

        if self.train_bool:
            self.counts = self.__compute_class_probability()

    def __len__(self):
        return self.all_images.shape[0]

    def __getitem__(self, index):
        im = self.all_images[index]
        label = self.all_labels[index]
        im = im[np.newaxis, :, :] / 255.0
        return np.concatenate((im, im, im), axis=0), label

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))
        sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, label = self.__getitem__(i)
            if label is not -1:
                for j in range(NUM_CLASSES):
                    counts[j] += np.sum(label == j)
        return counts

    def __compute_image_mean(self):
        sampleslist = np.arange(self.__len__())
        for i in sampleslist:
            img, target = self.__getitem__(i)
            if i==0:
                img_mean=img
            else:
                img_mean+=img
        return 1.0*img_mean/self.__len__()


    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values / np.sum(values)
        return torch.Tensor(p_values)