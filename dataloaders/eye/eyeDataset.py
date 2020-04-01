import os
import sys

import numpy as np
import glob
import pickle
from PIL import Image
import cv2

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../" % file_path)

from utils.utils import rescale_image, convt_array_to_PIL

NUM_CLASSES = 4

import torch
import torch.utils.data

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
                 transforms=None,
                 train_bool=False,
        ):
        self.root = root
        self.transforms = transforms
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

            if self.transforms is not None:
                lb = Image.fromarray(lb)
                im, lb = self.transforms(im, lb)

            self.all_images[idx,:] = np.array(im).astype(np.float16)
            self.all_labels[idx,:] = np.int64(np.array(lb))

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

    def apply_transform(
            self, img, lb
    ):
        img, lb = self.transforms(
            img,
            lb,
        )
        return img, lb

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
                 transforms=None,
        ):
        self.root = root
        self.transforms = transforms
        self.image_size = image_size

        self.data_to_train = glob.glob(self.root+"/images/*.png")

        print("OpenEDS: Loading {} images ........".format(len(self.data_to_train)))

        self.all_images = np.empty(
            (len(self.data_to_train), self.image_size[0], self.image_size[1]),
            dtype=np.float32
        )
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
            # # im = rescale_image(im)
            # im = convt_array_to_PIL(im)
            # ## pad into 400x400 ##
            ### opencv read images ###

            if self.transforms is not None:
                im, _ = self.transforms(im, None)

            self.all_images[idx, :] = np.array(im).astype(np.float16)

    def __len__(self):
        return len(self.data_to_train)

    def __getitem__(self, index):
        im = self.all_images[index]
        im = im[np.newaxis, :, :] / 255.0
        return np.concatenate((im, im, im), axis=0)
        # return im

    def apply_transform(
            self, img, lb
    ):
        img, lb = self.transforms(
            img,
            lb,
        )
        return img, lb

class EverestDataset(torch.utils.data.Dataset):
    """
    Everest dataset
    """
    def __init__(self,
                 root,
                 image_size,
                 transforms=None,
                 train_bool=False,
        ):
        self.root = root
        self.image_size = image_size
        self.transforms = transforms
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

            if self.transforms is not None:
                label = Image.fromarray(label)
                im, label = self.transforms(im, label)

            self.all_images[idx,:] = np.array(im).astype(np.float16)
            self.all_labels[idx,:] = np.int64(np.array(label))

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
