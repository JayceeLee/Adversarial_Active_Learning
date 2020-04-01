import os
import sys

import numpy as np
import glob
import pickle

file_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../" % file_path)

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
                 photometric_transform=None,
                 train_bool=False,
        ):
        self.root = root
        self.photometric_transform = photometric_transform
        self.train_bool = train_bool
        self.image_size = image_size
        self.data_to_train = data_to_train

        if self.data_to_train != None:
            with open(self.data_to_train, "rb") as f:
                self.train_data_list = pickle.load(f)
        else:
            self.train_data_list = glob.glob(self.root+"/images/*.png")

        print("OpenEDS: Found {} images".format(len(self.train_data_list)))

        self.all_images = np.empty((len(self.train_data_list), self.image_size, self.image_size), dtype=np.float32)
        self.all_labels = np.empty((len(self.train_data_list), self.image_size, self.image_size))
        for idx in range(len(self.train_data_list)):
            im = np.load(self.train_data_list[idx])
            im = np.float32(im)
            lb_filename = self.train_data_list[idx].replace("/images/", "/masks/")
            lb = np.load(lb_filename)
            self.all_images[idx, :] = im.copy()
            self.all_labels[idx, :] = lb.copy()

        if self.train_bool:
            self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.train_data_list)

    def __getitem__(self, index):
        im = self.all_images[index]
        lb = self.all_labels[index]

        if self.photometric_transform:
            im, lb = self.apply_transform(
                im,
                lb,
            )
            return im, np.asarray(lb)

        return np.expand_dims(im, axis=0), np.asarray(lb)

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
        img, lb = self.photometric_transform(
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
                 photometric_transform=None,
        ):
        self.root = root
        self.photometric_transform = photometric_transform
        self.image_size = image_size

        self.data_to_train = glob.glob(self.root+"/images/*.png")

        print("OpenEDS: Found {} images".format(len(self.data_to_train)))

        self.all_images = np.empty(
            (len(self.data_to_train), self.image_size, self.image_size),
            dtype=np.float32
        )
        for idx in range(len(self.data_to_train)):
            im = np.load(self.data_to_train[idx])
            im = np.float32(im)
            self.all_images[idx, :] = im.copy()

    def __len__(self):
        return len(self.data_to_train)

    def __getitem__(self, index):
        im = self.all_images[index]

        if self.photometric_transform:
            im, lb = self.apply_transform(
                im,
                None,
            )
            return im, np.asarray(lb)

        return np.expand_dims(im, axis=0)

    def apply_transform(
            self, img, lb
    ):
        img, lb = self.photometric_transform(
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
                 photometric_transform=None,
                 train_bool=False,
        ):
        self.root = root
        self.image_size = image_size
        self.photometric_transform = photometric_transform
        self.train_bool = train_bool

        self.img_list = glob.glob(os.path.join(self.root, "images")+"/*.png")
        self.label_list = glob.glob(os.path.join(self.root, "labels") + "/*.pkl")

        assert len(self.img_list) == len(self.label_list), \
            "Unmatched #images = {} with #labels = {}!".format(
                len(self.img_list),
                len(self.label_list)
            )

        print("Everest: Found {} images and {} labels".format(
            len(self.img_list), len(self.label_list))
        )

        self.all_images = np.empty(
            (len(self.img_list), self.image_size, self.image_size),
            dtype=np.float32
        )
        self.all_labels = np.empty(
            (len(self.label_list), self.image_size, self.image_size)
        )
        for idx in range(len(self.img_list)):
            im = np.load(self.img_list[idx])
            im = np.float32(im)
            lb_filename = self.img_list[idx].replace("/images/", "/labels/")
            label = np.load(lb_filename)
            label = np.int64(label)
            self.all_images[idx,:] = im.copy()
            self.all_labels[idx,:] = label.copy()

        if self.train_bool:
            self.counts = self.__compute_class_probability()

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        im = self.all_images[index]
        label = self.all_labels[index]

        if self.photometric_transform is not None:
            im_t = self.photometric_transform(im)
        else:
            im_t = im.copy()

        return np.expand_dims(im_t, axis=0), label

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
