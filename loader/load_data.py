# -*- coding: utf-8 -*-
# @Time    : 2019/10/24 22:36
# @Author  : Mingxing Li
# @FileName: load_data.py
# @Software: PyCharm

from __future__ import print_function, division
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import torchvision.transforms.functional as tf
import cv2
import torch
import numbers
from PIL import Image, ImageOps
import random
import pdb


class AdjustBrightness(object):
    def __init__(self, bf=0.1):
        self.bf = bf

    def __call__(self, img, mask):
        return tf.adjust_brightness(img, np.random.uniform(1 - self.bf, 1 + self.bf)), mask


class AdjustContrast(object):
    def __init__(self, cf=0.1):
        self.cf = cf

    def __call__(self, img, mask):
        # assert img.size == mask.size
        return tf.adjust_contrast(img, np.random.uniform(1 - self.cf, 1 + self.cf)), mask


class RandomHorizontallyFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if np.random.random() < self.p:
            return (img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT))
        return img, mask


class RandomVerticallyFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):
        if np.random.random() < self.p:
            return (img.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM))
        return img, mask


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = np.random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(
                img,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.BILINEAR,
                fillcolor=(0, 0, 0),
                shear=0.0,
            ),
            tf.affine(
                mask,
                translate=(0, 0),
                scale=1.0,
                angle=rotate_degree,
                resample=Image.NEAREST,
                fillcolor=255,
                shear=0.0,
            ),
        )


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return (img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST))

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return (img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th)))


class Crop_Resize(object):
    def __init__(self, scale):
        """
        :param scale:  a list [0.5, 1.0]
        """

        # self.size = tuple(reversed(size))  # size: (h, w)
        self.scale_pkg = scale

    def __call__(self, img, label):
        self.scale = random.sample(self.scale_pkg, 1)[0]
        if self.scale == 1:
            return img, label
        else:
            # pdb.set_trace()
            h, w = img.size
            h_security = range(int(h - self.scale * h))  # 282
            w_security = range(int(w - self.scale * w))  # 292
            seed_w = random.sample(w_security, 1)[0]
            seed_h = random.sample(h_security, 1)[0]
            # print("seed_h", seed_h)
            img_security = np.array(img)[seed_w: int(seed_w + self.scale * w),
                           seed_h: seed_h + int(self.scale * h)]
            label_security = np.array(label)[seed_w: int(seed_w + self.scale * w),
                             seed_h: seed_h + int(self.scale * h)]
            # print("img_security", img_security.shape)
            # pdb.set_trace()

            # pdb.set_trace()
            # check porint
            # cv2.imwrite("img.tiff", np.array(img))
            # cv2.imwrite("label.tiff", np.array(label))
            return Image.fromarray(img_security), Image.fromarray(label_security)


class MyDataset(Dataset):
    def __init__(self, data_path,
                 label_path,

                 transform=None):

        self.init_seed = False
        self.transform = transform
        self.data_name = os.listdir(data_path)
        self.label_name = os.listdir(label_path)

        self.data_name = sorted(self.data_name)
        self.label_name = sorted(self.label_name)
        # self.label_name = self.data_name.copy()
        # self.label_name = [name.replace(".tif", ".gif") for name in self.label_name]
        # pdb.set_trace()
        self.data_path = []
        self.label_path = []

        for data, label in zip(self.data_name, self.label_name):
            self.data_path.append(data_path + "/" + data)
            self.label_path.append(label_path + "/" + label)

        self.adjustBrightness = AdjustBrightness()
        self.adjustContrast = AdjustContrast()
        self.randomHorizontallyFlip = RandomHorizontallyFlip()
        self.randomVerticallyFlip = RandomVerticallyFlip()
        self.randomRotate = RandomRotate(7)
        self.RandomCrop = RandomCrop([128, 128])
        self.Crop_Resize = Crop_Resize([1])

    def rand_crop(self, data, label):
        width1 = np.random.randint(0, data.size[0] - 128)
        height1 = np.random.randint(0, data.size[1] - 128)
        width2 = width1 + 128
        height2 = height1 + 128

        data = data.crop((width1, height1, width2, height2))
        label = label.crop((width1, height1, width2, height2))

        return data, label

    def __getitem__(self, index):
        if not self.init_seed:
            random.seed(1234)
            np.random.seed(1234)
            torch.manual_seed(1234)
            torch.cuda.manual_seed_all(1234)
            self.init_seed = True

        data = Image.open(self.data_path[index])
        label = Image.open(self.label_path[index])

        # data, label = self.Crop_Resize(data, label)
        # data, label = self.adjustBrightness(data, label)
        # data, label = self.adjustContrast(data, label)
        data, label = self.randomHorizontallyFlip(data, label)
        data, label = self.randomVerticallyFlip(data, label)
        # data, label = self.randomRotate(data, label)

        # if np.random.randn(1) < 1.28: # 90% crop
        # pdb.set_trace()
        data = np.array(data)
        label = np.array(label)
        
        # pdb.set_trace()
        if data.shape[-1]==3:
            data = torch.from_numpy(np.array(data).transpose(2, 0, 1)).float() / 255
            label = torch.from_numpy(np.array(label)).float().unsqueeze(0) / 255
        else:
            data = torch.from_numpy(data).unsqueeze(0).float() / 255
            label = torch.from_numpy(label).float().unsqueeze(0) / 255
        # label[label<1]=0
        # mask = Image.open(self.mask_path[index])

        return data, label

    def __len__(self):
        return len(self.data_path)





if __name__ == "__main__":
    batch = 2
    train_dataloader = MyDataset(data_path=r"E:\risc2019\simple\train_raw",
                                 label_path=r"E:\risc2019\simple\train_label",
                                 transform=transforms.Compose([
                                     transforms.ToTensor()
                                 ]))

    train_dataset = DataLoader(train_dataloader, batch_size=batch, shuffle=True)
    for data, label in train_dataset:
        pass
