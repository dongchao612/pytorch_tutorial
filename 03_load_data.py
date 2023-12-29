#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/9/22 19:51
# Author  : dongchao
# File    : 03_load_data.py
# Software: PyCharm

import os
from PIL import Image
from torch.utils.data import Dataset


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


if __name__ == '__main__':
    root_dir = "./dataset/train"
    label_dir = "ants"
    ants_dataset = MyData(root_dir, label_dir)
    print(ants_dataset.__len__())
    print(ants_dataset[0], type(ants_dataset[0][0]))
    img, label = ants_dataset[0]
    img.show()
    print(label)

    root_dir = "./dataset/train"
    label_dir = "bees"
    bees_dataset = MyData(root_dir, label_dir)
    print(bees_dataset.__len__())
    print(bees_dataset[0], type(bees_dataset[0][0]))
    img, label = bees_dataset[0]
    img.show()
    print(label)

    train_dataset = ants_dataset + bees_dataset
    print(train_dataset.__len__())
    print(train_dataset[244], type(train_dataset[244][0]))
