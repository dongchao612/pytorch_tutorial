#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/9/22 20:49
# Author  : dongchao
# File    : 06_common_transform.py
# Software: PyCharm
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Normalize, Resize, Compose, RandomCrop

if __name__ == '__main__':
    image_path = "dataset/train/bees/132511197_0b86ad0fff.jpg"

    writer = SummaryWriter("tensor_board_log_dir")
    img = Image.open(image_path)
    print(img)

    transform_to_tensor = ToTensor()
    img_tensor = transform_to_tensor(img)
    writer.add_image("ToTensor", img_tensor, 1)

    transform_normalize = Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    img_tensor_normalize = transform_normalize(img_tensor)
    writer.add_image("Normalize", img_tensor_normalize, 1)

    # print(img.size)
    transform_resize = Resize((512, 512))
    img_resize = transform_resize(img)
    img_tensor_resize = transform_to_tensor(img_resize)
    writer.add_image("Resize", img_tensor_resize, 1)
    # print(img_tensor_resize)

    # use Compose
    transform_compose = Compose([
        transform_resize,
        transform_to_tensor
    ])
    img_tensor_resize = transform_compose(img)
    writer.add_image("Compose", img_tensor_resize, 1)

    transform_RandomCrop = RandomCrop((512, 512))
    transform_compose = Compose([
        transform_RandomCrop,
        transform_to_tensor
    ])
    for i in range(10):
        img_tensor_compose = transform_compose(img)
        writer.add_image("RandomCrop", img_tensor_compose, i)

    writer.close()
