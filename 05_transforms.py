#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/9/22 20:31
# Author  : dongchao
# File    : 05_transforms.py
# Software: PyCharm
import numpy as np
import cv2 as cv
from PIL import Image
import imageio
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

if __name__ == '__main__':
    image_path = "dataset/train/bees/132511197_0b86ad0fff.jpg"
    img_pil = Image.open(image_path)

    writer = SummaryWriter("tensor_board_log_dir")
    transform_to_tensor = ToTensor()
    img_tensor = transform_to_tensor(img_pil)
    print(img_pil.size, img_tensor.shape)  # (500, 365) torch.Size([3, 365, 500])
    writer.add_image("test_add_Image_tensor_image", img_tensor, 1)

    img_cv = cv.imread(image_path)  # 通道排列顺序不一样，需要重新进行通道拼接
    b, g, r = cv.split(img_cv)
    img_cv = cv.merge([r, g, b])

    img_tensor = transform_to_tensor(img_cv)
    print(img_cv.shape, img_tensor.shape)  # (365, 500, 3) torch.Size([3, 365, 500])
    writer.add_image("test_add_cv_tensor_image", img_tensor, 1)

    img_imageio = imageio.v2.imread(image_path)
    img_tensor = transform_to_tensor(img_imageio)
    print(img_cv.shape, img_tensor.shape)  # (365, 500, 3) torch.Size([3, 365, 500])
    writer.add_image("test_add_imageio_tensor_image", img_tensor, 1)

    writer.close()

