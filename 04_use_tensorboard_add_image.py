#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/9/22 20:23
# Author  : dongchao
# File    : 04_use_tensorboard_add_image.py
# Software: PyCharm

import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    writer = SummaryWriter("tensor_board_log_dir")
    img_pil = Image.open("dataset/train/bees/132511197_0b86ad0fff.jpg")
    img_np = np.array(img_pil)
    print(type(img_np), img_np.shape)  # (512, 768, 3)
    writer.add_image("test_add_image", img_np, 2, dataformats='HWC')
    writer.close()
