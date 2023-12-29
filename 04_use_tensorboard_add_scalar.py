#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/9/22 20:14
# Author  : dongchao
# File    : 04_use_tensorboard_add_scalar.py
# Software: PyCharm

from torch.utils.tensorboard import SummaryWriter

if __name__ == '__main__':
    writer = SummaryWriter("tensor_board_log_dir")
    # writer.add_image()
    # y = x
    for i in range(100):
        writer.add_scalar("y = 2x", 2*i, i)
    writer.close()
