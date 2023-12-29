#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/9/22 19:38
# Author  : dongchao
# File    : 01_test_pytorch_is_avaliable.py
# Software: PyCharm
import torch
if __name__ == '__main__':
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

