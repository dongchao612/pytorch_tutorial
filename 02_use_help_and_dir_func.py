#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/9/22 19:41
# Author  : dongchao
# File    : 02_use_help_and_dir_func.py
# Software: PyCharm

import torch

if __name__ == '__main__':
    print(dir(torch.cuda.is_available()))
    print("the use of torch.cuda.is_available...")
    help(torch.cuda.is_available)
