#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2023/9/30 16:28
# Author  : dongchao
# File    : train.py
# Software: PyCharm


import os

import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder

from AlexNet.model import AlexNet

torch.backends.enabled = True
torch.backends.benchmark = True

if __name__ == '__main__':

    tain_root = r"../dataset/train"
    test_root = r"../dataset/test"

    # 数据处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 重置大小到224*224
        transforms.ToTensor(),  # 变成Tensor格式
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 将图像像素归一化到[-1,1]之间
    ])

    batchSize = 16
    train_data = ImageFolder(root=tain_root, transform=train_transform)  # 244

    train_dataloader = DataLoader(train_data, batch_size=batchSize, shuffle=True, drop_last=True)  # 14

    # 如果有显卡，可转到GPU
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("device:{}".format(device))

    # 定义模型并转到GPU
    model = AlexNet().to(device)

    # 定义一个损失函数
    loss_fn = CrossEntropyLoss().to(device)

    # 定义一个优化器
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)

    # 学习率每隔10轮变为原来的0.5
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    # 开始训练
    epoch = 500
    min_acc = 0
    global_step = 0
    writer = SummaryWriter()

    for e in range(epoch):
        print("================= EPOCH: {}/{} =================".format(e + 1, epoch))

        for batch, (x, y) in enumerate(train_dataloader):  # 一次是16个数据，一共循环3750次
            # 前向传播
            x, y = x.to(device), y.to(device)
            output = model(x)

            cur_acc = sum(y == output.argmax(1)) / output.shape[0]

            cur_loss = loss_fn(output, y)

            writer.add_scalar("cur_acc", cur_acc, global_step)
            writer.add_scalar("cur_loss", cur_loss, global_step)

            # 梯度更新
            optimizer.zero_grad()
            cur_loss.backward()
            optimizer.step()
            # 保存最好的模型权重
            if cur_acc > min_acc:
                folder = "save_model"
                if not os.path.exists(folder):
                    os.mkdir(folder)
                min_acc = cur_acc
                print('save best model', "acc=", cur_acc.item(), )
                torch.save(model.state_dict(), "./best_model.pth")

            global_step += 1

    writer.close()
