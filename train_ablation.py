import random

import joblib
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler

from dataset import loader
from utils.log import log
from utils.save import get_net
from main_test import main_test

writer = SummaryWriter('log')


def change(x):
    x = x.reshape((x.shape[0], seq_len, x.shape[1] // seq_len))
    x = x.transpose(2, 1)
    return x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main_train(net_name, _ep, seq_len, channel, coding_type):
    # 给定随机数种子
    setup_seed(201)
    # 加载网络和损失函数
    net, epoch = get_net(net_name, _ep, seq_len, channel, coding_type[0])
    print(net)
    criterion1 = nn.CrossEntropyLoss()
    if use_gpu:
        net = net.cuda()
        criterion1 = criterion1.cuda()
    # 加载定义学习率
    lr_list = []
    LR = lr
    optimizer = optim.SGD(net.parameters(), lr=LR)
    # optimizer = optim.Adam(net.parameters(), lr=LR)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)

    # 获取数据集
    data = loader.DataSet(net_name, True, seq_len, channel, coding_type)
    dataloader = DataLoader(data, batch_size=batch_size)
    if use_gpu:
        net = net.cuda()
    while epoch < 20000:
        lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
        scheduler.step()
        right_sum, sum = 0, 0


        if len(coding_type) == 2:
            for i, (x1, x2, y) in enumerate(dataloader):
                if use_gpu:
                    x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
                x1 = change(x1).transpose(1, 2)
                x2 = change(x2).transpose(1, 2)
                out = net(x1, x2)
                right_sum += (out.argmax(dim=1) == y).float().sum().item()
                sum += y.shape[0]
                loss = criterion1(out, y)
                print_loss = loss.data.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epoch += 1
        writer.add_scalar("loss-{}-{}".format(net_name, bz),
                          loss.item(), epoch)
        if epoch % 10 == 0:
            print('epoch: {}, loss: {:.4}'.format(epoch, print_loss))
            acc = right_sum / sum
            print('label: {:.4}'.format(acc))
            writer.add_scalar("acc-{}-{}".format(net_name, bz), acc, epoch)
            if epoch % 10 == 0:
                print(scheduler.get_last_lr())
                log(net_name, epoch, coding_type, "", print_loss, acc, bz, lr, step_size)
                torch.save(net, './pth/{}/{}/network.pth{}'.format(coding_type[0], net_name, epoch))
                if train:
                    main_test('h', seq_len, channel,
                              coding_type, False, False, 0.5, True, epoch)


train = False
use_gpu = False
lr = 0.01
is_hy = False
step_size = 500
batch_size = 250
_ep = 0
seq_len = 5
# coding_type = ["blosum62"]
# coding_type = ["nmbroto"]
channel = 531
bz = ""
net_name = "tf_d_0_0"
coding_type = ["aaindex", "blosum62"]
main_train(net_name, _ep, seq_len, channel, coding_type)
