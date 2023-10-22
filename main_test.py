import os
import random

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tsne import plt_tsne
from dataset import load_dataset, encoder, loader
from dataset.encoder import Encoder
from dataset.load_dataset import load
from utils.log import log
from utils.save import get_net
from utils.print_plt import print_plt
from tsne import plot_embedding


def change(x):
    x = x.reshape((x.shape[0], seq_len, x.shape[1] // seq_len))
    x = x.transpose(2, 1)
    return x


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


setup_seed(20)



def main_test(type, seq_len, channel, coding_type, dan="a", new=True, a=1.0, is_p=False, ep=0):
    global last_out
    score_list = []  # 存储预测得分
    label_list = []  # 存储真实标签
    if tsne_:
        last_out = []  # 中间层输出
    if type == "h":
        sum = 0
        # 获取数据
        data = loader.DataSet(net_name, True, seq_len, channel, coding_type)
        dataloader = DataLoader(data, batch_size=batch_size)

        net, epoch = get_net(net_name, ep, seq_len,
                             channel, coding_type[0], new)

        if use_gpu:
            net = net.cuda()
        net.eval()
        print(net)

        right_sum = 0
        if len(coding_type) == 2 and dan == "a":
            net2 = joblib.load("pth/ml3/DF2-{}.model".format(df_type))
            for i, (x1, x2, y) in enumerate(dataloader):
                if use_gpu:
                    x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()

                data1 = change(x1)
                data2 = change(x2)
                print(data2.shape)
                print(data1.shape)
                data = np.concatenate((data1, data2), axis=1).reshape(data1.shape[0],(20+531)*5)
                print(data.shape)
                y2 = net2.predict_proba(data)
                x1 = change(x1).transpose(1, 2)
                x2 = change(x2).transpose(1, 2)
                if nums==2:
                    y1,y_m = net(x1,x2)
                else:
                    y1=net(x1)
                y2 = torch.Tensor(y2)
                y_p = y1 * (a) + y2 * (1 - a)
                right_sum += (y_p.argmax(dim=1) == y).float().sum().item()
                sum += y.shape[0]
                score_tmp = y_p
                score_list.extend(score_tmp.detach().cpu().numpy())
                label_list.extend(y.cpu().numpy())
                if tsne_:
                    last_out.append(y_m.detach().cpu().numpy())

        elif len(coding_type) == 1:
            net2 = joblib.load("pth/ml/DF-{}.model".format(df_type))
            for i, (x1, y) in enumerate(dataloader):
                x2=x1
                if use_gpu:
                    x1, x2, y = x1.cuda(), x2.cuda(), y.cuda()
                x1 = change(x1).transpose(1, 2)
                y2 = net2.predict_proba(x2)
                # y1 = net(x1).cuda()
                # y2 = torch.Tensor(y2).cuda()
                x2 = change(x2).transpose(1, 2)
                if nums==2:
                    y1 = net(x1,x2)
                else:
                    y1=net(x1)
                y2 = torch.Tensor(y2)
                y_p = y1 * (a) + y2 * (1 - a)
                right_sum += (y_p.argmax(dim=1) == y).float().sum().item()
                sum += y.shape[0]
                score_tmp = y_p
                score_list.extend(score_tmp.detach().cpu().numpy())
                label_list.extend(y.cpu().numpy())
                if tsne_:
                    last_out.append(y_m.detach().cpu().numpy())
        elif len(coding_type) == 3:
            for i, (x, x1, x2, y) in enumerate(dataloader):
                r = 0
                if use_gpu:
                    x, x1, x2, y = x.cuda(), x1.cuda(), x2.cuda(), y.cuda()
                x = change(x)
                x1 = change(x1)
                y_p = net(x, x1)
                sum += len(y)
                right_sum = (y_p.argmax(dim=1) == y).float().sum().item()
                print("right:{:.4}".format(right_sum))
                r += right_sum
                score_tmp = y_p
                score_list.extend(score_tmp.detach().cpu().numpy())
                label_list.extend(y.cpu().numpy())
        if tsne_:
            sub_out_list = last_out[0]
            for i in range(len(last_out) - 1):
                sub_out_list = np.concatenate((sub_out_list, last_out[i + 1]), 0)
            print(sub_out_list.shape)

            plt_tsne(sub_out_list, label_list,"tf3","fin")
        print(score_list[0][0])
        for i in range(len(score_list)):
            if score_list[i][ label_list[0]] <0.5:
                if label_list[i]==0:
                    print(i)
        print_plt(score_list, label_list, net_name,
                  "{}-{}-a{}".format(bz, epoch, a), is_p)


# net_name = "tf"
# net_name = "lstm4"
# net_name = "deep1"
# net_name = "el"
# net_name = "deep1"
# _ep = 7000
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
gpu = "cuda:0"
use_gpu = False
num_class = 1  # 类别数量
att_type = ""

net_name = "tf_d_4"
_ep=760
df_type="blosum62"
nums=2
path = "./dataset/test_set.txt"
batch_size = 2000
# coding_type = ["blosum62", "blosum62"]
coding_type = ["aaindex","blosum62"]
bz = "tf5.5-aaindex"
dan = "a"
seq_len = 5
channel = 20
tsne_ = True

main_test('h', seq_len, channel,
          coding_type, "a", False, 0, True, _ep)
#           # coding_type, "a", False, 0.479 , True, _ep)
#           coding_type, "a", False, 0.45839 , True, _ep)

# main_test('h', seq_len, channel,
#           coding_type, "a", False, 0.518, True, _ep);
