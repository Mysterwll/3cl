import os
import re
import torch
from torch import nn
from model.lstm import lstm4, lstm2, lstm1, lstm3, lstm6, lstm7, lstm5, hy
from model.deep1 import El, deep1
from model.tf import tf3, tf, tf4, tf5, tf8, tf_d
from model.ablation import  tf_d_0,tf_d_1,tf_d_2,tf_d_3,tf_d_0_0,tf_d_4,tf_d_w1_a,tf_d_w1_b
from model.transformer import trans
import joblib
import xgboost as xgb
from sklearn.ensemble import *
from deepforest import CascadeForestClassifier


def get_net(net_name, _ep, seq_len, channel, coding_type, new=True):
    if net_name == "RF":
        if not new:
            pth = "./pth/ml/{}.model".format(net_name)
            if os.path.exists(pth):
                return joblib.load("./pth/ml/{}.model".format(net_name)), True
        return RandomForestClassifier(random_state=10), False
    if net_name == "DF2":
        pth = "./pth/ml/{}.model".format(net_name)
        if os.path.exists(pth):
            return joblib.load("./pth/ml/{}.model".format(net_name)), True
        return CascadeForestClassifier(random_state=1,
                                       ), False
    if net_name == "DF":
        pth = "./pth/ml/{}.model".format(net_name)
        if os.path.exists(pth):
            return joblib.load("./pth/ml/{}.model".format(net_name)), True
        return CascadeForestClassifier(random_state=0,
                                       use_predictor=True,
                                       n_trees=550,
                                       n_estimators=4,
                                       ), False
    if net_name == "Xgboost":
        if not new:
            pth = "./pth/ml/{}.model".format(net_name)
            if os.path.exists(pth):
                return joblib.load("./pth/ml/{}.model".format(net_name)), True
        return xgb.XGBClassifier(), False

    path = "./pth/{}/{}".format(coding_type, net_name)
    if not os.path.exists(path):
        os.makedirs(path)
    list = os.listdir(path)
    max = 0
    if _ep != 0:
        max = _ep
    else:
        for s in list:
            epoch = re.search(r'(\d+)$', s).group()
            if max < int(epoch):
                max = int(epoch)
    if max != 0:
        print("epoch:{}".format(max))
        net = torch.load('./pth/{}/{}/network.pth{}'.format(coding_type, net_name, max), map_location='cpu')
        return net, max
    if net_name == "deep1":
        net = deep1(1062, 512, 256, 128, 64, 16, 2)
        return net, 0
    if net_name == "lstm1":
        net = lstm1(seq_len, channel)
        return net, 0
    if net_name == "lstm2":
        net = lstm2(seq_len, channel)
        return net, 0
    if net_name == "lstm3":
        net = lstm3(seq_len, channel)
        return net, 0
    if net_name == "lstm4":
        net = lstm4(seq_len, channel)
        return net, 0
    if net_name == "lstm5":
        net = lstm5(seq_len, channel)
        return net, 0
    if net_name == "lstm6":
        net = lstm6(seq_len, channel)
        return net, 0
    if net_name == "tf_d_w1_b":
        net = tf_d_w1_b()
        return net, 0
    if net_name == "tf_d_w1_a":
        net = tf_d_w1_a()
        return net, 0
    if net_name == "lstm7":
        net = lstm7(seq_len, channel)
        return net, 0
    if net_name == "lstm8":
        net = lstm8(seq_len, channel)
        return net, 0
    if net_name == "tf":
        net = tf()
        return net, 0
    if net_name == "tf2":
        net = tf2()
        return net, 0
    if net_name == "tf3":
        net = tf3()
        return net, 0
    if net_name == "tf_d_4":
        net = tf_d_4()
        return net, 0
    if net_name == "tf4":
        net = tf4()
        return net, 0
    if net_name == "tf_d_0":
        net = tf_d_0()
        return net, 0
    if net_name == "tf_d_0_0":
        net = tf_d_0_0()
        return net, 0
    if net_name == "tf_d_1":
        net = tf_d_1()
        return net, 0
    if net_name == "tf_d_2":
        net = tf_d_2()
        return net, 0
    if net_name == "tf_d_3":
        net = tf_d_3()
        return net, 0

    if net_name == "tf5":
        net = tf5()
        return net, 0
    if net_name == "tf8":
        net = tf8()
        return net, 0
    if net_name == "el":
        net = El(coord_channels=channel, one_hot_channels=20)
        return net, 0
    if net_name == "trans":
        net = trans()
        return net, 0
    if net_name == "tf_d":
        net = tf_d()
        return net, 0

    print("err")
