import os

import numpy as np
import pandas as pd
import torch

from sklearn.datasets._base import Bunch
from sklearn import preprocessing

from dataset.encoder import Encoder

# abs = "/home/wqs/3cl/"
# abs = "D:\\code\\3cl\\"


abs = "E:\\3cl\\"
def change(x):
    x = x.reshape((x.shape[0], 5, x.shape[1] // 5))
    x = x.transpose(2, 1)
    return x


def load(coding_types):
    """
    获取双色球数据集
    :return:
    """
    train = Bunch()
    test = Bunch()
    train.data = {}
    test.data = {}
    for coding_type in coding_types:
        if coding_type != "one_hot":
            print(coding_type)
            data_train_csv = pd.read_csv(abs + "dataset/data/train_set_{}.csv".format(coding_type), header=None)
            data_test_csv = pd.read_csv(abs + "dataset/data/test_set_{}.csv".format(coding_type), header=None)
            train.data[coding_type] = _get_3cl_data(data_train_csv)
            test.data[coding_type] = _get_3cl_data(data_test_csv)
    data_test_one_hot = pd.read_csv(abs + "dataset/data/test_set.csv", header=None)
    data_train_one_hot = pd.read_csv(abs + "dataset/data/train_set.csv", header=None)
    test.target = _get_3cl_target(data_test_one_hot)
    train.target = _get_3cl_target(data_train_one_hot)
    if "one_hot" in coding_types:
        data_onehot_train = np.array(data_train_one_hot[:])[:, :-1]
        data_onehot_test = np.array(data_test_one_hot[:])[:, :-1]
        # print(data_onehot_test)
        enc = preprocessing.OneHotEncoder()
        enc.fit(data_onehot_train)  # 训练。这里共有4个数据，3种特征
        one_hot_train = torch.Tensor(enc.transform(data_onehot_train).toarray())  # 测试。这里使用1个新数据来测试
        one_hot_test = torch.Tensor(enc.transform(data_onehot_test).toarray())  # 测试。这里使用1个新数据来测试
        # print(one_hot_test.shape)
        test.data["one_hot"] = one_hot_test
        train.data["one_hot"] = one_hot_train
        # print(change(one_hot_train)[0])
        # print(change(train.data["aaindex"])[0])
        # one_hot = one_hot.reshape((one_hot.shape[0], seq_len, one_hot.shape[1] // seq_len))
        # one_hot = one_hot.transpose(0, 2, 1)

    return train, test


def _get_3cl_data(data):
    """
    获取双色球特征值
    :return:
    """

    data_r = data.iloc[:, 1:]
    data = np.array(data_r, dtype=np.float)
    print(data.shape)
    return torch.Tensor(data)


# else:
#     for i in range(len(index)):
#         data_np = []
#         if i == 0:
#             data_r = data.iloc[1:, index[i]:index[i] + 1]
#             data_np = np.array(data_r)
#             for k in range(4):
#                 np.concatenate((data_np,
#                                 np.array(data.iloc[1:, (k + 1) * 531 + index[i]:(k + 1) * 531 + index[i] + 1])), 0)
#         else:
#             for k in range(5):
#                 np.concatenate((data_np, np.array(data.iloc[1:, k * 531 + index[i]:k * 531 + index[i] + 1])), 0)
#     return data_np


def _get_3cl_target(data):
    data_b = data.iloc[:, -1:]
    data_np = np.array(data_b, dtype=np.long).flatten()
    return torch.Tensor(data_np).type(torch.LongTensor)


def get_index(key):
    index_csv = pd.read_csv(abs + "dataset/data/aaindex.csv")
    index = index_csv["AA"].tolist()
    i = index.index(key)
    return np.array(index_csv.iloc[i:i + 1]).flatten()[1:]


def load_index():
    index_csv = pd.read_csv(abs + "dataset/data/aaindex.csv")
    return np.array(index_csv.iloc[:])[:, 1:]

# load(None, "el")
