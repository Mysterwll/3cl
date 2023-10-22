import numpy as np
import pandas as pd
import torch
from sklearn import preprocessing

abs = "/home/wqs/3cl/"


class Encoder:
    def __init__(self):
        data_csv = pd.read_csv(abs + "dataset/data/aaindex.csv")
        # data_csv = pd.read_csv("aaindex.csv")
        self.index = {}
        for i in range(data_csv.shape[0]):
            self.index[data_csv["AA"].iloc[i]] = np.array(data_csv[i:i + 1])[:, 1:]

    def encode(self, strs, net_name):
        strs = strs.replace(',', '')
        strs = strs.replace('Q↓', '')
        strs = strs.replace('\n', '')
        strs = strs.replace('1', '')
        strs = strs.replace('0', '')
        if len(strs) != 5:
            print("length is not 5")
            return
        if net_name == "deep1":
            data = self.index[strs[0]]
            for i in range(4):
                data = np.concatenate((data, self.index[strs[i + 1]]), 1)
            data = data.reshape(1, data.shape[1]).astype(np.float32)
            data = torch.from_numpy(data)
            return data
        if net_name[:4] == "lstm":
            data = self.index[strs[0]]
            for i in range(4):
                data = np.concatenate((data, self.index[strs[i + 1]]), 0)
            data = data.transpose(1, 0)
            data = np.expand_dims(data, 0).astype(np.float32)
            data = torch.from_numpy(data)
            return data
        if net_name == 'el':
            data = self.index[strs[0]]
            for i in range(4):
                data = np.concatenate((data, self.index[strs[i + 1]]), 1)
            data = data.reshape(1, data.shape[1]).astype(np.float32)
            data = torch.from_numpy(data)
            one_hot = pd.read_csv(abs+"dataset/data/train_set.csv", header=None)
            data_onehot = np.array(one_hot.iloc[:, :-1])
            enc = preprocessing.OneHotEncoder()
            enc.fit(data_onehot)  # 训练。这里共有4个数据，3种特征
            one_hot = enc.transform([np.array(list(strs))]).toarray()  # 测试。这里使用1个新数据来测试
            one_hot = torch.from_numpy(one_hot)
            return one_hot, data


def test():
    print()

#
# coder = Encoder()
# a, b = coder.encode("VRLSP", "el")
# print(a)
# print(b)
