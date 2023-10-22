import joblib
import numpy as np
import xgboost as xgb
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader

from dataset import load_dataset, loader
from utils import print_plt
from utils.save import get_net


def change(x):
    x = x.reshape((x.shape[0], 5, x.shape[1] // 5))

    return x  # 导入必要的工具包


def main_train(net_name, bz, coding_types, new):
    # score_list=[]
    train, test = load_dataset.load(coding_types)
    model, is_train = get_net(net_name, None, None, None, None, new)
    # model2, is_train = get_net("tf_d1", 440, None, None, "aaindex", new)
    print(model)
    y_train = np.array(train.target.numpy(), dtype=np.int32)
    y_test = np.array(test.target.numpy(), dtype=np.int32)
    # model.fit()
    # 使用交叉验证
    # from sklearn.model_selection import KFold
    # kfold = KFold(n_splits=5)
    # for train_index, test_index in kfold.split(train.data[coding_types[0]], y):
    #     # train_index 就是分类的训练集的下标，test_index 就是分配的验证集的下标
    #     this_train_x, this_train_y = train.data[train_index], y[train_index]  # 本组训练集
    #     this_test_x, this_test_y = train.data[test_index], y[test_index]  # 本组验证集
    #     # 训练本组的数据，并计算准确率
    #     model.fit(this_train_x, this_train_y)
    #     prediction = model.predict(test.data)
    #     score = accuracy_score(test.target, prediction)
    #     print(score)  # 得到预测结果区间[0,1]

    # print(train.data[coding_type])
    # if not is_train:
    #     # data1=change( train.data[coding_types[0]].numpy())
    #     #
    #     # data2=change(train.data[coding_types[1]].numpy())
    #     # data=np.concatenate((data1,data2),axis=2)
    #     # data=data.reshape(data.shape[0],-1)
    data1_train = train.data[coding_types[0]]
    data2_train = train.data[coding_types[1]]
    data1_train=change(train.data[coding_types[0]].numpy())
    data2_train=change(train.data[coding_types[1]].numpy())
    data_train=np.concatenate((data1_train,data2_train),axis=2)
    data_train=data_train.reshape(data_train.shape[0],-1)

    data1_test=change(test.data[coding_types[0]].numpy())
    data2_test=change(test.data[coding_types[1]].numpy())
    data_test=np.concatenate((data1_test,data2_test),axis=2)
    data_test=data_test.reshape(data_test.shape[0],-1)

    print("asdas",data_train.shape)
    model.fit(data_train, y_train)
    print(len(data_train))
    y_p=model.predict_proba(data_test)


    # x1 = model2(change(train.data["aaindex"]), change(train.data["blosum62"])).detach()
    # model.fit(x1.reshape(x1.shape[0], x1.shape[2] * x1.shape[1]), y)
    # data = loader.DataSet(net_name, False, 5, 531, ["aaindex", "blosum62"])
    # dataloader = DataLoader(data, batch_size=1000)
    # for i, (x, x1, y) in enumerate(dataloader):
    #     x1 = model2(change(x), change(x1)).detach()
    #     preds = model.predict_proba(x1.reshape(x1.shape[0], x1.shape[2] * x1.shape[1]))
    #     score_list.extend(preds)
    # data = test.data[coding_types[0]]
    # print(data.shape)

    if not is_train:
        path = "./pth/ml3/"
        if not os.path.exists(path):
            os.makedirs(path)
        joblib.dump(model, './pth/ml3/{}-{}.model'.format(net_name, coding_types[0]))
    # print(test.target)
    # print(preds)
    print(len(y_p))
    print_plt.print_plt(y_p, test.target.numpy(), net_name, "{}-{}".format(bz, coding_types[0]), is_p=True)


net_name = "DF2"
# net_name = "Xgboost"
# net_name = "DF2"
bz = "xg"
coding_types = ["blosum62","aaindex"]
# coding_types = ["nmbroto"]
# coding_types = ["aaindex"]
# coding_types = ["cksaap"]
# coding_types = ["cksaagp"]
# coding_types = ["egaac"]
# coding_types = ["one_hot"]
main_train(net_name, bz, coding_types, True)