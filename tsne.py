# coding='utf-8'
"""t-SNE 对手写数字进行可视化"""
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from dataset import loader, load_dataset


def plot_embedding(data, label, title, net_name, bz, save=True):
    print(data.shape)
    if save:
        np.save("pth/tsne_data.npy_train", data)
        np.save("pth/tsne_label_train.npy", label)
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(dpi=350, figsize=(12, 12))
    ax = plt.subplot(111)
    color = ["navy", "red"]
    # color = ["navy", (1, 0.1, 30 / 255, 1)]
    test = ["•", "•"]
    alpha = [1, 1]
    for i in range(data.shape[0]):
        if label[i] == 1:
            plt.text(data[i, 0], data[i, 1], str(test[label[i]]),
                     color=color[label[i]],
                     fontdict={'weight': 'bold', 'size': 18},
                     alpha=alpha[label[i]])
        else:
            plt.text(data[i, 0], data[i, 1], str(test[label[i]]),
                     color=color[label[i]],
                     fontdict={'size': 18},
                     alpha=alpha[label[i]])
    bwith = 4  # 边框宽度设置为2
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    # plt.scatter(0, 0, c=color[0],s=1, label="Positive",  edgecolors='white')
    # plt.scatter(0, 0, c=color[1],s=1, label="Negative",  edgecolors='white')
    plt.xticks([])
    plt.yticks([])
    plt.legend(loc='best', fontsize=30)
    plt.title("t-SNE Embedding of Features after Classification", fontdict={'weight': 'bold', 'size': 25})
    plt.rc('font', family='Arial')
    # plt.savefig('image/{}/tsen-{}-{}.jpg'
    plt.savefig('or.jpg'
                .format(net_name, net_name, bz), dpi=350,
                pad_inches=0)
    plt.show()


def plt_tsne(data, label, net_name, bz):
    n_samples, n_features = data.shape[0], data.shape[1]
    print(n_features, n_samples)
    print('Computing t-SNE embedding')
    # tsne = TSNE(n_components=2, perplexity=10,init='pca', random_state=0)
    # tsne = TSNE(n_components=2, perplexity=20, random_state=0)

    # tsne = TSNE(n_components=2, perplexity=10, init="pca"
                # , random_state=10)
    tsne = PCA(n_components=2,random_state=1221212)
    t0 = time()
    result = tsne.fit_transform(data)
    plot_embedding(result, label,
                   't-SNE embedding of the digits (time %.2fs)'
                   % (time() - t0), net_name, bz)


# data = np.load("pth/tsne_data.npy")
# # label = np.load("pth/tsne_label.npy")
data1 = load_dataset.load(["aaindex"])[0]
data2 = load_dataset.load(["blosum62"])[0]
test= data1.data["aaindex"]
test2= data2.data["blosum62"]
target=data1.target
test=np.concatenate((test2,test),axis=1)
# target=torch.Tensor(target).resize(530,1))
# label_onehot = torch.zeros(target.shape[0], 2)
# label_onehot.scatter_(dim=1, index=target, value=1 )
# label_onehot = np.array(label_onehot)
# print(test.shape)
# print(data1.target.shape)
# test=test.reshape(test.shape[0],test.shape[1]*test.shape[2])
plt_tsne(np.array(test), np.array( data1.target),"", "")
