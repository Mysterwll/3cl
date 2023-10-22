import random

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

from dataset import load_dataset, loader

data = loader.DataSet("lstm", False, 5, 531, ["aaindex"])
data2_train = loader.DataSet("lstm", True, 5, 531, ["aaindex"])
dataloader = DataLoader(data, batch_size=6226)
dataloader_train = DataLoader(data2_train, batch_size=530)
k_means = KMeans(n_clusters=20, random_state=10)
for i, (x, y) in enumerate(dataloader):
    k_means.fit(x)
    y_p1 = k_means.predict(x)
    w = open("lm1", "w")
    for i in y_p1:
        w.write("{}\n".format(i))
w1 = open("lm2", "w")
w2 = open("lm3", "w")
for i, (x, y) in enumerate(dataloader_train):
    y_p = k_means.predict(x)
    w1.write(str(y_p) + "\n")
joblib.dump(k_means, './pth/ml/kmeans.model')
data_csv = np.array(pd.read_csv("./error_aaindex.csv", header=None))
print(data_csv.shape)
y_p2 = k_means.predict(data_csv[:, 1:])
w2.write(str(y_p2) + "\n")
w.close()
# newData = pca.fit_transform(x)
# pca = PCA(n_components=3)
# plt.scatter(newData[:, 0], newData[:, 1], c=y_predict)
# plt.show()

# plt.scatter(x[:, 0], x[:, 1], x[:, 2], c=y_predict)
# print(k_means.predict((x[:30, :])))
# # print(metrics._calinski_harabaz_score(x, y_predict))
# print(k_means.cluster_centers_)
# print(k_means.inertia_)
# print(metrics.silhouette_score(x, y_predict))
