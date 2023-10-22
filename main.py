import numpy as np

from dataset import load_dataset
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from deepforest import CascadeForestRegressor

net_name = "deep1"
coding_type = "blosum62"
train, test = load_dataset.load(None, net_name, coding_type, 5)
model = CascadeForestRegressor(random_state=1)
y = np.array(train.target, dtype=np.int)
model = model.fit(train.data, y)

print(model)
