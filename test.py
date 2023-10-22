from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from random import Random

# from deepforest import CascadeForestRegressor
#
# X, y = load_boston(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
# model = CascadeForestRegressor(random_state=1)
# print(X_train)
# print(y_train)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print("\nTesting MSE: {:.3f}".format(mse))
with open("dataset/test_set.txt", "r") as file, open("dataset/test_set2.txt", "w") as file1:
    line = file.readline()
    rad = Random()
    while len(line) != 0:
        print(line[-2])
        if line[-2] == "1":
            file1.write(line)
        # else:
        #     int = rad.randint(0, (6444 - 114) // 114)
        #     if  int == 2:
        #         file1.write(line)
        line=file.readline()
