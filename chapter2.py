import numpy as np

# Exercise 2.1
def load_bat_data():
    train_data = np.loadtxt("data/chapter2/bat_train_data.csv", delimiter=",")
    test_data = np.loadtxt("data/chapter2/bat_test_data.csv", delimiter=",")

    X_train = train_data[:, :2]
    y_train = (train_data[:, 2]).astype(int)

    X_test = test_data[:, :2]
    y_test = (test_data[:, 2]).astype(int)

    return X_train, y_train, X_test, y_test

# Exercise 2.2
def load_apartment_data():
    train_data = np.loadtxt("data/chapter2/apartment_train.csv", delimiter=",")
    val_data = np.loadtxt("data/chapter2/apartment_val.csv", delimiter=",")

    X_train = train_data[:,:2]
    y_train = train_data[:,2]

    X_val = val_data[:,:2]
    y_val = val_data[:,2]

    return X_train, y_train, X_val, y_val

