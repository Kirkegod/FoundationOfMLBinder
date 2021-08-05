import numpy as np
from sklearn import datasets

def load_red_wine_data():
    """Load red wine dataset"""

    data_filename = "data/section7/winequality-red.csv"

    # We load the data and set any missing values to NaN
    data = np.genfromtxt(data_filename, delimiter=";", filling_values=np.nan)

    # Data features: "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide",
    # "total sulfur dioxide","density","pH","sulphates","alcohol","quality"
    X = data[:, :-1]
    y = np.array(data[:, -1], dtype=int)

    return X, y

def load_white_wine_data():
    """Load white wine dataset"""

    data_filename = "data/section7/winequality-white.csv"
    data = np.loadtxt(data_filename, delimiter=";")
    X = data[:, :-1]
    y = np.array(data[:, -1], dtype=int)

    # Remove the observations from class 9
    inds_to_keep = y!=9
    X = X[inds_to_keep, :]
    y = y[inds_to_keep]


    return X, y

def load_classification_diabetes_data():
    diabetes = datasets.load_diabetes()

    # Fix categorical variable
    diabetes.data[diabetes.data[:, 1] > 0, 1] = 1
    diabetes.data[diabetes.data[:, 1] < 0, 1] = 0

    # Randomly split the data into training and test sets
    np.random.seed(23)
    n_train = int(0.8 * diabetes.data.shape[0])
    shuffled_inds = np.random.choice(diabetes.data.shape[0], size=diabetes.data.shape[0], replace=False)

    X_train = diabetes.data[shuffled_inds[:n_train], :]
    X_test = diabetes.data[shuffled_inds[n_train:], :]

    # Convert numerical target to categorical
    split = 140
    diabetes.target[diabetes.target <= split] = 0
    diabetes.target[diabetes.target > split] = 1

    y_train = diabetes.target[shuffled_inds[:n_train]]
    y_test = diabetes.target[shuffled_inds[n_train:]]

    return X_train, y_train, X_test, y_test
