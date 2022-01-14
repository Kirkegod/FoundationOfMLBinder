from sklearn.datasets import fetch_california_housing
from sklearn.utils import shuffle
from palmerpenguins import load_penguins
import numpy as np


# Exercise 3.1
TRAIN_FRAC = 0.8  # Use 80% of data for training

def load_california_data():
    X, y = fetch_california_housing(return_X_y=True)

    X, y = shuffle(X, y, random_state=42)

    n = X.shape[0]
    n_train = int(n * TRAIN_FRAC)
    X_train = X[:n_train]
    X_test = X[n_train:]
    y_train = y[:n_train]
    y_test = y[n_train:]

    return X_train, y_train, X_test, y_test
    
def load_and_save_penguin_data():
    """ Load palmer penguin data including flipper length, bill length, body mass (attributes) and species (target)
    :return:
        X_train: train observations (n_train, n_attributes)
        y_train: train targets (n_train,)
        X_test: test observations (n_test, n_attributes)
        y_test: test targets (n_test,)
        classes: vector of available classes (n_classes,)
    """
    df = load_penguins()
    attributes = ["bill_length_mm", "body_mass_g", "flipper_length_mm"]
    target = "species"

    X = np.array(df[attributes])
    targets = np.array(df[target])

    # These two observations actually miss all chosen values, so we will exclude them
    nan_ind = np.isnan(X[:, 0] * X[:, 1] * X[:, 2])
    X = X[~nan_ind, :]
    targets = targets[~nan_ind]
    classes, y = np.unique(targets, return_inverse=True)

    # Split into train and test (randomly)
    n_train = 274

    np.random.seed(21)
    shuffled_inds = np.random.choice(X.shape[0], X.shape[0], replace=False)

    train_inds = shuffled_inds[:n_train]
    test_inds = shuffled_inds[n_train:]

    X_train, y_train = X[train_inds, :], y[train_inds]
    X_test, y_test = X[test_inds, :], y[test_inds]

    np.savetxt("data/section3/penguin_train.csv", np.column_stack((X_train, y_train)), delimiter=',')
    np.savetxt("data/section3/penguin_test.csv", np.column_stack((X_test, y_test)), delimiter=',')

    return X_train, y_train, X_test, y_test, classes


def load_penguin_data():
    train_data = np.loadtxt('data/section3/penguin_train.csv', delimiter=",")
    test_data = np.loadtxt('data/section3/penguin_test.csv', delimiter=",")

    X_train, y_train = train_data[:, :-1], train_data[:, -1]
    X_test, y_test = test_data[:, :-1], test_data[:, -1]
    classes, _ = np.unique(y_train, return_inverse=True)

    print(X_train.shape)
    print(y_train.shape)

    return X_train, y_train, X_test, y_test, classes

