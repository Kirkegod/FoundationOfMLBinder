from sklearn.datasets import fetch_california_housing
from sklearn.utils import shuffle


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