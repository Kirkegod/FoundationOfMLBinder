import numpy as np
from sklearn import datasets
import pickle

def load_classification_data():
    # Load classification training data
    train_data = np.load("data/chapter4/class_train_data.npy")
    X_train = train_data[:, :2]
    y_train = train_data[:, 2]

    # Load classification test/validation data
    test_data = np.load("data/chapter4/class_test_data.npy")
    X_test = test_data[:, :2]
    y_test = test_data[:, 2]

    return X_train, y_train, X_test, y_test
    
def load_classification_models():
    model_1 = pickle.load(open("data/chapter4/model_0", 'rb'))
    model_2 = pickle.load(open("data/chapter4/model_1", 'rb'))
    model_3 = pickle.load(open("data/chapter4/model_2", 'rb'))
    
    return model_1, model_2, model_3
    
def load_diabetes_data():
    diabetes = datasets.load_diabetes()
    
    # We exclude the categorical variable
    diabetes_X = np.column_stack((diabetes.data[:, 0], diabetes.data[:, 2:]))

    # Randomly split the data into training and test set
    np.random.seed(23)
    n_train = int(0.8 * diabetes_X.shape[0])
    shuffled_inds = np.random.choice(diabetes_X.shape[0], size=diabetes_X.shape[0], replace=False)

    X_train = diabetes_X[shuffled_inds[:n_train], :]
    X_test = diabetes_X[shuffled_inds[n_train:], :]

    y_train = diabetes.target[shuffled_inds[:n_train]]
    y_test = diabetes.target[shuffled_inds[n_train:]]

    return X_train, y_train, X_test, y_test

