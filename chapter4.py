import numpy as np

def load_classification_data():
    # Load classification training data
    train_data = np.load("class_train_data.npy")
    X_train = train_data[:, :2]
    y_train = train_data[:, 2]

    # Load classification test/validation data
    test_data = np.load("class_test_data.npy")
    X_test = test_data[:, :2]
    y_test = test_data[:, 2]

    print(X_train.shape)
    print(X_test.shape)

    return X_train, y_train, X_test, y_test
    
    
def load_diabetes_data():
    diabetes = datasets.load_diabetes()

    # De-scale categorical variable
    diabetes.data[diabetes.data[:, 1] < 0, 1] = 0
    diabetes.data[diabetes.data[:, 1] > 0, 1] = 1

    # Randomly split the data into training and test set
    np.random.seed(23)
    n_train = int(0.8 * diabetes.data.shape[0])
    shuffled_inds = np.random.choice(diabetes.data.shape[0], size=diabetes.data.shape[0], replace=False)

    X_train = diabetes.data[shuffled_inds[:n_train], :]
    X_test = diabetes.data[shuffled_inds[n_train:], :]

    y_train = diabetes.target[shuffled_inds[:n_train]]
    y_test = diabetes.target[shuffled_inds[n_train:]]

    return X_train, y_train, X_test, y_test