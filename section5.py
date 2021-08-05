import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_concrete_data():
    np.random.seed(42)

    concrete_data = np.genfromtxt("data/section5/concrete_data.csv", delimiter=",")
    X = concrete_data[:,:-1]
    y = concrete_data[:,-1]

    X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, train_size=0.6)
    X_val, X_test, y_val, y_test = train_test_split(X_holdout, y_holdout, train_size=0.5)

    # Normalize as pre-procesing
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_val, y_val, X_test, y_test
