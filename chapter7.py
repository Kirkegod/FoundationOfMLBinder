import numpy as np

def load_red_wine_data():
    """Load red wine dataset"""
    data_filename = "winequality-red.csv"
    
    # We load the data and set any missing values to NaN
    data = np.genfromtxt(data_filename, delimiter=";", filling_values=np.nan)
    
    # Data features: "fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide",
    # "total sulfur dioxide","density","pH","sulphates","alcohol","quality"
    X = data[:, :-1]
    y = np.array(data[:, -1], dtype=int)
    
    return X, y