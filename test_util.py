import numpy as np

def floatEqual(a, b, precision=1e-7):
    return (np.abs(a - b) < precision)

def arrayEqual(a, b, precision=1e-7):
    return (a.shape == b.shape) and (np.sum(np.abs(a - b)) < precision)


