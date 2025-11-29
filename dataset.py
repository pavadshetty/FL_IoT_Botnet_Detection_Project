import numpy as np

def load_data():
    X = np.random.rand(500, 10)
    y = np.random.randint(0, 2, 500)
    return X, y
