import numpy as np

def sigmoid(x):
    """ sigmoid function
    :param x: input, D dimension
    :return: value
    """
    # Avoid floating point overflow
    sigmoid_range = 34.538776394910684
    x = np.where(x <= -sigmoid_range, -sigmoid_range, x)
    x = np.where(x >= sigmoid_range, sigmoid_range, x)
    return 1 / (1 + np.exp(-x))

def cos(x, y):
    """ cosine similarity
    :param x, y: input, numpy vectors
    :return: value 
    """
    eps = 1e-8
    return np.dot(x, y) / (np.linalg.norm(x) + np.linalg.norm(y) + eps)
