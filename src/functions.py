import numpy as np

def sigmoid(x):
    """ sigmoid function
    :param x: input, single dimension
    :return: value
    """
    return 1 / (1 + np.exp(-x))

