import numpy as np

def swish(x):
    """
    Implement Swish activation function.
    """
    x = np.asarray(x, dtype=float)

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    return x * sigmoid(x)
