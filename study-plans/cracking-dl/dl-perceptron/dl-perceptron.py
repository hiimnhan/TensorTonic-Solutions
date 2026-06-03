import numpy as np

def perceptron(X, y, lr=0.1, epochs=100):
    """
    Returns: Tuple of (weights as list of floats, bias as float)
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    n, d = X.shape
    W = np.zeros(d)
    b = 0.0

    for _ in range(epochs):
        for i in range(n):
            z = W @ X[i] + b
            predicted = 1.0 if z >= 0 else 0.0
            error = y[i] - predicted
            W += lr * error * X[i]
            b +=lr * error

    return W.tolist(), float(b)