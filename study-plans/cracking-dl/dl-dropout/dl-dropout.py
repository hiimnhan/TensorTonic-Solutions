import numpy as np

def dropout(X, mask, drop_prob, mode):
    """
    Returns: 2D list with values rounded to 4 decimal places.
    """
    X = np.asarray(X, dtype=float)
    if mode == "test":
        return [[round(float(x), 4) for x in row] for row in X]
    mask = np.asarray(mask, dtype=float)
    if drop_prob == 0:
        return [[round(float(x), 4) for x in row] for row in X]
    output = (X * mask) / (1-drop_prob)
    return [[round(float(x), 4) for x in row] for row in output]