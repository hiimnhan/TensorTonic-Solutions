import numpy as np

def scale_rows(data, weights):
    """Returns: np.ndarray of shape (m, n), each row scaled by corresponding weight"""
    d = np.array(data, dtype=np.float64)
    w = np.array(weights, dtype=np.float64)

    return d * w[:, np.newaxis]