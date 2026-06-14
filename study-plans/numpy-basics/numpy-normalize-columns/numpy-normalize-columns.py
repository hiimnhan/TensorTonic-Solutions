import numpy as np

def normalize(data):
    """Returns: np.ndarray of shape (m, n), z-score normalized per column"""
    d = np.array(data, dtype=np.float64)
    mu = np.mean(d, axis=0)
    sigma = np.std(d, axis=0)
    return (d - mu) / sigma