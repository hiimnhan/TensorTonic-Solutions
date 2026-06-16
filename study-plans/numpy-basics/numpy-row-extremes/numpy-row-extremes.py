import numpy as np

def row_extremes(data):
    """Returns: np.ndarray of shape (4, m), rows are max_val, max_col, min_val, min_col"""
    data = np.array(data, dtype=np.float64)

    mx = np.max(data, axis=1)
    arg_mx = np.argmax(data, axis=1)
    mn = np.min(data, axis=1)
    arg_mn = np.argmin(data, axis=1)

    return np.stack([mx, arg_mx, mn, arg_mn])