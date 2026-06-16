import numpy as np

def norm_gate(X, W, threshold):
    """Returns: np.ndarray of shape (n, k), gated projection where rows below threshold are zeroed"""
    X = np.array(X, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    Z = X @ W
    Z_norm = np.linalg.norm(Z, axis=1)
    return np.where(Z_norm[:, np.newaxis] >= threshold, Z, 0.0)