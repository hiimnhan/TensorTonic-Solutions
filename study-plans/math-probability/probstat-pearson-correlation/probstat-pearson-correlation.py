import numpy as np

def pearson_correlation(X):
    """
    Returns: ndarray, the Pearson correlation matrix.
    """
    X = np.array(X, dtype=float)
    return np.corrcoef(X.T)