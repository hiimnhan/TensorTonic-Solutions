import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    X = np.asarray(X, dtype=float)
    n, d = X.shape
    centered = (X - np.mean(X, axis=0))

    C = (centered.T @ centered) / (n - 1)

    eigvals, eigvecs = np.linalg.eig(C)
    idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, idx]
    W = eigvecs[:, :k]

    X_proj = centered @ W
    return X_proj.tolist()
    
    
    
