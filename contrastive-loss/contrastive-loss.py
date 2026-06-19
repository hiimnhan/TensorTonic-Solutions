import numpy as np

def contrastive_loss(a, b, y, margin=1.0, reduction="mean") -> float:
    """
    a, b: arrays of shape (N, D) or (D,)  (will broadcast to (N,D))
    y:    array of shape (N,) with values in {0,1}; 1=similar, 0=dissimilar
    margin: float > 0
    reduction: "mean" (default) or "sum"
    Return: float
    """
    a = np.array(a, dtype=np.float64)
    b = np.array(b, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    d = a - b
    if d.ndim == 1:
        d = d[None, :]

    d = np.linalg.norm(d, axis=1)
    loss = y * d**2 + (1-y) * np.maximum(0, margin - d)**2

    return float(loss.mean() if reduction == "mean" else loss.sum())

    
