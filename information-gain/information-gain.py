import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    y = np.asarray(y)
    split_mask = np.asarray(split_mask)

    H_parent = _entropy(y)
    left, right = y[split_mask], y[~split_mask]
    N = y.size

    if left.size == 0 or y.size == 0:
        return 0.0

    weighted = (left.size / N) * _entropy(left) + (right.size / N) * _entropy(right)

    return H_parent - weighted

