import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Returns: dict with 'mean', 'median', 'mode' as floats.
    """
    x = np.array(x)
    counts = Counter(x)
    return {
        "mean": np.mean(x),
        "median": np.median(x),
        "mode": float(max(counts, key=counts.get))
    }