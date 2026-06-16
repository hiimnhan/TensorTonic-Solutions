import numpy as np

def winsorize(data, lo_q, hi_q):
    """Returns: np.ndarray of shape (3, m, n), stacked clipped values, lo_mask, hi_mask"""
    d = np.array(data, dtype=np.float64)
    lo = np.percentile(d, lo_q, axis=0)
    hi = np.percentile(d, hi_q, axis=0)
    clipped = np.clip(d, lo, hi)
    lo_mask = (d < lo).astype(np.float64)
    hi_mask = (d > hi).astype(np.float64)

    return np.stack([clipped, lo_mask, hi_mask])