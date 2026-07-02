import numpy as np

def sample_var_std(x):
    """
    Returns: dict with 'variance' and 'std_dev' as floats.
    """
    x = np.array(x, dtype=float)
    N = len(x)
    mean = x.mean()
    variance = np.sum((x - mean)**2) / (N - 1)
    std = np.sqrt(variance)

    return {"variance": variance, "std_dev": std}