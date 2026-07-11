import numpy as np

def t_test_one_sample(x, mu0):
    """
    Compute one-sample t-statistic.
    """
    x = np.array(x, dtype=float)
    n = len(x)
    x_tilde = np.mean(x)
    s = np.sqrt(1 / (n - 1) * np.sum((x - x_tilde)**2))

    t = (x_tilde - mu0) / (s / np.sqrt(n))
    return t
    
