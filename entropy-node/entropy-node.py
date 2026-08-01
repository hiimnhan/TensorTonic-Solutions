import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y = np.asarray(y)
    N = len(y)

    _, counts = np.unique(y, return_counts=True)
    p = counts / N

    return -np.sum(np.log2(p) * p)