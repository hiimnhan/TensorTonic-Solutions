import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    y_left = np.array(y_left, dtype=int)
    y_right = np.array(y_right, dtype=int)

    NL, NR = len(y_left), len(y_right)
    N = NL + NR

    if N == 0:
        return 0.0

    def gini(t):
        if len(t) == 0:
            return 0.0
        _, counts = np.unique(t, return_counts=True)
        probs = counts / len(t)
        return 1.0 - np.sum(probs ** 2)

    return (NL / N) * gini(y_left) + (NR / N) * gini(y_right)

        
