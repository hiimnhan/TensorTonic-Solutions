import numpy as np

def pooling(input, pool_size, stride, pool_type):
    """
    Returns: 3D list with pooled values rounded to 4 decimal places.
    """
    X = np.array(input, dtype=float)
    C, H, W = X.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1
    out = np.zeros((C, H_out, W_out))
    for c in range(C):
        for i in range(H_out):
            for j in range(W_out):
                window = X[c, i*stride:i*stride+pool_size, j*stride:j*stride+pool_size]
                if pool_type == "max":
                    out[c, i, j] = np.max(window)
                else:
                    out[c, i, j] = np.mean(window)
    return [[[round(float(out[c][i][j]), 4) for j in range(W_out)] for i in range(H_out)] for c in range(C)]
