import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    x = np.array(x, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta = np.array(beta, dtype=float)
    if x.ndim == 2:
        miu = x.mean(axis=0, keepdims=True)
        variance = x.var(axis=0, keepdims=True)
        x_hat = (x - miu) / np.sqrt(variance + eps)
        return x_hat * gamma[None, :] + beta[None, :]

    elif x.ndim == 4:
        miu = x.mean(axis=(0, 2, 3), keepdims=True)
        variance = x.var(axis=(0, 2, 3), keepdims=True)
        x_hat = (x - miu) / np.sqrt(variance + eps)
        return x_hat * gamma[None, :, None, None] + beta[None, :, None, None]

    else:
        raise ValueError("")
