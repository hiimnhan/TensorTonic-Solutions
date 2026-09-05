import numpy as np

def layer_normalization(x, gamma, beta, eps=1e-5, mode="forward", d_output=None):
    """
    Returns: Dict with "output", "mean", "var", "x_hat", and optionally "dx", "dgamma", "dbeta".
    """
    x = np.array(x, dtype=float)
    gamma = np.array(gamma, dtype=float)
    beta = np.array(beta, dtype=float)

    N, D = x.shape
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    std = np.sqrt(var + eps)
    x_hat = (x - mu) / std
    out = gamma * x_hat + beta

    def r4(arr):
        if arr.ndim == 1:
            return [round(float(v), 4) for v in arr]
        return [[round(float(v), 4) for v in row] for row in arr]

    result = {
        "output": r4(out),
        "mean": r4(mu.squeeze(-1)),
        "var": r4(var.squeeze(-1)),
        "x_hat": r4(x_hat)
    }

    if mode == "backward" and d_output is not None:
        d_out = np.array(d_output, dtype=float)
        dgamma = np.sum(d_out * x_hat, axis=0)
        dbeta = np.sum(d_out, axis=0)
        dx_hat = d_out * gamma
        dx_hat_mean = np.mean(dx_hat, axis=-1, keepdims=True)
        dx_hat_xhat_mean = np.mean(dx_hat * x_hat, axis=-1, keepdims=True)
        dx = (1.0 / std) * (dx_hat - dx_hat_mean - x_hat * dx_hat_xhat_mean)
        result["dx"] = r4(dx)
        result["dgamma"] = r4(dgamma)
        result["dbeta"] = r4(dbeta)

    return result
