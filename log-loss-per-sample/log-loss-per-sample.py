import math

def log_loss(y_true, y_pred, eps=1e-15):
    """
    Compute per-sample log loss.
    """
    p_hat = [0.0] * len(y_pred)

    for i, p in enumerate(y_pred):
        if p < eps:
            p_hat[i] = eps
        elif p > 1 - eps:
            p_hat[i] = 1 - eps
        else:
            p_hat[i] = p

    L = []

    for y, p in zip(y_true, p_hat):
        loss = - (y * math.log(p) + (1 - y) * math.log(1 - p))
        L.append(loss)

    return L
        