import numpy as np

def loss_functions(y_true, y_pred, loss_type):
    """
    Returns: Loss value as a float, rounded to 4 decimal places.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    match loss_type:
        case "mse":
            l = np.mean((y_true - y_pred)** 2)
            return float(round(l, 4))
        case "bce":
            eps = 1e-15
            y_pred = np.clip(y_pred, eps, 1 - eps)
            l = y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
            
            l = -np.mean(l)
            return float(round(l, 4))
        case "cce":
            y_true = np.asarray(y_true, dtype=int)
            n = len(y_true)
            losses = []
            for i in range(n):
                z = y_pred[i]
                max_z = np.max(z)
                logsumexp = max_z + np.log(np.sum(np.exp(z - max_z)))
                losses.append(-(z[y_true[i]] - logsumexp))
            return float(round(np.mean(losses), 4))
        case "hinge":
            return float(round(np.mean(np.maximum(0, 1 - y_true * y_pred)), 4))