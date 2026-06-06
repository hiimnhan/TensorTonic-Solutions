import numpy as np

def mini_batch_training(X, y, weights, biases, lr, epochs, batch_size):
    """
    Returns: list of floats
    """
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    W = [np.array(w, dtype=float) for w in weights]
    b = [np.array(bi, dtype=float) for bi in biases]
    N = len(X)
    L = len(W)
    losses = []
    for _ in range(epochs):
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            inputs, targets = X[start:end], y[start:end]
            mini_batch_size = end - start
            dW = [np.zeros_like(w) for w in W]
            db = [np.zeros_like(bi) for bi in b]

            for i in range(mini_batch_size):
                a = inputs[i]
                activations = [a]
                pre_activations = []
                for l in range(L):
                    z = W[l] @ a + b[l]
                    pre_activations.append(z)
                    a = np.maximum(0, z) if l < L - 1 else z # ReLU only hidden layers
                    activations.append(a)
                delta = activations[-1] - targets[i]
                for l in range(L - 1, -1, -1):
                    dW[l] += np.outer(delta, activations[l])
                    db[l] += delta
                    if l > 0:
                        delta = (W[l].T @ delta) * (pre_activations[l - 1] > 0).astype(float)
            for l in range(L):
                W[l] -= lr * dW[l] / mini_batch_size
                b[l] -= lr * db[l] / mini_batch_size
        loss = 0.0
        for i in range(N):
            a = X[i]
            for l in range(L):
                z = W[l] @ a + b[l]
                a = np.maximum(0, z) if l < L - 1 else z
            loss += 0.5 * np.sum((a - y[i]) ** 2)
        losses.append(round(loss / N, 4))
    return losses
            
                    