import torch

def batch_norm(X, gamma, beta, eps=1e-5):
    """
    Returns: tensor of shape (N, D), the batch-normalized output
    X (N,D)
    gamma(D,)
    beta (D,)
    """
    mean = X.mean(dim=0)
    var = X.var(dim=0, unbiased=False)
    X_hat = (X - mean) / torch.sqrt(var + eps)

    return gamma * X_hat + beta
    
