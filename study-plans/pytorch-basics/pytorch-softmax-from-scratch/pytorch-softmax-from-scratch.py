import torch

def softmax(logits):
    """
    Returns: tensor of same shape with softmax probabilities (each row sums to 1)
    """
    N, C = logits.shape
    m = torch.max(logits, dim=1, keepdim=True).values

    shifted = logits - m 
    exps = torch.exp(shifted)
    return exps / exps.sum(dim=1, keepdim=True)
