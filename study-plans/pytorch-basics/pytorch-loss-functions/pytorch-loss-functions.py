import torch
import torch.nn.functional as F
def compute_loss(pred, target, method, delta=1.0):
    """
    Returns: float, the mean loss value
    """
    def mse(pred, target):
        n = len(target)
        return ((pred - target.float())**2).mean().item()

    def cross_entropy(pred, target):
        log_probs = F.log_softmax(pred, dim=-1)
        return -log_probs[range(len(target)), target].mean().item()

    def huber(pred, target, delta):
        diff = (pred - target).abs()
        loss = torch.where(diff <= delta, 0.5 * diff ** 2, delta * (diff - 0.5*delta))
        return loss.mean().item()

    target = torch.tensor(target)
    pred = torch.tensor(pred, dtype=torch.float32)
    match method:
        case "mse":
            return mse(pred, target)
        case "cross_entropy":
            return cross_entropy(pred, target)
        case "huber":
            return huber(pred, target, delta)
        
