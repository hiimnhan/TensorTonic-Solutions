import torch

def rms_norm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Returns: Normalized tensor of same shape as x
    """
    nom = x * gamma
    denom = torch.sqrt(torch.mean(x**2, axis=-1, keepdim=True) + eps)

    return nom / denom