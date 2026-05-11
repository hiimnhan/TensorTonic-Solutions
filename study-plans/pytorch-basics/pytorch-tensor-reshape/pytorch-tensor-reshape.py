import torch

def reshape_tensor(x, op):
    """
    Returns: list
    """
    x = torch.tensor(x, dtype=torch.float32)
    match op:
        case "flatten":
            res = x.flatten()
        case "squeeze":
            res = x.squeeze()
        case "transpose":
            res = x.T

    return res.tolist()
            
