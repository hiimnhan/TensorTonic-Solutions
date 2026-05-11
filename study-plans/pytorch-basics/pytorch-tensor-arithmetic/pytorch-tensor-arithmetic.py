import torch

def tensor_op(x, y, op):
    """
    Returns: list (result tensor converted via .tolist())
    """
    a = torch.tensor(x, dtype=torch.float32)
    b = torch.tensor(y, dtype=torch.float32)
    match op:
        case "add":
            res = a + b
        case "multiply":
            res = a * b
        case "matmul":
            res = a @ b
        case "power":
            res = a ** b
        case "max":
            res = torch.maximum(a, b)

    return res.tolist()