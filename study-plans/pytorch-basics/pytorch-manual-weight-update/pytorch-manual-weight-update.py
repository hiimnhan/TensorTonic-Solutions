import torch
import torch.nn as nn

def manual_train_step(model, X, y, criterion, lr):
    """
    Returns: loss value as a Python float
    """
    outputs = model(X)
    loss = criterion(outputs, y)
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad

    model.zero_grad()

    return loss.item()
    
