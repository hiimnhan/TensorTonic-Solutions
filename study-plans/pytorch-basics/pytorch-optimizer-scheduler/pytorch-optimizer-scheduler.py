import torch
import torch.nn as nn

def train_with_scheduler(model, dataloader, criterion, optimizer, scheduler, num_epochs):
    """
    Returns: dict with 'losses' (list of per-epoch avg loss) and 'lrs' (list of learning rate per epoch)
    """
    losses = []
    lrs = []

    for epoch in range(num_epochs):
        model.train()
        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)

        total_loss = 0.0
        num_batches = 0

        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        losses.append(total_loss / num_batches)
        scheduler.step()

    return {
        "losses": losses,
        "lrs": lrs
    }
        
