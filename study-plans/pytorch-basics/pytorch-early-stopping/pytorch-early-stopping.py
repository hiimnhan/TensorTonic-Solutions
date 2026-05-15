import torch
import torch.nn as nn

def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, max_epochs, patience):
    """
    Returns: dict with 'train_losses' (list), 'val_losses' (list), 'stopped_epoch' (int, 1-indexed)
    """
    train_losses = []
    val_losses = []
    patience_count = 0
    best_val_loss = float("inf")

    for epoch in range(1, max_epochs+1):
        model.train()
        total_train = 0.0
        n_train = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_train += loss.item()
            n_train += 1

        train_losses.append(total_train / n_train)

        model.eval()
        total_val = 0.0
        n_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_val += loss.item()
                n_val += 1

            val_losses.append(total_val / n_val)

            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                patience_count = 0

            else:
                patience_count += 1
                if patience_count >= patience:
                    return {
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                        "stopped_epoch": epoch
                    }

    return {
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "stopped_epoch": epoch
                }