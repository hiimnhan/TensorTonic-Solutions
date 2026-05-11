import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    """
    Returns: two-layer MLP output (linear -> ReLU -> linear)
    """

    def __init__(self, in_features, hidden_size, out_features):
        super().__init__()
        self.ln1 = nn.Linear(in_features, hidden_size)
        self.relu = nn.ReLU()
        self.ln2 = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = self.ln1(x)
        x = self.relu(x)
        x = self.ln2(x)
        return x
