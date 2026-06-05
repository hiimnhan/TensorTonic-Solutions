import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    """
    Returns: (features, label) from __getitem__ where features is float32 (D,) and label is float32 (1,)
    """

    def __init__(self, data, label_col):
        t = torch.tensor(data, dtype=torch.float32)
        self.labels = t[:, label_col].unsqueeze(1)
        feature_cols = [i for i in range(t.shape[1]) if i != label_col]
        self.features = t[:, feature_cols]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx])
