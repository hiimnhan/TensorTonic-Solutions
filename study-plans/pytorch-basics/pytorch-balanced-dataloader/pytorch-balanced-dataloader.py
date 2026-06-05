import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

def create_balanced_loader(features, labels, batch_size):
    """
    Returns: a DataLoader that oversamples underrepresented classes
    """
    class_count = torch.bincount(labels)
    class_weights = 1.0 / class_count.float()
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels), replacement=True)
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)
