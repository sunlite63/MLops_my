"""Dataset utilities for synthetic data generation and loading."""

from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split

# Type alias for cleaner signatures
DataPoint = Tuple[torch.Tensor, torch.Tensor]


class SyntheticDataset(Dataset[DataPoint]):
    """
    Generates a reproducible synthetic dataset for binary classification.
    Labels are determined by whether the sum of features exceeds a threshold.
    """
    
    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 4,
        random_seed: int = 42
    ) -> None:
        torch.manual_seed(random_seed)
        
        self.features = torch.randn(n_samples, n_features)
        # Labels: 1 if sum > 0, else 0
        self.labels = (self.features.sum(dim=1) > 0).long()
    
    def __len__(self) -> int:
        return self.features.shape[0]
    
    def __getitem__(self, idx: int) -> DataPoint:
        return self.features[idx], self.labels[idx]
    
    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42
    ) -> Tuple["SyntheticDataset", "SyntheticDataset"]:
        """Split dataset into train and validation subsets."""
        generator = torch.Generator().manual_seed(seed)
        total = len(self)
        train_size = int(train_ratio * total)
        val_size = total - train_size
        
        train_subset, val_subset = random_split(
            self, [train_size, val_size], generator=generator
        )
        return train_subset, val_subset


def build_dataloader(
    dataset: Dataset[DataPoint],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader[DataPoint]:
    """
    Create a DataLoader with sensible defaults.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False
    )