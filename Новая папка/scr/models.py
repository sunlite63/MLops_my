"""Neural network architectures for classification tasks."""

import torch
from torch import nn


class DenseBlock(nn.Module):
    """
    Fully connected layer with ReLU activation.
    Used as a building block for larger networks.
    """
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.activation(x)
        return self.dropout(x)
    
    def __repr__(self) -> str:
        return f"DenseBlock(in={self.fc.in_features}, out={self.fc.out_features})"


class SimpleClassifier(nn.Module):
    """
    Binary classifier with one hidden layer.
    Architecture: Input -> DenseBlock -> Linear -> Logits
    """
    
    def __init__(
        self,
        input_features: int,
        hidden_size: int = 16,
        num_classes: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        self.hidden = DenseBlock(input_features, hidden_size, dropout)
        self.output = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through hidden layer and output head."""
        embeddings = self.hidden(x)
        return self.output(embeddings)
    
    def __repr__(self) -> str:
        return f"SimpleClassifier({self.hidden}, classes={self.output.out_features})"