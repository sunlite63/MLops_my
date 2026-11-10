"""PyTorch Lightning training with MLflow autologging."""

import io
import sys

import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
import torch
from torch import nn

from src.dataset import build_dataloader
from src.dataset import SyntheticDataset
from src.models import SimpleClassifier

# Fix encoding for Windows terminals
for stream in (sys.stdout, sys.stderr):
    if isinstance(stream, io.TextIOWrapper):
        stream.reconfigure(encoding="utf-8")

# Experiment configuration
TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "pytorch_lightning_autolog"


class LightningClassifier(pl.LightningModule):
    """
    PyTorch Lightning wrapper for SimpleClassifier.
    Automatically logs hyperparameters and metrics via MLflow autolog.
    """
    
    def __init__(
        self,
        input_dim: int = 8,
        hidden_dim: int = 32,
        output_dim: int = 2,
        learning_rate: float = 0.01
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.network = SimpleClassifier(
            input_features=input_dim,
            hidden_size=hidden_dim,
            num_classes=output_dim
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    
    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        
        # Compute accuracy for logging
        predictions = logits.argmax(dim=1)
        accuracy = (predictions == y).float().mean()
        
        # Log metrics (automatically captured by MLflow)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_accuracy", accuracy, prog_bar=True, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


def run_lightning_experiment(
    lr: float = 0.01,
    max_epochs: int = 5,
    seed: int = 42
) -> None:
    """
    Run training with PyTorch Lightning and MLflow autologging.
    No manual mlflow.log_* calls needed.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.pytorch.autolog(log_models=True, checkpoint=False)
    
    # Reproducibility
    pl.seed_everything(seed, verbose=False)
    
    # Data
    dataset = SyntheticDataset(n_samples=200, n_features=8, random_seed=seed)
    dataloader = build_dataloader(dataset, batch_size=16)
    
    # Model
    model = LightningClassifier(
        input_dim=8,
        hidden_dim=32,
        output_dim=2,
        learning_rate=lr
    )
    
    # Trainer configuration
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=False,  # MLflow handles logging
        enable_checkpointing=False,
        default_root_dir="output/lightning",
        log_every_n_steps=1,
        accelerator="cpu",
        devices=1
    )
    
    # Run training
    with mlflow.start_run(run_name=f"lightning_lr_{lr}"):
        trainer.fit(model, dataloader)
    
    print(f"Lightning training completed for lr={lr}")


if __name__ == "__main__":
    print("=" * 50)
    print("Autolog Experiment 1: lr=0.02")
    print("=" * 50)
    run_lightning_experiment(lr=0.02, max_epochs=5)
    
    print("\n" + "=" * 50)
    print("Autolog Experiment 2: lr=0.002")
    print("=" * 50)
    run_lightning_experiment(lr=0.002, max_epochs=5)