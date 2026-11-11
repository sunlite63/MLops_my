"""Manual MLflow tracking for PyTorch training experiments."""

import io
import sys
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch

from src.dataset import build_dataloader
from src.dataset import SyntheticDataset
from src.models import SimpleClassifier
from src.trainer import Trainer

# Fix Windows console encoding for MLflow emojis
for stream in (sys.stdout, sys.stderr):
    if isinstance(stream, io.TextIOWrapper):
        stream.reconfigure(encoding="utf-8")

# Configuration
MLFLOW_URI = "http://localhost:5000"
EXPERIMENT = "pytorch_classifier_manual"
N_FEATURES = 8
N_CLASSES = 2


def run_experiment(
    learning_rate: float = 0.01,
    n_epochs: int = 10,
    batch_size: int = 16,
    hidden_dim: int = 32,
    seed: int = 42
) -> None:
    """
    Run a single training experiment with MLflow tracking.
    Logs hyperparameters, metrics, and model artifacts.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    
    run_name = f"lr_{learning_rate}_epochs_{n_epochs}"
    
    with mlflow.start_run(run_name=run_name):
        # Log configuration
        mlflow.log_params({
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            "seed": seed,
            "n_features": N_FEATURES,
            "n_classes": N_CLASSES
        })
        
        mlflow.set_tags({
            "experiment_type": "binary_classification",
            "framework": "pytorch",
            "model_architecture": "simple_classifier"
        })
        
        # Setup data and model
        torch.manual_seed(seed)
        dataset = SyntheticDataset(
            n_samples=200,
            n_features=N_FEATURES,
            random_seed=seed
        )
        loader = build_dataloader(dataset, batch_size=batch_size)
        
        model = SimpleClassifier(
            input_features=N_FEATURES,
            hidden_size=hidden_dim,
            num_classes=N_CLASSES
        )
        
        trainer = Trainer(model=model, learning_rate=learning_rate)
        
        # Training loop with epoch-level metrics
        for epoch in range(n_epochs):
            results = trainer.train_loop(loader, num_epochs=1)
            epoch_loss = results[0].loss
            
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)
            print(f"[{run_name}] Epoch {epoch+1}/{n_epochs} - Loss: {epoch_loss:.4f}")
        
        # Final evaluation
        eval_result = trainer.validate(loader)
        mlflow.log_metrics({
            "final_loss": eval_result.loss,
            "final_accuracy": eval_result.accuracy
        })
        
        # Save artifacts
        artifact_dir = Path("output")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        weights_path = artifact_dir / "model_weights.pt"
        torch.save(model.state_dict(), weights_path)
        mlflow.log_artifact(str(weights_path), artifact_path="model")
        
        # Log full MLflow model
        mlflow.pytorch.log_model(
            model,
            artifact_path="pytorch_model",
            input_example=dataset.features[:1].numpy(),
            serialization_format="pickle"
        )
        
        print(f"\nExperiment '{run_name}' completed!")
        print(f"Final accuracy: {eval_result.accuracy:.2%}")


if __name__ == "__main__":
    print("=" * 50)
    print("Running experiment 1: lr=0.05")
    print("=" * 50)
    run_experiment(learning_rate=0.05, n_epochs=10)
    
    print("\n" + "=" * 50)
    print("Running experiment 2: lr=0.005")
    print("=" * 50)
    run_experiment(learning_rate=0.005, n_epochs=10)