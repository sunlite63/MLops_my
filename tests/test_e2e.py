"""End-to-end integration tests."""

import pytest
import torch

from src.dataset import build_dataloader, SyntheticDataset
from src.models import SimpleClassifier
from src.trainer import Trainer


@pytest.mark.e2e
def test_full_training_pipeline():
    """
    End-to-end test: data -> training -> evaluation -> inference.
    Verifies the complete ML workflow produces reasonable results.
    """
    # Setup
    dataset = SyntheticDataset(n_samples=200, n_features=8, random_seed=42)
    loader = build_dataloader(dataset, batch_size=16, shuffle=True)
    
    model = SimpleClassifier(input_features=8, hidden_size=32, num_classes=2)
    trainer = Trainer(model=model, learning_rate=0.01)
    
    # Train
    results = trainer.train_loop(loader, num_epochs=10)
    
    # Verify training progressed
    assert results[-1].loss < results[0].loss, "Loss should decrease during training"
    
    # Evaluate
    eval_result = trainer.validate(loader)
    
    # Model should achieve reasonable accuracy on linearly separable data
    assert eval_result.accuracy > 0.8, f"Accuracy too low: {eval_result.accuracy:.2%}"
    
    # Inference check
    model.eval()
    with torch.no_grad():
        x_sample, y_sample = dataset[0]
        prediction = model(x_sample.unsqueeze(0))
        predicted_class = prediction.argmax(dim=1).item()
        
        assert predicted_class in [0, 1]
        assert prediction.shape == (1, 2)