"""Stage 1 of the DVC pipeline: synthetic data generation."""

import csv
import random
from pathlib import Path

# Constants
SEED = 42
SAMPLES = 100
TARGET_FILE = Path("data/raw_data.csv")

def generate_dataset() -> None:
    """Create a simple linearly separable dataset and save it to CSV."""
    TARGET_FILE.parent.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)

    with open(TARGET_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["feature_a", "feature_b", "label"])
        
        for _ in range(SAMPLES):
            a = round(random.uniform(0, 10), 2)
            b = round(random.uniform(0, 10), 2)
            label = 1 if (a + b) > 10 else 0
            writer.writerow([a, b, label])
            
    print(f"[DVC] Generated {SAMPLES} samples in {TARGET_FILE}")

if __name__ == "__main__":
    generate_dataset()