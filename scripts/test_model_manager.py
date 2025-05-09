import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.model_manager import ModelManager
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import logging

def create_dummy_data():
    """Create dummy data for testing."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(100, 10))
    y = pd.Series(np.random.randint(0, 3, 100))  # 3 classes: buy, hold, sell
    return X, y

def create_dummy_model(model_type: str):
    """Create a dummy model for testing."""
    if model_type == "torch":
        from models.architectures import TimeSeriesCNN
        return TimeSeriesCNN(input_size=10, num_classes=3)
    else:
        return RandomForestClassifier(n_estimators=10, random_state=42)

def test_model_manager():
    # Initialize model manager
    mm = ModelManager()
    
    # Create dummy data
    X, y = create_dummy_data()
    
    try:
        # Test saving and loading models
        print("\nTesting model saving and loading...")
        
        # Test sklearn model
        sklearn_model = create_dummy_model("sklearn")
        sklearn_model.fit(X, y)
        version = mm.save_model("value", sklearn_model)
        print(f"Saved sklearn model version: {version}")
        
        loaded_model, loaded_version = mm.load_model("value", version)
        print(f"Loaded sklearn model version: {loaded_version}")
        
        # Test PyTorch model
        torch_model = create_dummy_model("torch")
        version = mm.save_model("cnn", torch_model)
        print(f"Saved PyTorch model version: {version}")
        
        loaded_model, loaded_version = mm.load_model("cnn", version)
        print(f"Loaded PyTorch model version: {loaded_version}")
        
        # Test model evaluation
        print("\nTesting model evaluation...")
        metrics = mm.evaluate_model("value", version, X, y)
        print("Evaluation metrics:", metrics)
        
        # Test weight updating
        print("\nTesting weight updating...")
        performance_metrics = {
            "value": {"f1": 0.8},
            "sentiment": {"f1": 0.7},
            "technical": {"f1": 0.6},
            "insider": {"f1": 0.5},
            "macro": {"f1": 0.4},
            "cnn": {"f1": 0.9}
        }
        mm.update_weights(performance_metrics)
        print("Updated weights:", mm.model_weights)
        
        # Test model history
        print("\nTesting model history...")
        history = mm.get_model_history("value")
        print(f"Model history entries: {len(history)}")
        
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_model_manager() 