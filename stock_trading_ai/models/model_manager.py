"""Model management for stock trading AI."""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import joblib

from ..config import Config
from ..utils.error_handling import ModelError
from .architectures import create_model_architecture

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/model_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages model creation, saving, loading, and evaluation."""

    def __init__(self, config: Config):
        """Initialize the model manager.
        
        Args:
            config: Configuration instance
        """
        self.config = config
        self.models_dir = Path(config.get("model.save_dir", "models/saved"))
        self.versions_dir = self.models_dir / "versions"
        self.metrics_dir = self.models_dir / "metrics"
        self.weights_file = self.models_dir / "model_weights.json"
        
        # Create necessary directories
        for directory in [self.versions_dir, self.metrics_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize or load model weights
        self.model_weights = self._load_weights()
        
        # Define model types and their evaluation metrics
        self.model_types = {
            "cnn": {
                "type": "torch",
                "metrics": ["accuracy", "precision", "recall", "f1"]
            },
            "value": {
                "type": "sklearn",
                "metrics": ["accuracy", "precision", "recall", "f1"]
            },
            "sentiment": {
                "type": "sklearn",
                "metrics": ["accuracy", "precision", "recall", "f1"]
            },
            "technical": {
                "type": "sklearn",
                "metrics": ["accuracy", "precision", "recall", "f1"]
            },
            "insider": {
                "type": "sklearn",
                "metrics": ["accuracy", "precision", "recall", "f1"]
            },
            "macro": {
                "type": "sklearn",
                "metrics": ["accuracy", "precision", "recall", "f1"]
            }
        }

        self.model = None
        self.model_history = {}

    def _load_weights(self) -> Dict:
        """Load model weights from file."""
        if self.weights_file.exists():
            with open(self.weights_file, 'r') as f:
                return json.load(f)
        return {
            "cnn": 0.25,
            "value": 0.2,
            "sentiment": 0.15,
            "technical": 0.15,
            "insider": 0.15,
            "macro": 0.1
        }

    def _save_weights(self):
        """Save model weights to file."""
        with open(self.weights_file, 'w') as f:
            json.dump(self.model_weights, f, indent=4)

    def create_model(self, model_name: str, **kwargs) -> None:
        """Create a new model.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Additional arguments for model creation
        
        Raises:
            ModelError: If model creation fails
        """
        try:
            self.model = create_model_architecture(model_name, **kwargs)
            self.model_history = {
                "train_loss": [],
                "val_loss": [],
                "metrics": {}
            }
        except Exception as e:
            raise ModelError(f"Failed to create model {model_name}: {str(e)}")

    def save_model(self, model_path: str) -> None:
        """Save the current model.
        
        Args:
            model_path: Path to save the model
        
        Raises:
            ModelError: If model saving fails
        """
        if self.model is None:
            raise ModelError("No model to save")

        try:
            save_path = self.models_dir / model_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model state
            torch.save({
                "model_state": self.model.state_dict(),
                "history": self.model_history
            }, save_path)
        except Exception as e:
            raise ModelError(f"Failed to save model to {model_path}: {str(e)}")

    def load_model(self, model_path: str, model_name: str, **kwargs) -> None:
        """Load a saved model.
        
        Args:
            model_path: Path to the saved model
            model_name: Name of the model architecture
            **kwargs: Additional arguments for model creation
        
        Raises:
            ModelError: If model loading fails
        """
        try:
            load_path = self.models_dir / model_path
            if not load_path.exists():
                raise ModelError(f"Model file not found: {model_path}")

            # Create model architecture
            self.model = create_model_architecture(model_name, **kwargs)
            
            # Load state
            checkpoint = torch.load(load_path)
            self.model.load_state_dict(checkpoint["model_state"])
            self.model_history = checkpoint.get("history", {})
        except Exception as e:
            raise ModelError(f"Failed to load model from {model_path}: {str(e)}")

    def evaluate_model(self, data: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Evaluate the current model.
        
        Args:
            data: Input data
            labels: True labels
        
        Returns:
            Dictionary of evaluation metrics
        
        Raises:
            ModelError: If evaluation fails
        """
        if self.model is None:
            raise ModelError("No model to evaluate")

        try:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(data)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == labels).float().mean()

            return {
                "loss": loss.item(),
                "accuracy": accuracy.item()
            }
        except Exception as e:
            raise ModelError(f"Model evaluation failed: {str(e)}")

    def update_history(self, metrics: Dict[str, float]) -> None:
        """Update model training history.
        
        Args:
            metrics: Dictionary of metrics to update
        """
        for key, value in metrics.items():
            if key not in self.model_history:
                self.model_history[key] = []
            self.model_history[key].append(value)

    def get_history(self) -> Dict[str, List[float]]:
        """Get model training history.
        
        Returns:
            Dictionary of training history
        """
        return self.model_history

    def process_batch(self, batch_data: torch.Tensor, batch_size: Optional[int] = None) -> torch.Tensor:
        """Process a batch of data through the model.
        
        Args:
            batch_data: Input batch data
            batch_size: Optional batch size override
        
        Returns:
            Model outputs
        
        Raises:
            ModelError: If batch processing fails
        """
        if self.model is None:
            raise ModelError("No model to process batch")

        try:
            if batch_size is None:
                batch_size = self.config.get("model.batch_size", 32)

            self.model.eval()
            with torch.no_grad():
                return self.model(batch_data)
        except Exception as e:
            raise ModelError(f"Batch processing failed: {str(e)}")

    def cleanup(self) -> None:
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
            self.model_history = {}

    def save_model_version(self, model_name: str, model, version: Optional[str] = None) -> str:
        """
        Save a model with versioning.
        
        Args:
            model_name: Name of the model (e.g., 'cnn', 'value')
            model: The model object to save
            version: Optional version string. If None, generates timestamp-based version
            
        Returns:
            Version string of the saved model
        """
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_dir = self.versions_dir / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            if self.model_types[model_name]["type"] == "torch":
                torch.save(model.state_dict(), model_dir / "model.pth")
            else:
                joblib.dump(model, model_dir / "model.joblib")
            
            # Save metadata
            metadata = {
                "version": version,
                "timestamp": datetime.now().isoformat(),
                "type": self.model_types[model_name]["type"]
            }
            
            with open(model_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Saved {model_name} model version {version}")
            return version
            
        except Exception as e:
            logger.error(f"Error saving model {model_name}: {str(e)}")
            raise

    def load_model_version(self, model_name: str, version: Optional[str] = None) -> Tuple[object, str]:
        """
        Load a model by name and optional version.
        
        Args:
            model_name: Name of the model to load
            version: Optional version string. If None, loads latest version
            
        Returns:
            Tuple of (model, version_string)
        """
        model_versions_dir = self.versions_dir / model_name
        
        if not model_versions_dir.exists():
            raise ValueError(f"No versions found for model {model_name}")
        
        if version is None:
            # Get latest version
            versions = sorted([d.name for d in model_versions_dir.iterdir() if d.is_dir()])
            if not versions:
                raise ValueError(f"No versions found for model {model_name}")
            version = versions[-1]
        
        model_dir = model_versions_dir / version
        
        try:
            if self.model_types[model_name]["type"] == "torch":
                model = create_model_architecture(model_name)
                model.load_state_dict(torch.load(model_dir / "model.pth"))
            else:
                model = joblib.load(model_dir / "model.joblib")
            
            logger.info(f"Loaded {model_name} model version {version}")
            return model, version
            
        except Exception as e:
            logger.error(f"Error loading model {model_name} version {version}: {str(e)}")
            raise

    def evaluate_model_version(self, model_name: str, version: str, 
                              X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate a model's performance.
        
        Args:
            model_name: Name of the model to evaluate
            version: Version of the model to evaluate
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        model, _ = self.load_model_version(model_name, version)
        
        try:
            if self.model_types[model_name]["type"] == "torch":
                model.eval()
                with torch.no_grad():
                    y_pred = model(torch.FloatTensor(X_test.values)).argmax(dim=1).numpy()
            else:
                y_pred = model.predict(X_test)
            
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average='weighted'),
                "recall": recall_score(y_test, y_pred, average='weighted'),
                "f1": f1_score(y_test, y_pred, average='weighted')
            }
            
            # Save metrics
            metrics_dir = self.metrics_dir / model_name
            metrics_dir.mkdir(parents=True, exist_ok=True)
            
            metrics_file = metrics_dir / f"{version}_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            logger.info(f"Evaluated {model_name} model version {version}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_name} version {version}: {str(e)}")
            raise

    def update_weights(self, performance_metrics: Dict[str, Dict[str, float]]):
        """
        Update model weights based on recent performance.
        
        Args:
            performance_metrics: Dictionary of model performance metrics
        """
        try:
            # Calculate new weights based on F1 scores
            f1_scores = {model: metrics['f1'] for model, metrics in performance_metrics.items()}
            total_f1 = sum(f1_scores.values())
            
            if total_f1 > 0:
                new_weights = {model: score/total_f1 for model, score in f1_scores.items()}
                
                # Smooth the weight update (70% new, 30% old)
                self.model_weights = {
                    model: 0.7 * new_weights.get(model, 0) + 0.3 * self.model_weights.get(model, 0)
                    for model in self.model_weights.keys()
                }
                
                # Normalize weights
                total = sum(self.model_weights.values())
                self.model_weights = {k: v/total for k, v in self.model_weights.items()}
                
                self._save_weights()
                logger.info("Updated model weights based on performance")
            
        except Exception as e:
            logger.error(f"Error updating model weights: {str(e)}")
            raise

    def get_model_history(self, model_name: str) -> List[Dict]:
        """
        Get version history and performance metrics for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of dictionaries containing version info and metrics
        """
        history = []
        model_versions_dir = self.versions_dir / model_name
        metrics_dir = self.metrics_dir / model_name
        
        if not model_versions_dir.exists():
            return history
        
        for version_dir in model_versions_dir.iterdir():
            if not version_dir.is_dir():
                continue
                
            version_info = {
                "version": version_dir.name,
                "metrics": {}
            }
            
            # Load metadata
            metadata_file = version_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    version_info.update(json.load(f))
            
            # Load metrics
            metrics_file = metrics_dir / f"{version_dir.name}_metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    version_info["metrics"] = json.load(f)
            
            history.append(version_info)
        
        return sorted(history, key=lambda x: x["timestamp"] if "timestamp" in x else "") 