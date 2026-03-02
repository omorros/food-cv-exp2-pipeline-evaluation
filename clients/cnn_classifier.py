"""
CNN classifier factory for Pipeline C.

Provides a pluggable interface so the CNN architecture winner from Experiment 1
can be swapped in via config. Pipeline C code never changes.

Supported formats:
    - .keras  (TensorFlow/Keras — Experiment 1 native format)
    - .pth    (PyTorch — alternative if weights are converted)

Supported architectures (PyTorch path only):
    - efficientnet  (EfficientNet-B0, transfer learning)
    - resnet        (ResNet-18, transfer learning)
    - custom        (user-defined CNN)

Usage:
    classifier = create_cnn_classifier()
    label = classifier.predict(crop)
    labels = classifier.predict_batch([crop1, crop2, crop3])
"""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
from PIL import Image

from config import CONFIG, CLASSES, NUM_CLASSES


# =============================================================================
# Abstract base
# =============================================================================

class BaseCNNClassifier(ABC):
    """Abstract base for CNN classifiers."""

    @abstractmethod
    def predict(self, crop: Image.Image) -> str:
        """Classify a single crop. Returns class name string."""
        ...

    @abstractmethod
    def predict_batch(self, crops: List[Image.Image]) -> List[str]:
        """Classify a batch of crops. Returns list of class name strings."""
        ...


# =============================================================================
# Keras implementation (Experiment 1 native format)
# =============================================================================

class KerasClassifier(BaseCNNClassifier):
    """
    Loads a .keras model directly from Experiment 1.

    Architecture (as trained):
        EfficientNetB0(include_top=False) -> GlobalAvgPool -> BN -> Dropout -> Dense(14)

    Preprocessing:
        tf.keras.applications.efficientnet.preprocess_input (scales to [-1, 1])
    """

    def __init__(self, weights_path: str):
        import tensorflow as tf
        # Suppress TF info/warning logs (YOLO already prints enough)
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        tf.get_logger().setLevel("ERROR")

        self._tf = tf
        self._preprocess = tf.keras.applications.efficientnet.preprocess_input
        self.model = tf.keras.models.load_model(weights_path)
        self._img_size = CONFIG.cnn_img_size  # 224

    def _prepare(self, crop: Image.Image) -> np.ndarray:
        """Resize and preprocess a single PIL image."""
        img = crop.convert("RGB").resize(
            (self._img_size, self._img_size), Image.LANCZOS
        )
        arr = np.array(img, dtype=np.float32)  # (224, 224, 3)
        return self._preprocess(arr)

    def predict(self, crop: Image.Image) -> str:
        batch = np.expand_dims(self._prepare(crop), axis=0)  # (1, 224, 224, 3)
        probs = self.model(batch, training=False)
        idx = int(self._tf.argmax(probs, axis=1).numpy()[0])
        return CLASSES[idx]

    def predict_batch(self, crops: List[Image.Image]) -> List[str]:
        if not crops:
            return []
        batch = np.stack([self._prepare(c) for c in crops])  # (N, 224, 224, 3)
        probs = self.model(batch, training=False)
        indices = self._tf.argmax(probs, axis=1).numpy().tolist()
        return [CLASSES[i] for i in indices]


# =============================================================================
# PyTorch implementations (kept for flexibility)
# =============================================================================

class _PyTorchBaseClassifier(BaseCNNClassifier):
    """Base for PyTorch-based classifiers."""

    def __init__(self, weights_path: str, device: str | None = None):
        import torch
        import torch.nn as nn
        from torchvision import transforms, models

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((CONFIG.cnn_img_size, CONFIG.cnn_img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.model = self._build_model()
        self._load_weights(weights_path)
        self.model.to(self.device)
        self.model.eval()

    @abstractmethod
    def _build_model(self):
        ...

    def _load_weights(self, weights_path: str):
        state = self._torch.load(weights_path, map_location=self.device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        self.model.load_state_dict(state)

    def predict(self, crop: Image.Image) -> str:
        with self._torch.no_grad():
            tensor = self.transform(crop.convert("RGB")).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            idx = logits.argmax(dim=1).item()
            return CLASSES[idx]

    def predict_batch(self, crops: List[Image.Image]) -> List[str]:
        if not crops:
            return []
        with self._torch.no_grad():
            tensors = self._torch.stack([
                self.transform(c.convert("RGB")) for c in crops
            ]).to(self.device)
            logits = self.model(tensors)
            indices = logits.argmax(dim=1).tolist()
            return [CLASSES[i] for i in indices]


class EfficientNetClassifier(_PyTorchBaseClassifier):
    """EfficientNet-B0 with replaced head for 14 classes (PyTorch)."""

    def _build_model(self):
        from torchvision import models
        import torch.nn as nn
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        return model


class ResNetClassifier(_PyTorchBaseClassifier):
    """ResNet-18 with replaced head for 14 classes (PyTorch)."""

    def _build_model(self):
        from torchvision import models
        import torch.nn as nn
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        return model


class CustomCNNClassifier(_PyTorchBaseClassifier):
    """Lightweight custom CNN for 14 classes (PyTorch)."""

    def _build_model(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, NUM_CLASSES),
        )


# =============================================================================
# Factory
# =============================================================================

_PYTORCH_REGISTRY = {
    "efficientnet": EfficientNetClassifier,
    "resnet": ResNetClassifier,
    "custom": CustomCNNClassifier,
}

# Singleton
_classifier: BaseCNNClassifier | None = None


def create_cnn_classifier(
    model_name: str | None = None,
    weights_path: str | None = None,
) -> BaseCNNClassifier:
    """
    Factory: create or return a CNN classifier.

    Auto-detects format from file extension:
        - .keras → loads as TensorFlow/Keras model (Experiment 1 native)
        - .pth   → loads as PyTorch model

    Args:
        model_name: "efficientnet" | "resnet" | "custom" (only for .pth files).
        weights_path: Path to weights file (default: CONFIG.cnn_weights).

    Returns:
        BaseCNNClassifier instance (singleton).
    """
    global _classifier
    if _classifier is not None:
        return _classifier

    weights_path = weights_path or CONFIG.cnn_weights

    if weights_path.endswith(".keras"):
        _classifier = KerasClassifier(weights_path=weights_path)
    else:
        model_name = model_name or CONFIG.cnn_model_name
        cls = _PYTORCH_REGISTRY.get(model_name)
        if cls is None:
            raise ValueError(
                f"Unknown CNN model '{model_name}'. Available: {list(_PYTORCH_REGISTRY.keys())}"
            )
        _classifier = cls(weights_path=weights_path)

    return _classifier


def warmup():
    """Pre-load the CNN model for timing fairness."""
    create_cnn_classifier()
