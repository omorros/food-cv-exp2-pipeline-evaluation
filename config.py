"""
Configuration and reproducibility module for Experiment 2:
End-to-End Pipeline Comparison (14-class Fruit/Veg Inventory).

This module provides:
- 14-class inventory constants
- Frozen ExperimentConfig for fair comparison
- Environment validation (API keys, dependencies)
- Structured logging setup
- Random seed control for reproducibility
- Timing utilities that exclude initialization

DISSERTATION ARTIFACT: All settings frozen for fair comparison.
"""

import os
import sys
import hashlib
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager
import time
import json

import numpy as np
from dotenv import load_dotenv

# Load environment variables at module import
load_dotenv()


# =============================================================================
# 14-CLASS INVENTORY CONSTANTS
# =============================================================================

CLASSES: tuple[str, ...] = (
    "apple", "banana", "bell_pepper_green", "bell_pepper_red",
    "carrot", "cucumber", "grape", "lemon",
    "onion", "orange", "peach", "potato",
    "strawberry", "tomato",
)

CLASS_TO_ID: Dict[str, int] = {name: idx for idx, name in enumerate(CLASSES)}
ID_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASSES)}

NUM_CLASSES: int = len(CLASSES)  # 14


# =============================================================================
# FROZEN EXPERIMENTAL SETTINGS
# =============================================================================

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Frozen configuration for reproducible experiments.
    All values locked - do not modify during experiment runs.

    Pipelines:
        A (vlm):      Image -> GPT-4o-mini (constrained to 14 labels) -> inventory
        B (yolo-14):  Image -> 14-class YOLO -> boxes + labels -> inventory
        C (yolo-cnn): Image -> 1-class objectness YOLO -> crops -> CNN -> inventory
    """
    # --- VLM Settings (Pipeline A) ---
    vlm_model: str = "gpt-4o-mini"
    vlm_temperature: float = 0.0
    vlm_image_detail: str = "high"
    vlm_max_tokens: int = 500

    # --- YOLO Settings (shared by B and C) ---
    yolo_conf_threshold: float = 0.25
    yolo_iou_threshold: float = 0.45
    yolo_max_detections: int = 30
    yolo_img_size: int = 640

    # --- YOLO 14-class (Pipeline B) ---
    yolo_14class_weights: str = "weights/yolo_14class_best.pt"

    # --- YOLO objectness (Pipeline C) ---
    yolo_objectness_weights: str = "weights/yolo_objectness_best.pt"

    # --- CNN Settings (Pipeline C) ---
    cnn_model_name: str = "efficientnet"  # efficientnet | resnet | custom
    cnn_weights: str = "weights/cnn_winner.keras"
    cnn_img_size: int = 224
    cnn_crop_padding: float = 0.10

    # --- Training defaults ---
    train_epochs: int = 100
    train_batch_size: int = 16
    train_patience: int = 15
    train_lr: float = 0.01

    # --- Random seed for reproducibility ---
    random_seed: int = 42

    # --- Paths ---
    results_dir: str = "results"
    logs_dir: str = "logs"
    weights_dir: str = "weights"


# Global config instance
CONFIG = ExperimentConfig()


# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

def validate_environment(require_openai: bool = False) -> Dict[str, Any]:
    """
    Validate all required environment variables and dependencies.

    Args:
        require_openai: If True, fail when OPENAI_API_KEY is missing (Pipeline A only).

    Returns:
        Dict with validation results and any errors.
    """
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "config": {}
    }

    # Check OpenAI API key (only required for Pipeline A)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        if require_openai:
            results["valid"] = False
            results["errors"].append(
                "OPENAI_API_KEY not found. Set it in .env file or environment. "
                "Required for Pipeline A (VLM)."
            )
        else:
            results["warnings"].append(
                "OPENAI_API_KEY not set. Pipeline A (VLM) will not work."
            )
    elif not api_key.startswith(("sk-", "sk-proj-")):
        results["warnings"].append(
            "OPENAI_API_KEY doesn't start with expected prefix (sk- or sk-proj-)"
        )

    # Record config for logging
    results["config"] = {
        "vlm_model": CONFIG.vlm_model,
        "vlm_temperature": CONFIG.vlm_temperature,
        "yolo_conf_threshold": CONFIG.yolo_conf_threshold,
        "yolo_max_detections": CONFIG.yolo_max_detections,
        "cnn_model_name": CONFIG.cnn_model_name,
        "num_classes": NUM_CLASSES,
        "random_seed": CONFIG.random_seed,
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
    }

    return results


# =============================================================================
# RANDOM SEED CONTROL
# =============================================================================

def set_reproducibility_seed(seed: Optional[int] = None) -> int:
    """
    Set random seeds for reproducibility.

    Args:
        seed: Random seed (defaults to CONFIG.random_seed)

    Returns:
        The seed that was set
    """
    seed = seed or CONFIG.random_seed

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    return seed


# =============================================================================
# STRUCTURED LOGGING
# =============================================================================

@dataclass
class PipelineLog:
    """Structured log entry for pipeline execution."""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    pipeline: str = ""
    image: str = ""
    step: str = ""
    duration_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ExperimentLogger:
    """
    Structured logger for experiment runs.
    Captures all data needed for post-hoc analysis and reproducibility.
    """

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = Path(log_dir) if log_dir else Path(CONFIG.logs_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs: list[PipelineLog] = []

        # Setup file logging
        self.log_file = self.log_dir / f"experiment_{self.session_id}.jsonl"

        # Also setup standard logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(self.log_dir / f"experiment_{self.session_id}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def log(self, entry: PipelineLog):
        """Add a log entry."""
        self.logs.append(entry)

        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                "timestamp": entry.timestamp,
                "pipeline": entry.pipeline,
                "image": entry.image,
                "step": entry.step,
                "duration_ms": entry.duration_ms,
                "details": entry.details,
                "error": entry.error
            }) + "\n")

        if entry.error:
            self.logger.error(f"[{entry.pipeline}] {entry.step}: {entry.error}")
        else:
            self.logger.info(
                f"[{entry.pipeline}] {entry.step} ({entry.duration_ms:.2f}ms)"
            )

    def log_detection(
        self,
        pipeline: str,
        image: str,
        bbox: Dict,
        confidence: float,
        label: str = "object"
    ):
        """Log a single detection event."""
        self.log(PipelineLog(
            pipeline=pipeline,
            image=image,
            step="detection",
            details={
                "bbox": bbox,
                "confidence": confidence,
                "label": label
            }
        ))

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the session."""
        return {
            "session_id": self.session_id,
            "total_logs": len(self.logs),
            "errors": sum(1 for log in self.logs if log.error),
            "log_file": str(self.log_file)
        }


# Global logger instance (lazy initialization)
_logger: Optional[ExperimentLogger] = None


def get_logger() -> ExperimentLogger:
    """Get or create the global experiment logger."""
    global _logger
    if _logger is None:
        _logger = ExperimentLogger()
    return _logger


# =============================================================================
# TIMING UTILITIES
# =============================================================================

@dataclass
class TimingResult:
    """Result of a timed operation."""
    name: str
    duration_ms: float
    start_time: float
    end_time: float


class Timer:
    """Context manager for timing operations."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time: float = 0
        self.end_time: float = 0
        self.duration_ms: float = 0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000

    def result(self) -> TimingResult:
        return TimingResult(
            name=self.name,
            duration_ms=self.duration_ms,
            start_time=self.start_time,
            end_time=self.end_time
        )


@contextmanager
def timed_operation(name: str):
    """Context manager that yields timing information."""
    timer = Timer(name)
    with timer:
        yield timer


# =============================================================================
# INITIALIZATION
# =============================================================================

def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in (CONFIG.results_dir, CONFIG.logs_dir, CONFIG.weights_dir):
        Path(d).mkdir(exist_ok=True)


def init_experiment(require_openai: bool = False) -> Dict[str, Any]:
    """
    Initialize experiment environment.
    Call this before running any pipelines.

    Returns:
        Dict with initialization status and config.
    """
    ensure_dirs()

    env_result = validate_environment(require_openai=require_openai)

    if not env_result["valid"]:
        raise RuntimeError(
            "Environment validation failed:\n" +
            "\n".join(f"  - {e}" for e in env_result["errors"])
        )

    seed = set_reproducibility_seed()

    logger = get_logger()
    logger.logger.info(f"Experiment initialized with seed {seed}")

    return {
        "status": "initialized",
        "seed": seed,
        "config": env_result["config"],
        "warnings": env_result["warnings"],
        "logger": logger.get_summary()
    }
