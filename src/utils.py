"""
utils.py
========
Small, generic helper functions used by multiple modules (training,
evaluation, inference). Keeping these here avoids duplicating the same
plotting / seeding / checkpoint logic in three different training scripts.
"""

import os
import json
import random
import time
from contextlib import contextmanager

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from src import config


def set_seed(seed: int = config.SEED) -> None:
    """
    Fix every relevant random-number generator so runs are reproducible.

    Why: PyTorch's DataLoader shuffling, weight initialisation, dropout
    masks, and augmentation all draw from RNGs. Without fixing seeds,
    re-running the same notebook can give different numbers each time,
    which makes debugging and fair model comparison impossible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN kernels (slightly slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@contextmanager
def timer(name: str = "block"):
    """Context manager that prints how long a block of code took to run."""
    start = time.time()
    yield
    elapsed = time.time() - start
    print(f"[TIMER] {name}: {elapsed:.2f}s")


def count_parameters(model: torch.nn.Module) -> int:
    """Total number of trainable parameters — used in the model comparison table."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Approximate on-disk size of a model in megabytes, based on parameter
    and buffer byte counts. Useful for comparing deployment footprint
    (e.g. MobileNetV3 vs EfficientNet-B3) without needing to save to disk.
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def save_checkpoint(model, class_names, model_name, image_size, extra=None, path=None):
    """
    Save a full checkpoint dict (not just raw weights) so the model can be
    reloaded anywhere without needing the training script's globals.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": class_names,
        "model_name": model_name,
        "image_size": image_size,
        "mean": config.IMAGENET_MEAN,
        "std": config.IMAGENET_STD,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, path)
    print(f"[CHECKPOINT SAVED] {path}")


def load_checkpoint(path, map_location=None):
    """Load a checkpoint dict saved by save_checkpoint()."""
    map_location = map_location or config.DEVICE
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model checkpoint not found at '{path}'. "
            f"Train the model first, or check the path in config.py."
        )
    return torch.load(path, map_location=map_location)


def plot_training_curves(history: dict, model_name: str, save_dir: str = config.FIGURES_DIR):
    """
    Plot train vs val accuracy and train vs val loss side by side.
    `history` is expected to have keys: train_loss, val_loss, train_acc, val_acc.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], label="Val Acc")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{model_name}_training_curves.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"[FIGURE SAVED] {out_path}")


def plot_confusion_matrix(cm, class_names, model_name: str, save_dir: str = config.FIGURES_DIR):
    """Render and save a confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} — Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    out_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(out_path, dpi=150)
    plt.show()
    print(f"[FIGURE SAVED] {out_path}")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
