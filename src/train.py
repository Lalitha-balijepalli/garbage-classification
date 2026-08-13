"""
train.py
========
A single, reusable training function used identically for all three models
(EfficientNet-B3, ResNet50, MobileNetV3) so the comparison in Section 11 is
fair — same loop, same scheduler logic, same early-stopping rule, only the
architecture differs.

Features:
 - Mixed-precision training via torch.amp (falls back cleanly on CPU)
 - ReduceLROnPlateau scheduler (drops LR when val loss stalls)
 - Best-checkpoint tracking (by validation accuracy)
 - Early stopping
 - tqdm progress bars
 - Full training history returned for plotting
"""

import copy
import time

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src import config


def _run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train() if train else model.eval()

    running_loss, running_correct, total = 0.0, 0, 0
    use_amp = scaler is not None

    phase_name = "train" if train else "val"
    pbar = tqdm(loader, desc=phase_name, leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            if use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += images.size(0)
        pbar.set_postfix(loss=running_loss / total, acc=running_correct / total)

    return running_loss / total, running_correct / total


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                 epochs: int, device=config.DEVICE, patience: int = config.EARLY_STOPPING_PATIENCE,
                 model_name: str = "model", checkpoint_path: str = None, class_names=None):
    """
    Train `model` for up to `epochs`, with early stopping on validation
    accuracy. Returns (best_model_state_dict, history_dict).
    """
    model.to(device)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, optimizer, device, scaler, train=False)

        if scheduler is not None:
            # ReduceLROnPlateau steps on the metric being monitored (val_loss)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - start
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"[{model_name}] Epoch {epoch}/{epochs} "
              f"- train_loss {train_loss:.4f} train_acc {train_acc:.4f} "
              f"- val_loss {val_loss:.4f} val_acc {val_acc:.4f} "
              f"- lr {current_lr:.2e} - {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            if checkpoint_path:
                from src.utils import save_checkpoint
                save_checkpoint(model, class_names, model_name, config.IMAGE_SIZE, path=checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[{model_name}] Early stopping at epoch {epoch} "
                      f"(no val_acc improvement for {patience} epochs).")
                break

    model.load_state_dict(best_state)
    return model, history, best_val_acc


def make_criterion(class_weights=None, device=config.DEVICE):
    """
    CrossEntropyLoss, optionally weighted by inverse class frequency to
    counter class imbalance (see Section 7 / dataset.compute_class_weights).
    """
    if class_weights is not None:
        return nn.CrossEntropyLoss(weight=class_weights.to(device))
    return nn.CrossEntropyLoss()


def make_optimizer(model, lr: float = config.LEARNING_RATE_HEAD):
    """
    AdamW: Adam with decoupled weight decay. Chosen over plain Adam/SGD
    because it converges quickly on transfer-learning tasks (few trainable
    params in phase 1) while the decoupled weight decay generalises better
    than L2-regularised Adam.
    """
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(trainable_params, lr=lr, weight_decay=config.WEIGHT_DECAY)


def make_scheduler(optimizer):
    """
    ReduceLROnPlateau: halves the LR when validation loss stops improving
    for 2 consecutive epochs. Chosen over a fixed step schedule because it
    adapts to each model's actual convergence speed instead of assuming all
    three architectures plateau at the same epoch.
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
