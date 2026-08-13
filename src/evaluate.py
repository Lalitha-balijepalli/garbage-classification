"""
evaluate.py
===========
Model evaluation: accuracy/precision/recall/F1, confusion matrix,
classification report, inference-time benchmarking, and the final
cross-model comparison table (Section 11).
"""

import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, f1_score,
)

from src import config
from src.utils import count_parameters, get_model_size_mb


@torch.no_grad()
def get_predictions(model, loader, device=config.DEVICE):
    model.eval()
    model.to(device)
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

    return (np.concatenate(all_preds), np.concatenate(all_labels), np.concatenate(all_probs))


def compute_metrics(y_true, y_pred, class_names):
    """Accuracy, macro/weighted precision-recall-F1, and a full per-class report."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                    zero_division=0, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
        "classification_report": report,
        "confusion_matrix": cm,
    }


def most_confused_pairs(cm, class_names, top_n: int = 5):
    """
    Return the top-N (true_class, predicted_class, count) off-diagonal
    confusion-matrix entries — i.e. the most common misclassifications.
    Used to explain *why* certain classes get confused (Section 16).
    """
    pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                pairs.append((class_names[i], class_names[j], int(cm[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


@torch.no_grad()
def benchmark_inference_time(model, image_size, device=config.DEVICE, n_runs: int = 50):
    """Average single-image inference latency in milliseconds."""
    model.eval()
    model.to(device)
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    # warm-up (first call pays CUDA kernel compilation cost)
    for _ in range(5):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_runs):
        model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.time() - start

    return (elapsed / n_runs) * 1000  # ms per image


def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    `results` = {
        model_name: {
            "model": <nn.Module>,
            "metrics": <dict from compute_metrics>,
            "training_time_sec": float,
            "image_size": int,
        }, ...
    }
    Returns the Section 11 comparison DataFrame and saves it to CSV.
    """
    rows = []
    for name, r in results.items():
        model = r["model"]
        metrics = r["metrics"]
        inference_ms = benchmark_inference_time(model, r["image_size"])
        rows.append({
            "Model": name,
            "Parameters": count_parameters(model),
            "Training Time (s)": round(r.get("training_time_sec", float("nan")), 1),
            "Test Accuracy": round(metrics["accuracy"], 4),
            "Precision": round(metrics["precision"], 4),
            "Recall": round(metrics["recall"], 4),
            "F1 Score": round(metrics["f1"], 4),
            "Model Size (MB)": round(get_model_size_mb(model), 2),
            "Inference Time (ms)": round(inference_ms, 2),
        })

    df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False).reset_index(drop=True)
    df.to_csv(config.MODEL_COMPARISON_CSV, index=False)
    print(f"[SAVED] {config.MODEL_COMPARISON_CSV}")
    return df


def select_best_model(comparison_df: pd.DataFrame,
                       f1_weight: float = 0.7,
                       speed_weight: float = 0.3) -> dict:
    """
    Pick the best model using a weighted score of test F1 (accuracy) and
    inference speed, rather than picking the largest architecture by
    default (per Section 33's requirement). Speed is normalised so faster
    models score higher; F1 is used directly (already 0-1).
    """
    df = comparison_df.copy()
    max_latency = df["Inference Time (ms)"].max()
    df["speed_score"] = 1 - (df["Inference Time (ms)"] / max_latency)
    df["combined_score"] = f1_weight * df["F1 Score"] + speed_weight * df["speed_score"]

    best_row = df.sort_values("combined_score", ascending=False).iloc[0]

    reason = (
        f"Selected for the best combined score of test F1 ({best_row['F1 Score']:.4f}, "
        f"weight={f1_weight}) and inference speed "
        f"({best_row['Inference Time (ms)']:.2f} ms/image, weight={speed_weight})."
    )

    return {
        "model_name": best_row["Model"],
        "test_accuracy": float(best_row["Test Accuracy"]),
        "f1_score": float(best_row["F1 Score"]),
        "inference_time_ms": float(best_row["Inference Time (ms)"]),
        "reason": reason,
    }
