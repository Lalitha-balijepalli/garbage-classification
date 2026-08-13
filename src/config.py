"""
config.py
=========
Single source of truth for every hyperparameter, path, and constant used
across the project. Nothing else in the codebase should hard-code these
values — always import them from here.

Why centralise config?
-----------------------
- Makes experiments reproducible (one place to change, one place to read).
- Avoids "magic numbers" scattered across training/eval/inference code.
- Lets you swap datasets/models/hyperparameters without touching logic code.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
SPLITS_DIR = os.path.join(DATA_DIR, "splits")

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

for _d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, SPLITS_DIR,
           MODELS_DIR, RESULTS_DIR, FIGURES_DIR, METRICS_DIR]:
    os.makedirs(_d, exist_ok=True)

# The dataset, once downloaded/extracted, is expected somewhere under
# RAW_DATA_DIR. We do NOT assume a fixed sub-folder name because Kaggle
# archives for this dataset have shipped with slightly different internal
# layouts over time (e.g. "Garbage classification/Garbage classification/*",
# or a flat "<class_name>/*.jpg" layout). dataset.py auto-detects this.
DATASET_KAGGLE_SLUG = "asdasdasasdas/garbage-classification"

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Data splitting
# ---------------------------------------------------------------------------
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # must sum to 1.0 with the two above

# ---------------------------------------------------------------------------
# Image / augmentation
# ---------------------------------------------------------------------------
# 300px is the native input resolution EfficientNet-B3 was pretrained at.
# ResNet50 and MobileNetV3 were pretrained at 224, but they tolerate other
# square input sizes fine (global average pooling removes the resolution
# dependency), so we standardise every model on ONE input size to keep the
# comparison in Section 11 fair (same input pixels -> same information
# budget for every architecture).
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
BATCH_SIZE = 32          # lower this to 16 if you hit CUDA OOM on Colab's T4
NUM_EPOCHS_HEAD = 8       # phase 1: train classifier head only
NUM_EPOCHS_FINETUNE = 12  # phase 2: fine-tune deeper layers
LEARNING_RATE_HEAD = 1e-3
LEARNING_RATE_FINETUNE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
NUM_WORKERS = 2           # Colab default; set to os.cpu_count() locally if higher

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.60
TOP_K = 3

# ---------------------------------------------------------------------------
# Models compared in this project
# ---------------------------------------------------------------------------
MODEL_NAMES = ["efficientnet_b3", "resnet50", "mobilenet_v3"]

MODEL_CHECKPOINT_PATHS = {
    "efficientnet_b3": os.path.join(MODELS_DIR, "efficientnet_b3_best.pth"),
    "resnet50": os.path.join(MODELS_DIR, "resnet50_best.pth"),
    "mobilenet_v3": os.path.join(MODELS_DIR, "mobilenet_v3_best.pth"),
}

BEST_MODEL_INFO_PATH = os.path.join(MODELS_DIR, "best_model_info.json")

# Path the Streamlit app / predict.py loads by default.
# best_model.py (Section 33 / CELL 24 in the notebook) copies whichever
# checkpoint wins into this filename after training.
BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")

# Comparison table produced by the notebook / evaluate.py
MODEL_COMPARISON_CSV = os.path.join(RESULTS_DIR, "model_comparison.csv")
