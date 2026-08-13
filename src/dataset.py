"""
dataset.py
==========
Everything related to *finding*, *validating*, and *splitting* the garbage
classification dataset.

Design goal: the Kaggle "garbage-classification" dataset has shipped with
slightly different folder layouts over time (nested "Garbage classification/
Garbage classification/<class>/*.jpg", or a flatter "<class>/*.jpg" layout).
Rather than hard-coding one layout, we WALK the raw data directory and infer
class folders automatically: any directory whose immediate children are only
image files is treated as a class.
"""

import os
import hashlib
from collections import defaultdict

from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from src import config

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# 1. Class / image discovery
# ---------------------------------------------------------------------------
def discover_class_folders(root_dir: str) -> dict:
    """
    Walk `root_dir` and find every folder that directly contains image
    files. Returns {class_name: [list of absolute image paths]}.

    Why a generic walk instead of a fixed path?
    The task requires the code to keep working "even if the exact directory
    structure of the Kaggle dataset changes slightly" — a walk that treats
    "a folder full of images" as a class is robust to nesting differences.
    """
    class_to_images = defaultdict(list)

    for dirpath, dirnames, filenames in os.walk(root_dir):
        image_files = [
            f for f in filenames
            if os.path.splitext(f)[1].lower() in VALID_EXTENSIONS
        ]
        if not image_files:
            continue  # not a leaf image folder — keep walking

        class_name = os.path.basename(dirpath.rstrip("/"))
        # Skip nonsense folder names that sometimes appear in Kaggle zips
        if class_name.lower() in {"raw", "data", "dataset", "garbage classification"}:
            continue

        for f in image_files:
            class_to_images[class_name].append(os.path.join(dirpath, f))

    if not class_to_images:
        raise FileNotFoundError(
            f"No class folders with images were found under '{root_dir}'. "
            f"Make sure the dataset has been downloaded/extracted into "
            f"config.RAW_DATA_DIR (see README 'Dataset Setup')."
        )

    return dict(class_to_images)


def get_class_names(class_to_images: dict) -> list:
    """Sorted, deterministic list of class names (order matters for label indices)."""
    return sorted(class_to_images.keys())


# ---------------------------------------------------------------------------
# 2. Dataset quality checks
# ---------------------------------------------------------------------------
def _file_hash(path, block_size=65536):
    """MD5 hash of file bytes, used for exact-duplicate detection."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_dataset(class_to_images: dict, min_side: int = 32) -> dict:
    """
    Scan every image and report:
      - corrupted / unreadable images
      - duplicate images (exact byte-duplicates, via MD5)
      - extremely small images (either side < min_side px)
      - unsupported formats (should be none, since discover_class_folders
        already filters by extension, but we double check the file can
        actually be opened as the extension implies)

    Returns a report dict. Does NOT delete anything automatically — the
    caller decides how to handle flagged files, so nothing is silently lost.
    """
    report = {
        "corrupted": [],
        "duplicates": [],   # list of (original, duplicate) path pairs
        "too_small": [],
        "total_scanned": 0,
    }

    seen_hashes = {}

    for class_name, paths in class_to_images.items():
        for path in paths:
            report["total_scanned"] += 1
            try:
                with Image.open(path) as img:
                    img.verify()  # cheap structural check
                # re-open (verify() invalidates the file handle) to check size
                with Image.open(path) as img:
                    w, h = img.size
                    if w < min_side or h < min_side:
                        report["too_small"].append(path)
            except (UnidentifiedImageError, OSError, ValueError):
                report["corrupted"].append(path)
                continue

            try:
                fh = _file_hash(path)
            except OSError:
                continue
            if fh in seen_hashes:
                report["duplicates"].append((seen_hashes[fh], path))
            else:
                seen_hashes[fh] = path

    return report


def clean_dataset(class_to_images: dict, report: dict) -> dict:
    """
    Return a NEW class_to_images dict with corrupted images, one copy of
    each exact-duplicate pair, and undersized images removed.
    """
    bad_paths = set(report["corrupted"]) | set(report["too_small"])
    bad_paths |= {dup for _, dup in report["duplicates"]}  # keep the original, drop the dup

    cleaned = {}
    for class_name, paths in class_to_images.items():
        cleaned[class_name] = [p for p in paths if p not in bad_paths]
    return cleaned


def dataset_summary(class_to_images: dict) -> dict:
    """Numeric summary used by the notebook's EDA section."""
    counts = {c: len(paths) for c, paths in class_to_images.items()}
    total = sum(counts.values())
    return {
        "num_classes": len(counts),
        "class_names": sorted(counts.keys()),
        "counts_per_class": counts,
        "total_images": total,
        "min_images": min(counts.values()),
        "max_images": max(counts.values()),
        "avg_images": total / len(counts),
    }


# ---------------------------------------------------------------------------
# 3. Stratified train/val/test split
# ---------------------------------------------------------------------------
def stratified_split(class_to_images: dict, seed: int = config.SEED):
    """
    Build (path, label_index) lists for train/val/test using stratified
    sampling per class, so every class is proportionally represented in
    all three splits and no image appears in more than one split
    (this also rules out the accidental leakage described in Section 6,
    since each path is assigned to exactly one split).
    """
    class_names = get_class_names(class_to_images)
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    train_items, val_items, test_items = [], [], []

    for class_name, paths in class_to_images.items():
        label = class_to_idx[class_name]
        labels = [label] * len(paths)

        # First split off the test set
        train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
            paths, labels,
            test_size=config.TEST_RATIO,
            random_state=seed,
            shuffle=True,
        )
        # Then split remaining into train/val
        val_fraction_of_remainder = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_val_paths, train_val_labels,
            test_size=val_fraction_of_remainder,
            random_state=seed,
            shuffle=True,
        )

        train_items += list(zip(train_paths, train_labels))
        val_items += list(zip(val_paths, val_labels))
        test_items += list(zip(test_paths, test_labels))

    return train_items, val_items, test_items, class_names


# ---------------------------------------------------------------------------
# 4. PyTorch Dataset
# ---------------------------------------------------------------------------
class GarbageDataset(Dataset):
    """
    Thin wrapper around a list of (image_path, label_index) tuples.
    Transform (augmentation/normalisation) is injected so the SAME class
    can serve train/val/test simply by passing different transforms.
    """

    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def compute_class_weights(train_items, num_classes):
    """
    Inverse-frequency class weights for CrossEntropyLoss, used when the
    dataset is imbalanced (see Section 7 / preprocessing.py).
    """
    counts = torch.zeros(num_classes)
    for _, label in train_items:
        counts[label] += 1
    counts = torch.clamp(counts, min=1)
    weights = counts.sum() / (num_classes * counts)
    return weights
