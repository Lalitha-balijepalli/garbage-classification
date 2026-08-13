"""
preprocessing.py
=================
Image transform pipelines (train vs val/test) and the DataLoader factory,
including class-imbalance handling.

Why separate transforms for train vs val/test?
Augmentation (random crop/flip/rotation/color jitter) is a regularisation
technique — it should only touch training data. Validation and test data
must reflect real-world images exactly, otherwise reported accuracy would
not represent how the model performs on genuinely unseen, unmodified images.
"""

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

from src import config
from src.dataset import GarbageDataset, compute_class_weights


def get_train_transform(image_size: int = config.IMAGE_SIZE):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_eval_transform(image_size: int = config.IMAGE_SIZE):
    """Used for validation, test, and single-image inference — NO randomness."""
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),   # short-side resize
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def build_dataloaders(train_items, val_items, test_items, class_names,
                       batch_size: int = config.BATCH_SIZE,
                       use_weighted_sampler: bool = True):
    """
    Build train/val/test DataLoaders.

    Imbalance handling: if `use_weighted_sampler` is True, we use a
    WeightedRandomSampler on the TRAINING loader only, so each mini-batch
    sees roughly equal representation of every class regardless of how
    skewed the raw dataset is. We combine this with class-weighted loss
    (see train.py) rather than aggressive oversampling alone, because
    oversampling minority images verbatim risks overfitting to the few
    duplicated/augmented copies of rare classes — splitting the correction
    between sampler + loss weighting is gentler.
    """
    train_ds = GarbageDataset(train_items, transform=get_train_transform())
    val_ds = GarbageDataset(val_items, transform=get_eval_transform())
    test_ds = GarbageDataset(test_items, transform=get_eval_transform())

    if use_weighted_sampler:
        class_weights = compute_class_weights(train_items, len(class_names))
        sample_weights = [class_weights[label].item() for _, label in train_items]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler,
            num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
