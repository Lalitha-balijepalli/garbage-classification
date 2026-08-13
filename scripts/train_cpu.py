"""
scripts/train_cpu.py
=====================
Standalone training script for CPU environments (e.g. GitHub Codespaces
without a GPU). Trains ONE model at a time so you can monitor progress in
the terminal and run models sequentially/overnight rather than all at once
in a notebook.

Usage:
    python3 scripts/train_cpu.py --model resnet50
    python3 scripts/train_cpu.py --model efficientnet_b3 --epochs-head 3 --epochs-finetune 4
    python3 scripts/train_cpu.py --model mobilenet_v3

Safe to re-run: if a checkpoint already exists for the chosen model, this
script loads those weights as a warm start instead of retraining from
ImageNet weights again.
"""

import argparse
import os
import sys
import time

# Add the repo root (parent of this script's `scripts/` folder) to the
# import path, so `from src import ...` works no matter what directory
# you run this script from.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, utils, dataset, preprocessing, models, train

# Force single-process data loading — Codespaces / constrained Docker
# containers ship with a tiny /dev/shm, and DataLoader worker processes
# (num_workers > 0) hand tensors back via shared memory, which fails with
# "unable to allocate shared memory(shm)... No space left on device" even
# though real disk space is fine. Setting this here (rather than only in
# config.py) guarantees the fix applies regardless of whether config.py
# was edited/saved correctly.
config.NUM_WORKERS = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["efficientnet_b3", "resnet50", "mobilenet_v3"])
    parser.add_argument("--epochs-head", type=int, default=3, help="Lower than default (8) — sensible for CPU")
    parser.add_argument("--epochs-finetune", type=int, default=4, help="Lower than default (12) — sensible for CPU")
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Override config.BATCH_SIZE — lower this (e.g. 8 or 4) if you hit OOM kills")
    parser.add_argument("--unfreeze-blocks", type=int, default=2,
                         help="How many backbone blocks to unfreeze during fine-tuning. Lower this "
                              "(e.g. 1) to reduce memory use on constrained machines — fewer trainable "
                              "layers means less activation memory needed for backprop.")
    args = parser.parse_args()

    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size
        print(f"Overriding batch size to {config.BATCH_SIZE}")

    utils.set_seed(config.SEED)
    print(f"Device: {config.DEVICE}  (CPU training will be slow — this is expected without a GPU)")

    print("\n--- Discovering & cleaning dataset ---")
    class_to_images = dataset.discover_class_folders(config.RAW_DATA_DIR)
    report = dataset.audit_dataset(class_to_images)
    print(f"Corrupted: {len(report['corrupted'])}  Duplicates: {len(report['duplicates'])}  "
          f"Too small: {len(report['too_small'])}")
    clean = dataset.clean_dataset(class_to_images, report)

    print("\n--- Splitting ---")
    train_items, val_items, test_items, class_names = dataset.stratified_split(clean, seed=config.SEED)
    print(f"Train: {len(train_items)}  Val: {len(val_items)}  Test: {len(test_items)}  Classes: {class_names}")

    print("\n--- Building DataLoaders ---")
    train_loader, val_loader, test_loader = preprocessing.build_dataloaders(
        train_items, val_items, test_items, class_names, batch_size=config.BATCH_SIZE
    )

    print(f"\n--- Building {args.model} ---")
    model = models.build_model(args.model, num_classes=len(class_names), freeze_backbone=True)

    checkpoint_path = config.MODEL_CHECKPOINT_PATHS[args.model]
    existing = utils.load_checkpoint(checkpoint_path, map_location=config.DEVICE) \
        if __import__("os").path.exists(checkpoint_path) else None
    if existing is not None:
        print(f"Found existing checkpoint at {checkpoint_path} — warm-starting from it.")
        model.load_state_dict(existing["model_state_dict"])

    class_weights = dataset.compute_class_weights(train_items, len(class_names))
    criterion = train.make_criterion(class_weights, device=config.DEVICE)

    print(f"\n--- Phase 1: training head ({args.epochs_head} epochs) ---")
    optimizer = train.make_optimizer(model, lr=config.LEARNING_RATE_HEAD)
    scheduler = train.make_scheduler(optimizer)
    start = time.time()
    model, history_head, best_val_acc = train.train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        epochs=args.epochs_head, device=config.DEVICE, model_name=args.model,
        checkpoint_path=checkpoint_path, class_names=class_names,
    )

    print(f"\n--- Phase 2: fine-tuning ({args.epochs_finetune} epochs) ---")
    model = models.unfreeze_for_finetuning(model, args.model, num_blocks=args.unfreeze_blocks)
    optimizer_ft = train.make_optimizer(model, lr=config.LEARNING_RATE_FINETUNE)
    scheduler_ft = train.make_scheduler(optimizer_ft)
    model, history_ft, best_val_acc = train.train_model(
        model, train_loader, val_loader, criterion, optimizer_ft, scheduler_ft,
        epochs=args.epochs_finetune, device=config.DEVICE, model_name=args.model,
        checkpoint_path=checkpoint_path, class_names=class_names,
    )

    elapsed = time.time() - start
    print(f"\nDone. Best val_acc: {best_val_acc:.4f}. Total time: {elapsed/60:.1f} min.")
    print(f"Checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()