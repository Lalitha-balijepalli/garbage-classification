"""
scripts/compare_models.py
==========================
Evaluates every trained model checkpoint found in models/ on the SAME
held-out test set, builds the fair comparison table (results/model_comparison.csv),
and copies the winning checkpoint to models/best_model.pth.

Run this only after training the models you want to compare
(scripts/train_cpu.py --model ...).

Usage:
    python3 scripts/compare_models.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config, utils, dataset, preprocessing, models, evaluate


def main():
    utils.set_seed(config.SEED)
    config.NUM_WORKERS = 0  # same shared-memory fix as train_cpu.py

    print("--- Rebuilding the exact same dataset split used during training ---")
    class_to_images = dataset.discover_class_folders(config.RAW_DATA_DIR)
    report = dataset.audit_dataset(class_to_images)
    clean = dataset.clean_dataset(class_to_images, report)
    train_items, val_items, test_items, class_names = dataset.stratified_split(clean, seed=config.SEED)
    print(f"Test set: {len(test_items)} images across {len(class_names)} classes")

    _, _, test_loader = preprocessing.build_dataloaders(
        train_items, val_items, test_items, class_names, batch_size=config.BATCH_SIZE
    )

    results = {}
    for model_name, ckpt_path in config.MODEL_CHECKPOINT_PATHS.items():
        if not os.path.exists(ckpt_path):
            print(f"Skipping {model_name} — no checkpoint found at {ckpt_path}")
            continue

        print(f"\n--- Evaluating {model_name} ---")
        checkpoint = utils.load_checkpoint(ckpt_path, map_location=config.DEVICE)
        model = models.build_model(model_name, num_classes=len(class_names), freeze_backbone=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(config.DEVICE)

        y_pred, y_true, _ = evaluate.get_predictions(model, test_loader, device=config.DEVICE)
        metrics = evaluate.compute_metrics(y_true, y_pred, class_names)

        print(f"Test Accuracy: {metrics['accuracy']*100:.2f}%  "
              f"Precision: {metrics['precision']:.4f}  "
              f"Recall: {metrics['recall']:.4f}  "
              f"F1: {metrics['f1']:.4f}")

        utils.plot_confusion_matrix(metrics["confusion_matrix"], class_names, model_name)
        confused = evaluate.most_confused_pairs(metrics["confusion_matrix"], class_names)
        if confused:
            print("Most confused pairs (true -> predicted, count):")
            for true_c, pred_c, cnt in confused:
                print(f"  {true_c} -> {pred_c}: {cnt}")

        results[model_name] = {
            "model": model,
            "metrics": metrics,
            # Training time isn't stored in the checkpoint itself — refer back
            # to the "Total time: X min" line each train_cpu.py run printed
            # to your terminal if you want that figure. Using NaN here rather
            # than fabricating or mislabeling a number.
            "training_time_sec": float("nan"),
            "image_size": checkpoint["image_size"],
        }

    if not results:
        print("\nNo trained checkpoints found — train at least one model first.")
        return

    print("\n--- Building comparison table ---")
    comparison_df = evaluate.build_comparison_table(results)
    print(comparison_df.to_string(index=False))

    print("\n--- Selecting best model ---")
    best_info = evaluate.select_best_model(comparison_df, f1_weight=0.7, speed_weight=0.3)
    print("=" * 40)
    print("BEST MODEL")
    print("=" * 40)
    print(f"Model: {best_info['model_name']}")
    print(f"Test Accuracy: {best_info['test_accuracy']*100:.2f}%")
    print(f"F1 Score: {best_info['f1_score']:.4f}")
    print(f"Inference Time: {best_info['inference_time_ms']:.2f} ms")
    print(f"Reason: {best_info['reason']}")

    import shutil
    best_ckpt = config.MODEL_CHECKPOINT_PATHS[best_info["model_name"]]
    shutil.copyfile(best_ckpt, config.BEST_MODEL_PATH)
    print(f"\nCopied {best_ckpt} -> {config.BEST_MODEL_PATH}")
    print("The Streamlit app will now load this model automatically.")


if __name__ == "__main__":
    main()