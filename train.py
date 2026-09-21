import os
import sys
import json
import argparse
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
)
import tensorflow as tf

from dataset import load_and_align_metadata, get_dataset_splits, DEFAULT_CONFIG
from model import (
    get_model, count_parameters, estimate_macs, benchmark_inference_latency
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_class_weights(train_pos: int, train_neg: int) -> dict:
    """
    Computes balanced class weights:
    w_j = N / (2 * N_j)
    Helps prevent the model from ignoring the minority bird presence class.
    """
    total = train_pos + train_neg
    if train_pos == 0 or train_neg == 0:
        return {0: 1.0, 1: 1.0}
    w0 = total / (2.0 * train_neg)
    w1 = total / (2.0 * train_pos)
    logger.info(f"Computed Class Weights -> Class 0 (No Bird): {w0:.3f}, Class 1 (Bird): {w1:.3f}")
    return {0: float(w0), 1: float(w1)}


def plot_training_history(history: tf.keras.callbacks.History, save_path: str):
    """
    Plots training and validation loss, accuracy, and AUC curves over epochs.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs = range(1, len(history.history["loss"]) + 1)

    # 1. Loss
    axes[0].plot(epochs, history.history["loss"], "b-", label="Train Loss")
    if "val_loss" in history.history:
        axes[0].plot(epochs, history.history["val_loss"], "r--", label="Val Loss")
    axes[0].set_title("Binary Cross-Entropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 2. Accuracy
    acc_key = "accuracy" if "accuracy" in history.history else "binary_accuracy"
    val_acc_key = f"val_{acc_key}"
    axes[1].plot(epochs, history.history[acc_key], "b-", label="Train Accuracy")
    if val_acc_key in history.history:
        axes[1].plot(epochs, history.history[val_acc_key], "r--", label="Val Accuracy")
    axes[1].set_title("Classification Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 3. ROC-AUC
    axes[2].plot(epochs, history.history["auc"], "b-", label="Train AUC")
    if "val_auc" in history.history:
        axes[2].plot(epochs, history.history["val_auc"], "r--", label="Val AUC")
    axes[2].set_title("Area Under ROC Curve (AUC)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("AUC")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    logger.info(f"Saved training curves to: {save_path}")


def evaluate_and_plot_test_set(
    model: tf.keras.Model,
    test_gen,
    results_dir: str
) -> dict:
    """
    Evaluates the model on the held-out test split and generates:
    - Confusion Matrix (Heatmap)
    - ROC Curve with AUC
    - Precision-Recall Curve with AP
    - Classification metrics summary
    """
    logger.info("Gathering test set predictions for comprehensive evaluation...")
    y_true = []
    y_pred_probs = []

    for i in range(len(test_gen)):
        batch_x, batch_y = test_gen[i]
        probs = model.predict(batch_x, verbose=0).flatten()
        y_true.extend(batch_y.tolist())
        y_pred_probs.extend(probs.tolist())

    y_true = np.array(y_true, dtype=int)
    y_pred_probs = np.array(y_pred_probs, dtype=float)
    y_pred_binary = (y_pred_probs >= 0.5).astype(int)

    # Metrics computation
    acc = accuracy_score(y_true, y_pred_binary)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    ap = average_precision_score(y_true, y_pred_probs)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred_binary, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred_binary)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 1. Confusion Matrix Plot
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues", 
        cbar=False,
        xticklabels=["No Bird (0)", "Bird (1)"],
        yticklabels=["No Bird (0)", "Bird (1)"]
    )
    plt.title(f"Test Confusion Matrix\nAccuracy: {acc:.4f} | F1: {f1:.4f}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    cm_path = os.path.join(results_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. ROC Curve Plot
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = os.path.join(results_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # 3. Precision-Recall Curve Plot
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, color="green", lw=2, label=f"PR curve (AP = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = os.path.join(results_dir, "precision_recall_curve.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()

    metrics = {
        "test_samples": len(y_true),
        "accuracy": round(float(acc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "average_precision_pr_auc": round(float(ap), 4),
        "precision": round(float(prec), 4),
        "recall_sensitivity": round(float(rec), 4),
        "specificity": round(float(specificity), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }
    }
    return metrics


def train_pipeline(args):
    """
    End-to-end workflow: Data Ingestion -> Model Compilation -> Training -> Evaluation.
    """
    # Create output directories
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "saved_models"
    logs_dir = output_dir / "logs" / "fit" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = output_dir / "results"

    for d in [models_dir, logs_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info("=== STEP 1: Loading & Aligning Dataset ===")
    df = load_and_align_metadata(args.data_dir, metadata_path=args.metadata)

    if len(df) == 0:
        logger.error("No matched audio samples found. Please check data directory.")
        sys.exit(1)

    logger.info("=== STEP 2: Creating Stratified Splits ===")
    train_gen, val_gen, test_gen, split_info = get_dataset_splits(
        df,
        batch_size=args.batch_size,
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed
    )

    class_weights = compute_class_weights(split_info["train_pos"], split_info["train_neg"])

    logger.info(f"=== STEP 3: Initializing Model ({args.model_type}) ===")
    input_shape = (DEFAULT_CONFIG["n_mels"], 431, 1)
    model = get_model(
        model_type=args.model_type,
        input_shape=input_shape,
        learning_rate=args.lr
    )
    model.summary(print_fn=logger.info)

    # Setup callbacks
    best_model_path = str(models_dir / "best_bird_model.keras")
    callbacks = [
        tf.keras.callbacks.TensorBoard(
            log_dir=str(logs_dir),
            histogram_freq=1,
            write_graph=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(results_dir / "training_history.csv")
        )
    ]

    logger.info("=== STEP 4: Training Model ===")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Plot training curves
    plot_training_history(history, str(results_dir / "training_curves.png"))

    # ==========================================================================
    # STEP 5: AUTOMATIC POST-TRAINING TEST EVALUATION (User Requirement)
    # ==========================================================================
    logger.info("=== STEP 5: In-Training Evaluation on Held-Out Test Set ===")
    if os.path.exists(best_model_path):
        logger.info(f"Loading best checkpoint for evaluation: {best_model_path}")
        best_model = tf.keras.models.load_model(best_model_path)
    else:
        best_model = model

    test_metrics = evaluate_and_plot_test_set(best_model, test_gen, str(results_dir))

    # Model Complexity Profiling
    logger.info("=== STEP 6: Profiling Model Complexity (Params, MACs, Latency) ===")
    param_counts = count_parameters(best_model)
    macs_info = estimate_macs(best_model, input_shape=input_shape)
    latency_info = benchmark_inference_latency(best_model, input_shape=input_shape, num_runs=50)

    # Combine into comprehensive summary
    full_summary = {
        "project": "Bird Audio Classification",
        "dataset": "Freefield1010",
        "timestamp": datetime.datetime.now().isoformat(),
        "hyperparameters": {
            "model_type": args.model_type,
            "sample_rate": DEFAULT_CONFIG["sample_rate"],
            "duration_seconds": DEFAULT_CONFIG["duration"],
            "n_mels": DEFAULT_CONFIG["n_mels"],
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "epochs_trained": len(history.history["loss"])
        },
        "dataset_splits": {
            "total": split_info["total"],
            "train": split_info["train_count"],
            "val": split_info["val_count"],
            "test": split_info["test_count"]
        },
        "test_evaluation_metrics": test_metrics,
        "model_complexity": {
            "total_parameters": param_counts["total_params"],
            "trainable_parameters": param_counts["trainable_params"],
            "estimated_macs": macs_info["total_macs"],
            "mega_macs": round(macs_info["macs_m"], 2),
            "giga_flops": round(macs_info["flops_g"], 3),
            "inference_latency": latency_info
        }
    }

    summary_path = results_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(full_summary, f, indent=4)

    # Pretty print summary table to stdout
    print("\n" + "=" * 70)
    print("                FINAL PROJECT EVALUATION REPORT")
    print("=" * 70)
    print(f" Best Checkpoint Saved:   {best_model_path}")
    print(f" TensorBoard Log Path:    {logs_dir}")
    print("-" * 70)
    print(" [HELD-OUT TEST SET METRICS]")
    print(f"  * Accuracy:             {test_metrics['accuracy'] * 100:.2f}%")
    print(f"  * ROC-AUC:              {test_metrics['roc_auc']:.4f}  (DCASE benchmark metric)")
    print(f"  * PR-AUC (Avg Prec):    {test_metrics['average_precision_pr_auc']:.4f}")
    print(f"  * F1-Score:             {test_metrics['f1_score']:.4f}")
    print(f"  * Precision:            {test_metrics['precision']:.4f}")
    print(f"  * Recall (Sensitivity): {test_metrics['recall_sensitivity']:.4f}")
    print(f"  * Specificity:          {test_metrics['specificity']:.4f}")
    print("-" * 70)
    print(" [MODEL COMPLEXITY & EFFICIENCY]")
    print(f"  * Total Parameters:     {param_counts['total_params']:,}")
    print(f"  * Trainable Parameters: {param_counts['trainable_params']:,}")
    print(f"  * Computational MACs:   {macs_info['macs_m']:.2f} M-MACs ({macs_info['flops_g']:.3f} GFLOPs)")
    print(f"  * Single-Item Latency:  {latency_info['mean_latency_ms']} ms (P50: {latency_info['p50_latency_ms']} ms, P95: {latency_info['p95_latency_ms']} ms)")
    print(f"  * Throughput:           {latency_info['throughput_samples_per_sec']} samples/sec")
    print("-" * 70)
    print(f" Generated Plots in:      {results_dir}/")
    print("   ├── confusion_matrix.png")
    print("   ├── roc_curve.png")
    print("   ├── precision_recall_curve.png")
    print("   └── training_curves.png")
    print(f" Full JSON Metrics:       {summary_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Evaluate Bird Audio Classifier")
    parser.add_argument("--data-dir", type=str, default="data", help="Path to data directory containing 01-10 subdirectories")
    parser.add_argument("--metadata", type=str, default="data/ff1010bird_metadata_2018.csv", help="Path to ff1010_metadata.csv")
    parser.add_argument("--model-type", type=str, default="audio_resnet", choices=["audio_resnet", "audio_mobilenet"], help="Model architecture")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--early-stopping-patience", type=int, default=8, help="Early stopping patience")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for saved models, logs, and results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train_pipeline(args)
