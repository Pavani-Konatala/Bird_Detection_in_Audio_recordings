"""
evaluate.py - Standalone Model Evaluation & Metrics Generation
==============================================================
Loads a trained model checkpoint (.keras) and evaluates on test data or any dataset:
- Computes Accuracy, ROC-AUC, Precision, Recall, F1-Score, Specificity
- Generates Confusion Matrix, ROC Curve, and Precision-Recall plots
- Measures inference latency (single-item & batch) and parameter footprint
- Saves evaluation results to JSON and PNG files
"""

import os
import sys
import json
import argparse
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

from dataset import scan_audio_files, extract_log_mel_spectrogram, DEFAULT_CONFIG
from model import count_parameters, estimate_macs, benchmark_inference_latency

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_checkpoint(
    model_path: str,
    data_dir: str,
    metadata_path: str = None,
    output_dir: str = "output/eval_results",
    batch_size: int = 32
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading trained model: {model_path}")
    model = tf.keras.models.load_model(model_path)

    # 1. Locate and parse metadata
    if metadata_path is None or not os.path.exists(metadata_path):
        for candidate in ["ff1010_metadata.csv", "ff1010bird_metadata.csv", "metadata.csv"]:
            p = os.path.join(data_dir, candidate)
            if os.path.exists(p):
                metadata_path = p
                break

    if metadata_path is None or not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Cannot find metadata CSV in {data_dir}. Pass --metadata /path/to/csv")

    df = pd.read_csv(metadata_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "itemid" not in df.columns:
        df.rename(columns={df.columns[0]: "itemid"}, inplace=True)
    if "hasbird" not in df.columns:
        for alias in ["label", "target", "bird", "class"]:
            if alias in df.columns:
                df.rename(columns={alias: "hasbird"}, inplace=True)
                break

    df["itemid"] = df["itemid"].astype(str).str.strip()
    df["hasbird"] = df["hasbird"].astype(int)

    audio_map = scan_audio_files(data_dir)
    df["filepath"] = df["itemid"].map(audio_map)
    df = df.dropna(subset=["filepath"]).reset_index(drop=True)

    logger.info(f"Evaluating model on {len(df)} samples from {data_dir}...")

    # 2. Extract features and run inference
    y_true = []
    y_pred_probs = []

    for idx, row in df.iterrows():
        spec = extract_log_mel_spectrogram(row["filepath"])
        batch_spec = np.expand_dims(spec, axis=0)
        prob = float(model.predict(batch_spec, verbose=0)[0][0])
        y_true.append(row["hasbird"])
        y_pred_probs.append(prob)

    y_true = np.array(y_true, dtype=int)
    y_pred_probs = np.array(y_pred_probs, dtype=float)
    y_pred = (y_pred_probs >= 0.5).astype(int)

    # 3. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    ap = average_precision_score(y_true, y_pred_probs)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 4. Generate Plots
    # Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=["No Bird (0)", "Bird (1)"],
        yticklabels=["No Bird (0)", "Bird (1)"]
    )
    plt.title(f"Confusion Matrix (Acc: {acc:.4f}, F1: {f1:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(output_path / "confusion_matrix.png", dpi=300)
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "roc_curve.png", dpi=300)
    plt.close()

    # Precision-Recall Curve
    p_vals, r_vals, _ = precision_recall_curve(y_true, y_pred_probs)
    plt.figure(figsize=(6, 5))
    plt.plot(r_vals, p_vals, color="green", lw=2, label=f"PR (AP = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / "precision_recall_curve.png", dpi=300)
    plt.close()

    # Model complexity
    param_counts = count_parameters(model)
    macs_info = estimate_macs(model, input_shape=(DEFAULT_CONFIG["n_mels"], 431, 1))
    latency_info = benchmark_inference_latency(model, input_shape=(DEFAULT_CONFIG["n_mels"], 431, 1))

    summary = {
        "model_path": str(model_path),
        "total_samples_evaluated": len(df),
        "metrics": {
            "accuracy": round(float(acc), 4),
            "roc_auc": round(float(roc_auc), 4),
            "average_precision_pr_auc": round(float(ap), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "specificity": round(float(specificity), 4),
            "f1_score": round(float(f1), 4),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "complexity": {
            "total_params": param_counts["total_params"],
            "trainable_params": param_counts["trainable_params"],
            "macs_mega": round(macs_info["macs_m"], 2),
            "flops_giga": round(macs_info["flops_g"], 3),
            "latency": latency_info
        }
    }

    with open(output_path / "evaluation_report.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n" + "=" * 65)
    print("                 STANDALONE EVALUATION REPORT")
    print("=" * 65)
    print(f" Samples Evaluated:      {len(df)}")
    print(f" Accuracy:               {acc * 100:.2f}%")
    print(f" ROC-AUC Score:          {roc_auc:.4f}")
    print(f" PR-AUC (Avg Precision): {ap:.4f}")
    print(f" F1-Score:               {f1:.4f}")
    print(f" Precision:              {prec:.4f}")
    print(f" Recall (Sensitivity):   {rec:.4f}")
    print(f" Specificity:            {specificity:.4f}")
    print("-" * 65)
    print(f" Total Parameters:       {param_counts['total_params']:,}")
    print(f" Estimated MACs:         {macs_info['macs_m']:.2f} M-MACs ({macs_info['flops_g']:.3f} GFLOPs)")
    print(f" Inference Latency:      {latency_info['mean_latency_ms']} ms/sample")
    print(f" Saved Plots & Report:   {output_path.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Standalone Model Evaluation")
    parser.add_argument("--model", type=str, default="output/saved_models/best_bird_model.keras", help="Path to saved model (.keras)")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing audio subfolders")
    parser.add_argument("--metadata", type=str, default="data/ff1010bird_metadata_2018.csv", help="Path to ff1010_metadata.csv")
    parser.add_argument("--output-dir", type=str, default="output/eval_results", help="Directory to save evaluation plots & json")
    args = parser.parse_args()

    evaluate_checkpoint(
        model_path=args.model,
        data_dir=args.data_dir,
        metadata_path=args.metadata,
        output_dir=args.output_dir
    )
