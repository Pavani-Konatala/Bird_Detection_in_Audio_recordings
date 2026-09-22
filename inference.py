import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import librosa
import tensorflow as tf

from dataset import extract_log_mel_spectrogram, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def predict_single_audio(model: tf.keras.Model,audio_path: str,threshold: float = 0.5,visualize: bool = False,output_viz_dir: str = "output/visualizations") -> Dict[str, Any]:
    
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # 1. Measure Preprocessing Time (Spectrogram extraction)
    t_start_prep = time.perf_counter()
    spec = extract_log_mel_spectrogram(
        audio_path,
        sr=DEFAULT_CONFIG["sample_rate"],
        duration=DEFAULT_CONFIG["duration"],
        n_mels=DEFAULT_CONFIG["n_mels"],
        n_fft=DEFAULT_CONFIG["n_fft"],
        hop_length=DEFAULT_CONFIG["hop_length"]
    )
    t_prep_ms = (time.perf_counter() - t_start_prep) * 1000.0

    # 2. Add batch dimension: (1, 128, 431, 1)
    input_tensor = np.expand_dims(spec, axis=0)

    # 3. Measure Neural Network Forward Pass Latency
    t_start_infer = time.perf_counter()
    pred_prob = float(model.predict(input_tensor, verbose=0)[0][0])
    t_infer_ms = (time.perf_counter() - t_start_infer) * 1000.0

    total_latency_ms = t_prep_ms + t_infer_ms
    has_bird = bool(pred_prob >= threshold)
    confidence = (pred_prob if has_bird else (1.0 - pred_prob)) * 100.0

    result = {
        "file": str(Path(audio_path).resolve()),
        "has_bird": has_bird,
        "verdict": "BIRD DETECTED" if has_bird else "NO BIRD DETECTED",
        "probability_hasbird": round(pred_prob, 4),
        "confidence_percent": round(confidence, 2),
        "latency_breakdown_ms": {
            "audio_preprocessing": round(t_prep_ms, 2),
            "model_forward_pass": round(t_infer_ms, 2),
            "total_latency": round(total_latency_ms, 2)
        }
    }

    # Optional Spectrogram Visualization
    if visualize:
        viz_path = Path(output_viz_dir)
        viz_path.mkdir(parents=True, exist_ok=True)
        save_file = viz_path / f"{Path(audio_path).stem}_spec.png"

        y, sr = librosa.load(audio_path, sr=DEFAULT_CONFIG["sample_rate"])

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        # Waveform
        librosa.display.waveshow(y, sr=sr, ax=axes[0], color="#2b5c8f")
        axes[0].set_title(f"Audio Waveform: {Path(audio_path).name}", fontsize=11)
        axes[0].set_xlabel("Time (seconds)")
        axes[0].set_ylabel("Amplitude")

        # Log-Mel Spectrogram
        mel_display = spec.squeeze()
        img = axes[1].imshow(
            mel_display, 
            aspect="auto", 
            origin="lower", 
            cmap="inferno"
        )
        axes[1].set_title(
            f"Log-Mel Spectrogram | Verdict: {result['verdict']} "
            f"({result['confidence_percent']:.1f}% confidence, P={result['probability_hasbird']:.3f})",
            fontsize=11,
            color="green" if has_bird else "darkred"
        )
        axes[1].set_xlabel("Time Frames")
        axes[1].set_ylabel("Mel Bins (0 - 128)")
        plt.colorbar(img, ax=axes[1], format="%+2.0f")

        plt.tight_layout()
        plt.savefig(save_file, dpi=300)
        plt.close()
        result["visualization_saved"] = str(save_file)

    return result


def main():
    parser = argparse.ArgumentParser(description="Bird Audio Classifier - Inference CLI")
    parser.add_argument("--model", type=str, default="output/saved_models/best_bird_model.keras", help="Path to saved model (.keras)")
    parser.add_argument("--audio", type=str, default="data/01/2534.wav", help="Path to single unseen .wav file")
    parser.add_argument("--audio-dir", type=str, default=None, help="Directory containing multiple .wav files")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for bird presence (default 0.5)")
    parser.add_argument("--visualize", default=True, help="Save spectrogram plot with prediction banner")
    parser.add_argument("--output-json", type=str, default=None, help="Save predictions to JSON file")
    parser.add_argument("--viz-dir", type=str, default="output/visualizations", help="Directory for saved visualizations")
    args = parser.parse_args()

    if args.audio is None and args.audio_dir is None:
        print("Error: Please provide either --audio <path.wav> or --audio-dir <folder>")
        sys.exit(1)

    # Load model
    logger.info(f"Loading model checkpoint: {args.model}")
    model = tf.keras.models.load_model(args.model)

    # Collect audio files
    audio_paths = []
    if args.audio:
        audio_paths.append(args.audio)
    if args.audio_dir:
        audio_paths.extend(glob.glob(os.path.join(args.audio_dir, "**/*.wav"), recursive=True))

    if len(audio_paths) == 0:
        logger.error("No .wav files found to process.")
        sys.exit(1)

    print("\n" + "=" * 75)
    print("                     BIRD CLASSIFICATION INFERENCE")
    print("=" * 75)
    print(f" Loaded Model:    {args.model}")
    print(f" Threshold:       {args.threshold}")
    print(f" Audio Files:     {len(audio_paths)}")
    print("-" * 75)

    results = []
    for p in audio_paths:
        res = predict_single_audio(
            model=model,
            audio_path=p,
            threshold=args.threshold,
            visualize=args.visualize,
            output_viz_dir=args.viz_dir
        )
        results.append(res)

        # Print formatted output for each sample
        verdict_color = "\033[92m" if res["has_bird"] else "\033[91m"
        reset_color = "\033[0m"
        print(f" File:       {Path(res['file']).name}")
        print(f" Verdict:    [{res['verdict']}]")
        print(f" Confidence: {res['confidence_percent']}%  (P={res['probability_hasbird']})")
        print(f" Latency:    {res['latency_breakdown_ms']['total_latency']} ms  "
              f"(Extract: {res['latency_breakdown_ms']['audio_preprocessing']} ms, "
              f"Model: {res['latency_breakdown_ms']['model_forward_pass']} ms)")
        if "visualization_saved" in res:
            print(f" Plot Saved: {res['visualization_saved']}")
        print("-" * 75)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Predictions saved to JSON: {args.output_json}")

    print("Inference completed successfully.\n")


if __name__ == "__main__":
    import glob
    main()
