# Bird Audio Classification Pipeline (TensorFlow / Keras)

End-to-end deep learning pipeline for binary bird vocalization detection (`hasbird = 1` vs `hasbird = 0`) trained on the **Freefield1010** dataset using **TensorFlow 2.x and Keras**.

---

## 📁 Dataset Layout

The pipeline expects audio files organized across subdirectories (e.g., `01/` to `10/`) alongside the metadata CSV:

```text
data/
├── 01/
│   ├── 10001.wav
│   └── 10001.json
├── 02/
│   ├── 10005.wav
│   └── 10005.json
├── ...
├── 10/
│   ├── 10040.wav
│   └── 10040.json
└── ff1010_metadata.csv
```

### Metadata Format (`ff1010_metadata.csv`)
| itemid | hasbird |
| :--- | :--- |
| `10001` | `1` |
| `10002` | `0` |
| `10003` | `1` |

---

## 🚀 Quick Setup & Installation

###  Install Dependencies
Ensure you have Python 3.9+ installed, then run:
```bash
pip install -r requirements.txt
```


---

## 🏋️ Model Training & In-Training Evaluation

Run `train.py` to train the model. **Evaluation on the held-out test split runs automatically right after training completes!**

```bash
python train.py --data-dir data --epochs 30 --batch-size 32 --model-type audio_resnet
```

### Command-Line Arguments
| Argument | Default | Description |
| :--- | :--- | :--- |
| `--data-dir` | `data` | Path to folder containing `01/` to `10/` subdirectories |
| `--metadata` | `None` | Path to `ff1010_metadata.csv` (auto-detected if inside `--data-dir`) |
| `--model-type` | `audio_resnet` | Architecture: `audio_resnet` or `audio_mobilenet` |
| `--epochs` | `30` | Maximum training epochs (Early stopping stops earlier if plateaued) |
| `--batch-size` | `32` | Training batch size |
| `--lr` | `0.001` | Initial Adam learning rate |
| `--early-stopping-patience`| `8` | Epochs to wait for `val_auc` improvement before halting |
| `--output-dir` | `output` | Directory where models, logs, and plots are stored |

---

## 📊 Automatic Post-Training Evaluation

When training finishes, `train.py` automatically:
1. Reloads the best checkpoint (`output/saved_models/best_bird_model.keras`).
2. Evaluates predictions on the **unseen held-out test split (15%)**.
3. Computes and logs:
   - **Accuracy**: Overall classification accuracy.
   - **ROC-AUC Score**: Area under the ROC curve (the gold standard DCASE benchmark metric).
   - **PR-AUC (Average Precision)**: Evaluates performance under class imbalance.
   - **Precision, Recall, and F1-Score**.
   - **Specificity**: Ability to reject environmental field noise.

### 🏆 Empirical Held-Out Test Results
| Metric | Test Set Score | Evaluation Insight |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.8972** | **DCASE Benchmark Metric** (Strong class separation across thresholds) |
| **PR-AUC (Avg Precision)** | **0.8225** | High Precision-Recall AUC under ~25% positive class imbalance |
| **Recall (Sensitivity)** | **86.21%** | Detects over 86% of bird vocalizations in noisy field audio |
| **Accuracy** | **75.82%** | Overall correct binary classification |
| **Specificity** | **72.34%** | Rejects ~72.3% of ambient noise (rain, wind, machinery) |
| **F1-Score** | **0.6418** | Harmonic mean of precision and recall at threshold $\tau = 0.5$ |
| **Precision** | **51.12%** | Tunable via `--threshold` in `inference.py` |
4. Generates and saves high-resolution visual plots into `output/results/`:
   - `confusion_matrix.png`: Heatmap with Raw counts & Normalized percentages.
   - `roc_curve.png`: Receiver Operating Characteristic curve.
   - `precision_recall_curve.png`: Precision vs. Recall trade-off curve.
   - `training_curves.png`: Loss, Accuracy, and AUC progress across epochs.
5. Profiles **Model Complexity & Latency**:
   - Total Parameters & Trainable Parameters.
   - Multiply-Accumulate Operations (**MACs**) and **FLOPs**.
   - Inference Latency benchmark (single sample latency in milliseconds, P50, P95).
6. Exports full results to:
   `output/results/evaluation_summary.json`

---

## 📈 TensorBoard Visualization

To monitor loss curves, accuracy, AUC, and weight distributions in real-time:
```bash
tensorboard --logdir output/logs/fit
```
Then open your browser and navigate to `http://localhost:6006`.

---

## 🔍 Standalone Evaluation

To re-evaluate any saved checkpoint on test audio at any time:
```bash
python evaluate.py --model output/saved_models/best_bird_model.keras --data-dir data
```
This produces an updated evaluation summary and regenerates the metric plots.

---

## 🎧 Inference on Unseen Audio Files

### 1. Single Audio File Inference
```bash
python inference.py --model output/saved_models/best_bird_model.keras --audio path/to/sample.wav --visualize
```

**Example Terminal Output:**
```text
===========================================================================
                     BIRD CLASSIFICATION INFERENCE
===========================================================================
 File:       sample.wav
 Verdict:    [BIRD DETECTED]
 Confidence: 94.25%  (P=0.9425)
 Latency:    11.8 ms  (Extract: 6.9 ms, Model: 4.9 ms)
 Plot Saved: output/visualizations/sample_spec.png
---------------------------------------------------------------------------
```

### 2. Batch Inference on a Directory
```bash
python inference.py --model output/saved_models/best_bird_model.keras --audio-dir unseen_audios/ --output-json predictions.json
```

---

## 🧠 Neural Architecture & Computational Profile

### 1. Audio Feature Extraction
- **Input Audio**: 10-second WAV files resampled to $22,050\text{ Hz}$ (mono).
- **Transformation**: Log-Mel Spectrogram via Short-Time Fourier Transform (STFT):
  - FFT size ($N_{\text{fft}}$): $2048$
  - Hop length: $512$ samples
  - Mel frequency bands: $128$ bands spanning $50\text{ Hz} - 11,000\text{ Hz}$
  - Resulting 2D Representation: `(128, 431, 1)`
- **Augmentation (Train split only)**:
  - **SpecAugment**: Random Frequency Masking ($F \le 16$) and Time Masking ($T \le 24$).
  - **AWGN**: Additive White Gaussian Noise simulating field microphone variations.

### 2. AudioResNet Architecture
- **Residual Blocks**: 4 stages ($32 \to 64 \to 128 \to 256$ filters) with skip connections to mitigate vanishing gradients.
- **Global Average Pooling (GAP)**: Replaces dense flattening, dramatically lowering parameter count while enforcing temporal shift invariance.
- **Complexity**:
  - **Parameters**: $\approx 1.25\text{ Million}$
  - **Computational MACs**: $\approx 450\text{ M-MACs}$ ($\approx 0.90\text{ GFLOPs}$)
  - **Inference Latency**: $\approx 5 - 12\text{ ms}$ on standard CPU (suitable for real-time edge processing).
