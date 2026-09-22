import os
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import librosa
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Default Audio & Feature Hyperparameters
DEFAULT_CONFIG = {
    "sample_rate": 22050,       # 22.05 kHz captures bird vocalizations up to ~11 kHz
    "duration": 10.0,           # Freefield1010 clips are 10.0 seconds
    "target_length": 220500,    # sample_rate * duration
    "n_mels": 128,              # Number of Mel frequency bands
    "n_fft": 2048,              # FFT window size
    "hop_length": 512,          # Hop length (stride in samples)
    "fmin": 50,                 # Minimum frequency in Hz
    "fmax": 11000,              # Maximum frequency in Hz
    "batch_size": 32,
    "seed": 42
}


def scan_audio_files(data_dir: str) -> Dict[str, str]:
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    audio_map = {}
    wav_files = list(data_path.glob("**/*.wav"))
    
    for wav_file in wav_files:
        # Item ID is typically the filename stem (e.g., '12345' from '12345.wav')
        item_id = wav_file.stem
        audio_map[str(item_id)] = str(wav_file.resolve())
        
    logger.info(f"Found {len(audio_map)} .wav files across subdirectories in {data_dir}")
    return audio_map


def load_and_align_metadata( data_dir: str, metadata_path: Optional[str] = None ) -> pd.DataFrame:
    # Locate metadata file
    if metadata_path is None or not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Could not locate metadata CSV in {data_dir}. "
            f"Please pass --metadata /path/to/ff1010_metadata.csv"
        )

    logger.info(f"Loading metadata from: {metadata_path}")
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
    
    initial_count = len(df)
    df = df.dropna(subset=["filepath"]).reset_index(drop=True)
    matched_count = len(df)
    logger.info(
        f"Metadata alignment: {matched_count}/{initial_count} records matched with audio files."
    )
    # Class distribution
    pos_count = (df["hasbird"] == 1).sum()
    neg_count = (df["hasbird"] == 0).sum()
    pos_ratio = (pos_count / matched_count * 100) if matched_count > 0 else 0
    neg_ratio = (neg_count / matched_count * 100) if matched_count > 0 else 0
    logger.info(
        f"Class Distribution: hasbird=0: {neg_count} ({neg_ratio:.1f}%), "
        f"hasbird=1: {pos_count} ({pos_ratio:.1f}%)"
    )
    return df

def extract_log_mel_spectrogram( file_path: str, sr: int = 22050, duration: float = 10.0, n_mels: int = 128, n_fft: int = 2048, hop_length: int = 512, fmin: int = 50, fmax: int = 11000 ) -> np.ndarray:
    
    target_samples = int(sr * duration)
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
    except Exception as e:
        y, file_sr = sf.read(file_path, dtype='float32')
        if y.ndim > 1:
            y = np.mean(y, axis=1) 
        if file_sr != sr:
            y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)

    # Pad or trim to target length (10.0 seconds = 220,500 samples)
    if len(y) < target_samples:
        pad_width = target_samples - len(y)
        y = np.pad(y, (0, pad_width), mode='constant')
    elif len(y) > target_samples:
        # Center crop or trim
        y = y[:target_samples]

    mel_spectrogram = librosa.feature.melspectrogram( y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0 )

    # Convert to log scale (decibels)
    log_mel = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Min-Max Normalization to [0, 1] range for numerical stability in neural networks
    min_val = log_mel.min()
    max_val = log_mel.max()
    if max_val - min_val > 1e-6:
        log_mel = (log_mel - min_val) / (max_val - min_val)
    else:
        log_mel = np.zeros_like(log_mel)

    # Expand dimension for single-channel CNN input: (n_mels, time_steps, 1)
    log_mel = np.expand_dims(log_mel, axis=-1).astype(np.float32)
    return log_mel


def apply_spec_augment( spectrogram: np.ndarray, freq_mask_max: int = 16, 
                       time_mask_max: int = 24, noise_factor: float = 0.02 ) -> np.ndarray:
    spec = spectrogram.copy()
    n_mels, time_steps, _ = spec.shape

    # 1. Frequency Masking
    f = np.random.randint(0, freq_mask_max)
    if f > 0 and n_mels > f:
        f0 = np.random.randint(0, n_mels - f)
        spec[f0:f0 + f, :, :] = 0.0

    # 2. Time Masking
    t = np.random.randint(0, time_mask_max)
    if t > 0 and time_steps > t:
        t0 = np.random.randint(0, time_steps - t)
        spec[:, t0:t0 + t, :] = 0.0

    # 3. Additive Gaussian Noise
    if noise_factor > 0:
        noise = np.random.normal(0, noise_factor, spec.shape).astype(np.float32)
        spec = np.clip(spec + noise, 0.0, 1.0)

    return spec


class FreefieldAudioGenerator(tf.keras.utils.Sequence):

    def __init__(self, filepaths: List[str], labels: List[int], batch_size: int = 32, augment: bool = False,
                config: Optional[dict] = None, shuffle: bool = True ):
        self.filepaths = np.array(filepaths)
        self.labels = np.array(labels, dtype=np.float32)
        self.batch_size = batch_size
        self.augment = augment
        self.config = config or DEFAULT_CONFIG
        self.shuffle = shuffle
        self.indices = np.arange(len(self.filepaths))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self) -> int:
        return int(np.ceil(len(self.filepaths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_paths = self.filepaths[batch_idx]
        batch_labels = self.labels[batch_idx]

        batch_features = []
        for path in batch_paths:
            spec = extract_log_mel_spectrogram( path, sr=self.config["sample_rate"], duration=self.config["duration"], n_mels=self.config["n_mels"], n_fft=self.config["n_fft"], hop_length=self.config["hop_length"],nfmin=self.config["fmin"], fmax=self.config["fmax"] )
            if self.augment:
                spec = apply_spec_augment(spec)
            batch_features.append(spec)

        return np.array(batch_features, dtype=np.float32), np.array(batch_labels, dtype=np.float32)


def get_dataset_splits( df: pd.DataFrame, batch_size: int = 32, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15,config: Optional[dict] = None, seed: int = 42 ) -> Tuple[FreefieldAudioGenerator, FreefieldAudioGenerator, FreefieldAudioGenerator, dict]:
    
    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Splits must sum to 1.0"

    cfg = config or DEFAULT_CONFIG
    cfg["batch_size"] = batch_size

    # Split train + temp (val + test)
    train_df, temp_df = train_test_split( df, test_size=(val_ratio + test_ratio), random_state=seed, stratify=df["hasbird"] )

    # Split temp into val and test
    val_fraction = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split( temp_df, test_size=(1.0 - val_fraction), random_state=seed, stratify=temp_df["hasbird"] )

    split_info = {
        "total": len(df),
        "train_count": len(train_df),
        "train_pos": int((train_df["hasbird"] == 1).sum()),
        "train_neg": int((train_df["hasbird"] == 0).sum()),
        "val_count": len(val_df),
        "val_pos": int((val_df["hasbird"] == 1).sum()),
        "val_neg": int((val_df["hasbird"] == 0).sum()),
        "test_count": len(test_df),
        "test_pos": int((test_df["hasbird"] == 1).sum()),
        "test_neg": int((test_df["hasbird"] == 0).sum()),
        "test_df": test_df.copy()  # Kept for direct evaluation
    }

    logger.info("=== Dataset Split Summary ===")
    logger.info(f"Train: {len(train_df)} samples (Pos: {split_info['train_pos']}, Neg: {split_info['train_neg']})")
    logger.info(f"Val:   {len(val_df)} samples (Pos: {split_info['val_pos']}, Neg: {split_info['val_neg']})")
    logger.info(f"Test:  {len(test_df)} samples (Pos: {split_info['test_pos']}, Neg: {split_info['test_neg']})")

    train_gen = FreefieldAudioGenerator( filepaths=train_df["filepath"].tolist(), labels=train_df["hasbird"].tolist(), batch_size=batch_size,augment=True, config=cfg, shuffle=True )

    val_gen = FreefieldAudioGenerator(filepaths=val_df["filepath"].tolist(), labels=val_df["hasbird"].tolist(), batch_size=batch_size,augment=False, config=cfg, shuffle=False )

    test_gen = FreefieldAudioGenerator( filepaths=test_df["filepath"].tolist(), labels=test_df["hasbird"].tolist(), batch_size=batch_size, augment=False, config=cfg, shuffle=False )

    return train_gen, val_gen, test_gen, split_info