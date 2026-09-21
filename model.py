import time
import logging
from typing import Tuple, Dict, Any

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def residual_block(x: tf.Tensor, filters: int, stride: int = 1) -> tf.Tensor:
    """
    Standard Residual block with shortcut projection if spatial dimensions or channels change.
    Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> Add(shortcut) -> ReLU
    """
    shortcut = x

    # First conv block
    y = layers.Conv2D(
        filters, 
        kernel_size=(3, 3), 
        strides=stride, 
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4)
    )(x)
    y = layers.BatchNormalization()(y)
    y = layers.ReLU()(y)

    # Second conv block
    y = layers.Conv2D(
        filters, 
        kernel_size=(3, 3), 
        strides=1, 
        padding="same",
        use_bias=False,
        kernel_regularizer=regularizers.l2(1e-4)
    )(y)
    y = layers.BatchNormalization()(y)

    # Adjust shortcut if dimensions differ
    in_channels = x.shape[-1]
    if stride != 1 or in_channels != filters:
        shortcut = layers.Conv2D(
            filters, 
            kernel_size=(1, 1), 
            strides=stride, 
            padding="same",
            use_bias=False
        )(x)
        shortcut = layers.BatchNormalization()(shortcut)

    out = layers.add([shortcut, y])
    out = layers.ReLU()(out)
    return out


def build_audio_resnet(
    input_shape: Tuple[int, int, int] = (128, 431, 1),
    num_classes: int = 1,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    Builds the AudioResNet architecture:
    - Input: Log-Mel Spectrogram (128 Mel bands x 431 time frames x 1 channel)
    - 4 Residual Stages: 32 -> 64 -> 128 -> 256 filters
    - Downsampling via strided convs and MaxPool2D
    - Global Average Pooling 2D to prevent overfitting & ensure temporal translation invariance
    - Binary classification head with Sigmoid activation
    """
    inputs = layers.Input(shape=input_shape, name="audio_spectrogram_input")

    # Initial Convolution & Downsample
    x = layers.Conv2D(
        32, 
        kernel_size=(5, 5), 
        strides=(2, 2), 
        padding="same", 
        use_bias=False
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # Stage 1: 32 filters
    x = residual_block(x, filters=32, stride=1)
    x = residual_block(x, filters=32, stride=1)

    # Stage 2: 64 filters (downsampling)
    x = residual_block(x, filters=64, stride=2)
    x = residual_block(x, filters=64, stride=1)

    # Stage 3: 128 filters (downsampling)
    x = residual_block(x, filters=128, stride=2)
    x = residual_block(x, filters=128, stride=1)

    # Stage 4: 256 filters (downsampling)
    x = residual_block(x, filters=256, stride=2)
    x = residual_block(x, filters=256, stride=1)

    # Regularization & Classification Head
    x = layers.SpatialDropout2D(dropout_rate)(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)

    # Output: Probability of bird presence
    outputs = layers.Dense(num_classes, activation="sigmoid", name="hasbird_prediction")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="AudioResNet")
    return model


def build_audio_mobilenet(
    input_shape: Tuple[int, int, int] = (128, 431, 1),
    num_classes: int = 1,
    dropout_rate: float = 0.3
) -> tf.keras.Model:
    """
    Builds an Audio MobileNetV2 architecture:
    - Replicates 1-channel spectrogram to 3 channels via 1x1 Conv
    - MobileNetV2 backbone (inverted residual depthwise separable convs)
    - Low computational footprint (ideal for edge and embedded deployment)
    """
    inputs = layers.Input(shape=input_shape, name="audio_spectrogram_input")

    # Project 1 channel to 3 channels for MobileNetV2 compatibility
    x = layers.Conv2D(3, kernel_size=(1, 1), padding="same", name="channel_expander")(inputs)

    # MobileNetV2 backbone
    backbone = tf.keras.applications.MobileNetV2(
        input_tensor=x,
        include_top=False,
        weights=None  # Trained from scratch on audio or can load pre-trained
    )
    x = backbone.output

    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="sigmoid", name="hasbird_prediction")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="AudioMobileNetV2")
    return model


# ==============================================================================
# Model Complexity Profiling (Parameters, MACs / FLOPs, Latency)
# ==============================================================================

def count_parameters(model: tf.keras.Model) -> Dict[str, int]:
    """
    Calculates total, trainable, and non-trainable parameter counts.
    """
    trainable_count = int(np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]))
    non_trainable_count = int(np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]))
    total_count = trainable_count + non_trainable_count
    return {
        "total_params": total_count,
        "trainable_params": trainable_count,
        "non_trainable_params": non_trainable_count
    }


def estimate_macs(model: tf.keras.Model, input_shape: Tuple[int, int, int] = (128, 431, 1)) -> Dict[str, Any]:
    """
    Analytically calculates MACs (Multiply-Accumulate Operations) and FLOPs for the model.
    Standard convention:
      - 1 MAC ≈ 2 FLOPs (1 multiply + 1 accumulate)
      - Conv2D: MACs = H_out * W_out * C_out * (K_h * K_w * C_in)
      - Dense:  MACs = in_features * out_features
    """
    total_macs = 0

    # Ensure model is built
    if not model.built:
        dummy_input = tf.zeros((1, *input_shape))
        model(dummy_input)

    for layer in model.layers:
        if isinstance(layer, layers.Conv2D):
            # Output dimensions
            out_shape = layer.output.shape
            h_out = out_shape[1] if out_shape[1] is not None else 1
            w_out = out_shape[2] if out_shape[2] is not None else 1
            c_out = layer.filters
            k_h, k_w = layer.kernel_size
            c_in = layer.input.shape[-1]
            macs = int(h_out * w_out * c_out * (k_h * k_w * c_in))
            total_macs += macs

        elif isinstance(layer, layers.DepthwiseConv2D):
            out_shape = layer.output.shape
            h_out = out_shape[1] if out_shape[1] is not None else 1
            w_out = out_shape[2] if out_shape[2] is not None else 1
            c_in = layer.input.shape[-1]
            k_h, k_w = layer.kernel_size
            macs = int(h_out * w_out * c_in * (k_h * k_w))
            total_macs += macs

        elif isinstance(layer, layers.Dense):
            in_features = int(layer.input.shape[-1])
            out_features = int(layer.units)
            macs = in_features * out_features
            total_macs += macs

    total_flops = total_macs * 2

    return {
        "total_macs": total_macs,
        "macs_m": total_macs / 1e6,       # Mega-MACs
        "total_flops": total_flops,
        "flops_g": total_flops / 1e9,     # Giga-FLOPs
    }


def benchmark_inference_latency(
    model: tf.keras.Model,
    input_shape: Tuple[int, int, int] = (128, 431, 1),
    batch_size: int = 1,
    num_runs: int = 50,
    warmup_runs: int = 10
) -> Dict[str, float]:
    """
    Measures CPU / GPU inference latency for a single audio sample (batch_size=1).
    Returns Mean, Median (P50), P95, and Throughput (samples/sec).
    """
    dummy_input = tf.random.normal((batch_size, *input_shape))

    # Warmup runs to initialize CUDA context / TF graph caching
    for _ in range(warmup_runs):
        _ = model(dummy_input, training=False)

    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = model(dummy_input, training=False)
        end = time.perf_counter()
        latencies.append((end - start) * 1000.0)  # in milliseconds

    latencies = np.array(latencies)
    mean_lat = float(np.mean(latencies))
    p50_lat = float(np.percentile(latencies, 50))
    p95_lat = float(np.percentile(latencies, 95))
    p99_lat = float(np.percentile(latencies, 99))
    throughput = float(1000.0 / mean_lat * batch_size)

    return {
        "mean_latency_ms": round(mean_lat, 2),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "p99_latency_ms": round(p99_lat, 2),
        "throughput_samples_per_sec": round(throughput, 1)
    }


def get_model(
    model_type: str = "audio_resnet",
    input_shape: Tuple[int, int, int] = (128, 431, 1),
    learning_rate: float = 1e-3
) -> tf.keras.Model:
    """
    Factory function to instantiate and compile the desired model architecture.
    """
    if model_type.lower() == "audio_resnet":
        model = build_audio_resnet(input_shape=input_shape)
    elif model_type.lower() == "audio_mobilenet":
        model = build_audio_mobilenet(input_shape=input_shape)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'audio_resnet' or 'audio_mobilenet'.")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.BinaryCrossentropy()
    
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc", curve="ROC"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


if __name__ == "__main__":
    print("Building AudioResNet model...")
    model = get_model("audio_resnet")
    model.summary()

    params = count_parameters(model)
    macs_info = estimate_macs(model)
    print("\n--- Model Complexity Profile ---")
    print(f"Total Parameters:      {params['total_params']:,}")
    print(f"Trainable Parameters:  {params['trainable_params']:,}")
    print(f"Estimated MACs:        {macs_info['macs_m']:.2f} M-MACs")
    print(f"Estimated FLOPs:       {macs_info['flops_g']:.3f} GFLOPs")
