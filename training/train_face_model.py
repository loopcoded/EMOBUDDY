# training/train_face_model_tl.py
import os
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
if "training" in ROOT_PATH:
    ROOT_PATH = os.path.abspath(os.path.join(ROOT_PATH, os.pardir))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

# Import your model & loader
from models.face_emotion_model import FaceEmotionModelTL, FocalLoss  # ensure filename matches
from utils.data_preprocessing import load_face_dataset

# -------------------------
# Config
# -------------------------
IMG_SIZE = (96, 96)   # backbone input
BATCH_SIZE = 64
EPOCHS_HEAD = 10      # head training epochs
EPOCHS_FINE = 20      # fine-tune epochs
LR_HEAD = 1e-3
LR_FINE = 1e-4
PATIENCE = 8
MODEL_SAVE_PATH = 'static/models/face_emotion_mobilenet.keras'
LABEL_SMOOTHING = 0.05
USE_FOCAL_LOSS = True
NUM_CLASSES = 7

# -------------------------
# Utility: balanced dataset creation
# -------------------------
def make_balanced_dataset(X, y, batch_size):
    labels = np.argmax(y, axis=1)
    counts = Counter(labels)
    max_count = max(counts.values())
    parts = []
    for cls in range(y.shape[1]):
        idx = np.where(labels == cls)[0]
        if len(idx) == 0:
            continue
        reps = int(np.ceil(max_count / len(idx)))
        tiled = np.tile(idx, reps)[:max_count]
        parts.append(tiled)
    all_idx = np.concatenate(parts)
    np.random.shuffle(all_idx)
    Xb = X[all_idx]
    yb = y[all_idx]
    ds = tf.data.Dataset.from_tensor_slices((Xb, yb))
    ds = ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# -------------------------
# Mixup (tf implementation)
# -------------------------
def mixup_batch(images, labels, alpha=0.2):
    """Mixup in TF (batch-level). images: [B,H,W,C], labels: [B,classes]"""
    batch_size = tf.shape(images)[0]
    # Beta distribution via gamma trick for each sample
    lam = tf.random.gamma(shape=[batch_size], alpha=alpha)
    lam = tf.cast(lam, tf.float32)
    lam_x = tf.reshape(lam, (batch_size,1,1,1))
    lam_y = tf.reshape(lam, (batch_size,1))
    idx = tf.random.shuffle(tf.range(batch_size))
    x2 = tf.gather(images, idx)
    y2 = tf.gather(labels, idx)
    mixed_x = images * lam_x + x2 * (1.0 - lam_x)
    mixed_y = labels * lam_y + y2 * (1.0 - lam_y)
    return mixed_x, mixed_y

# -------------------------
# Augmentation pipeline (expects float images in [0,1])
# -------------------------
def get_augmentation():
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.12),
        keras.layers.RandomZoom(0.12),
        keras.layers.RandomContrast(0.12),
    ])

# -------------------------
# Preprocess images with cv2 (numpy)
# -------------------------
def preprocess_images_numpy(X, target_size=IMG_SIZE):
    """Resize and ensure RGB (numpy array) — returns float32 array"""
    out = np.zeros((len(X), target_size[0], target_size[1], 3), dtype=np.float32)
    for i, img in enumerate(X):
        # If single channel, convert accordingly
        if img is None:
            continue
        # If image is float in [0,1], convert to 0-255 for cv2
        if img.dtype == np.float32 or img.dtype == np.float64:
            tmp = (img * 255.0).astype(np.uint8) if img.max() <= 1.1 else img.astype(np.uint8)
        else:
            tmp = img.astype(np.uint8)
        # If grayscale (H,W) or (H,W,1)
        if tmp.ndim == 2:
            tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2RGB)
        elif tmp.shape[-1] == 1:
            tmp = cv2.cvtColor(tmp[:,:,0], cv2.COLOR_GRAY2RGB)
        elif tmp.shape[-1] == 3:
            # convert BGR (cv2 default) to RGB
            tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2RGB)
        else:
            # fallback: convert to RGB by slicing or expanding
            tmp = tmp[..., :3]
        resized = cv2.resize(tmp, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        out[i] = resized.astype(np.float32)
    return out
# -------------------------
# Dataset preparation (pure TensorFlow)
# -------------------------
def prepare_datasets(X_train, y_train, X_val, y_val, X_test, y_test):
    """Creates tf.data pipelines with augmentation and mixup (graph-safe)."""
    # Preprocess numpy images (resize + RGB) -> returns float32 0..255
    X_train = preprocess_images_numpy(X_train, IMG_SIZE)
    X_val = preprocess_images_numpy(X_val, IMG_SIZE)
    X_test = preprocess_images_numpy(X_test, IMG_SIZE)

    # Balanced training dataset
    train_ds = make_balanced_dataset(X_train, y_train, BATCH_SIZE)

    # Data augmentation pipeline
    aug = get_augmentation()

    @tf.function
    def augment_and_mixup(x, y):
        """Runs augmentation + mixup in TensorFlow graph."""
        x = tf.cast(x, tf.float32) / 255.0
        x = aug(x, training=True)

        # Mixup
        batch_size = tf.shape(x)[0]
        lam = tf.random.uniform([], 0.2, 0.8)
        idx = tf.random.shuffle(tf.range(batch_size))
        x2 = tf.gather(x, idx)
        y2 = tf.gather(y, idx)
        x = lam * x + (1 - lam) * x2
        y = lam * y + (1 - lam) * y2
        return x, y

    # Map augmentation safely inside TF graph
    train_ds = train_ds.map(augment_and_mixup, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    # Validation / test datasets: scale only
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val.astype(np.float32) / 255.0, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        tf.data.Dataset.from_tensor_slices((X_test.astype(np.float32) / 255.0, y_test))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, test_ds


# -------------------------
# Training loop (two-phase)
# -------------------------
def train():
    print("Loading dataset...")
    X_full, y_full, X_test, y_test = load_face_dataset('datasets/face_emotions/', img_size=(96,96), emotions=None)
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.15, stratify=np.argmax(y_full, axis=1), random_state=42)

    # debug prints
    print("Sample image shape:", X_train[0].shape)
    print("Pixel value range (train):", X_train.min(), X_train.max())

    # Prepare tf.data datasets
    train_ds, val_ds, test_ds = prepare_datasets(X_train, y_train, X_val, y_val, X_test, y_test)

    # Build model (head training)
    model_inst = FaceEmotionModelTL(num_classes=NUM_CLASSES, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    model = model_inst.build_model(backbone_trainable=False)
    model_inst.compile_model(lr=LR_HEAD, label_smoothing=LABEL_SMOOTHING, use_focal_loss=False)

    cbs = model_inst.get_callbacks(MODEL_SAVE_PATH, patience=PATIENCE)

    print("\n=== Phase 1: Train head (backbone frozen) ===")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, callbacks=cbs, verbose=1)

    # Phase 2: Fine-tune - unfreeze backbone (top layers) and keep BN frozen
    print("\n=== Phase 2: Fine-tune backbone (unfreeze top layers) ===")
    # Find base model inside model (Rescaling is usually layer 1, base is layer 2)
    base_model = None
    for layer in model.layers:
        # MobileNetV2 instance has attribute 'layers' and name 'mobilenetv2' typically
        if hasattr(layer, 'layers') and len(layer.layers) > 0 and 'mobilenet' in layer.__class__.__name__.lower() or 'mobilenetv2' in layer.name.lower():
            base_model = layer
            break
    # fallback: assume layer index 2
    if base_model is None and len(model.layers) > 2:
        base_model = model.layers[2]

    # Unfreeze the base model (you can unfreeze partially if desired)
    try:
        base_model.trainable = True
        # Freeze BatchNorm layers to avoid instability
        for l in base_model.layers:
            if isinstance(l, tf.keras.layers.BatchNormalization):
                l.trainable = False
        print("Base model set to trainable (BatchNorm layers frozen).")
    except Exception as e:
        print("Warning: could not set base model trainable via detected layer. Attempting fallback.")
        model.layers[2].trainable = True
        for l in model.layers[2].layers:
            if isinstance(l, tf.keras.layers.BatchNormalization):
                l.trainable = False

    # lower learning rate for fine-tuning
    model_inst.compile_model(lr=LR_FINE, label_smoothing=0.05, use_focal_loss=USE_FOCAL_LOSS)

    print(f"Starting fine-tune with lr={LR_FINE}, use_focal_loss={USE_FOCAL_LOSS}")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE, callbacks=cbs, verbose=1)

    # Load best model and evaluate
    try:
        model_inst.load_model(MODEL_SAVE_PATH)
        print("\nLoaded best model from checkpoint.")
    except Exception as e:
        print("Could not load best checkpoint, using current model instance. Detail:", e)

    print("\nValidation metrics:")
    print(model_inst.model.evaluate(val_ds))
    print("\nTest metrics:")
    print(model_inst.model.evaluate(test_ds))

if __name__ == "__main__":
    train()
