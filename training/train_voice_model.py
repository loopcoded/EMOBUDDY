# training/train_voice_model.py â€“ Colab-ready training script
import sys
import os

print("PYTHONPATH:", sys.path)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
if "training" in ROOT_PATH:
    ROOT_PATH = os.path.abspath(os.path.join(ROOT_PATH, os.pardir))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
from models.voice_emotion_model import VoiceEmotionModel
from utils.data_preprocessing import load_voice_dataset   # must return one-hot labels
from config import config

# --------------------------- configuration --------------------------- #
DATA_PATH        = "datasets/voice_emotions"
MODEL_SAVE_PATH  = config.VOICE_MODEL_PATH        # e.g., "/content/voice_emotion_model_best.keras"
EMOTIONS         = config.EMOTIONS                # class list
NUM_CLASSES      = config.NUM_EMOTIONS

SAMPLE_RATE      = config.VOICE_SAMPLE_RATE       # e.g., 22050
DURATION         = config.VOICE_DURATION          # e.g., 3 seconds
N_MFCC           = config.VOICE_N_MFCC            # e.g., 40

BATCH_SIZE       = 16  # Reduced from 32 to help with small dataset
EPOCHS           = 100  # Reduced from 120
LEARNING_RATE    = 1e-4  # Reduced from 5e-4 for more stable training
PATIENCE         = 15  # Increased from 12
ARCHITECTURE     = "cnn_lstm"   # "cnn", "cnn_lstm", or "attention"

# Optional: mixed precision for speed on T4/A100
if tf.config.list_physical_devices("GPU"):
    try:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        pass

def _per_sample_standardize(X):
    """Standardize each sample to zero mean / unit variance across (freq,time)."""
    # X shape: (N, n_mfcc, time, 1)
    eps = 1e-6
    mean = X.mean(axis=(1,2,3), keepdims=True)
    std  = X.std(axis=(1,2,3), keepdims=True)
    return (X - mean) / (std + eps)

def _compute_class_weights(y_onehot):
    """
    Compute class weights for imbalanced data.
    y_onehot: shape (N, C)
    """
    counts = y_onehot.sum(axis=0)
    total  = counts.sum()
    weights = total / (len(counts) * counts + 1e-8)
    return {i: float(weights[i]) for i in range(len(counts))}

def add_augmentation(X, y, augment_factor=2):
    """
    Simple data augmentation for audio:
    - Add small amounts of noise
    - Time shift
    """
    X_aug = []
    y_aug = []
    
    for i in range(len(X)):
        # Original sample
        X_aug.append(X[i])
        y_aug.append(y[i])
        
        # Create augmented versions
        for _ in range(augment_factor - 1):
            sample = X[i].copy()
            
            # Add random noise (10% of samples)
            if np.random.random() > 0.5:
                noise = np.random.normal(0, 0.005, sample.shape)
                sample = sample + noise
            
            # Random time shift (small)
            if np.random.random() > 0.5:
                shift = np.random.randint(-5, 5)
                sample = np.roll(sample, shift, axis=1)
            
            X_aug.append(sample)
            y_aug.append(y[i])
    
    return np.array(X_aug), np.array(y_aug)

def train_voice_model():
    print("--- Starting Voice Emotion Model Training ---")
    print("âœ… TensorFlow:", tf.__version__)
    print("âœ… GPU:", tf.config.list_physical_devices('GPU'))

    # ---------------------------- load data ---------------------------- #
    try:
        # X: (N, n_mfcc, time_steps), y: one-hot (N, C)
        X_train, y_train, X_test, y_test = load_voice_dataset(
            DATA_PATH,
            sample_rate=SAMPLE_RATE,
            duration=DURATION,
            n_mfcc=N_MFCC,
            emotions=EMOTIONS,
        )

        # add channel dim for Conv2D
        X_train = np.expand_dims(X_train, axis=-1)
        X_test  = np.expand_dims(X_test, axis=-1)

        # standardize per sample (helps with amplitude variation)
        X_train = _per_sample_standardize(X_train)
        X_test  = _per_sample_standardize(X_test)

        # Data augmentation to combat overfitting
        print("\nðŸ”„ Applying data augmentation...")
        X_train, y_train = add_augmentation(X_train, y_train, augment_factor=2)
        
        # Create validation split from training data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=np.argmax(y_train, axis=1)
        )

        # sanity logs
        _, n_mfcc, time_steps, ch = X_train.shape
        input_shape = (n_mfcc, time_steps, ch)
        print(f"\nâœ… Calculated Input Shape: {input_shape}")
        print(f"Training data shape: {X_train.shape}, Labels: {y_train.shape}")
        print(f"Validation data shape: {X_val.shape}, Labels: {y_val.shape}")
        print(f"Testing  data shape: {X_test.shape},  Labels: {y_test.shape}")
    except Exception as e:
        print(f"âŒ ERROR: Could not load voice dataset at {DATA_PATH}. Detail: {e}")
        return

    # --------------------------- class weights ------------------------- #
    class_weight = _compute_class_weights(y_train)
    print("\nâš–ï¸ Class weights:", class_weight)

    # ------------------------------ model ------------------------------ #
    # Delete old model file to avoid loading issues
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"\nðŸ—‘ï¸ Removing old model file: {MODEL_SAVE_PATH}")
        os.remove(MODEL_SAVE_PATH)
    
    model_instance = VoiceEmotionModel(num_classes=NUM_CLASSES, input_shape=input_shape)
    model = model_instance.build_model(
        architecture=ARCHITECTURE,
        l2=1e-3,  # Increased regularization
        dropout=0.5  # Increased dropout
    )
    model_instance.compile_model(
        learning_rate=LEARNING_RATE, 
        label_smoothing=0.1  # Increased label smoothing
    )

    print(f"\nðŸ“Š Model Summary (Architecture: {ARCHITECTURE}):")
    model_instance.get_model_summary()

    # ----------------------------- training ---------------------------- #
    callbacks = model_instance.get_callbacks(MODEL_SAVE_PATH, patience=PATIENCE)

    # shuffle once up-front (in case loader didn't)
    idx = np.arange(len(X_train))
    np.random.shuffle(idx)
    X_train, y_train = X_train[idx], y_train[idx]

    print("\nðŸ”¥ Training startedâ€¦\n")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),  # Use validation split, not test
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    print("\nâœ… --- Training Complete ---")

    # ---------------------------- evaluation --------------------------- #
    try:
        # Build a fresh model with the same architecture
        print("\nðŸ”„ Loading best model...")
        model_instance_eval = VoiceEmotionModel(num_classes=NUM_CLASSES, input_shape=input_shape)
        model_instance_eval.build_model(
            architecture=ARCHITECTURE,
            l2=1e-3,
            dropout=0.5
        )
        model_instance_eval.load_model(MODEL_SAVE_PATH)
        
        # Evaluate on test set
        print("\nðŸ“Š Evaluating on TEST set (unseen data):")
        results = model_instance_eval.model.evaluate(X_test, y_test, verbose=0)
        metric_names = ["Loss", "Accuracy", "Top-2 Acc", "Precision", "Recall", "AUC"]
        print("\nðŸ† Final Evaluation (best checkpoint on TEST set):")
        for name, val in zip(metric_names, results):
            if "Acc" in name or name in ("Accuracy", "Top-2 Acc"):
                print(f"{name:>12}: {val*100:.2f}%")
            else:
                print(f"{name:>12}: {val:.4f}")
        
        # Also evaluate on validation set
        print("\nðŸ“Š Evaluating on VALIDATION set:")
        results_val = model_instance_eval.model.evaluate(X_val, y_val, verbose=0)
        print("\nðŸ† Validation Performance:")
        for name, val in zip(metric_names, results_val):
            if "Acc" in name or name in ("Accuracy", "Top-2 Acc"):
                print(f"{name:>12}: {val*100:.2f}%")
            else:
                print(f"{name:>12}: {val:.4f}")
        
        print(f"\nðŸ’¾ Best model saved to: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"âŒ ERROR during evaluation or model loading: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # make sure output directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    train_voice_model()