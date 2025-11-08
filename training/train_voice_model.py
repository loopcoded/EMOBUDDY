import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from models.voice_emotion_model import VoiceEmotionModel
from utils.data_preprocessing import load_voice_dataset
from config import config

# --- Configuration ---
DATA_PATH = 'datasets/voice_emotions/'
MODEL_SAVE_PATH = config.VOICE_MODEL_PATH
EMOTIONS = config.EMOTIONS
NUM_CLASSES = config.NUM_EMOTIONS

# Voice Feature Parameters (Must match data_preprocessing.py)
SAMPLE_RATE = config.VOICE_SAMPLE_RATE
DURATION = config.VOICE_DURATION
N_MFCC = config.VOICE_N_MFCC

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 0.0005
PATIENCE = 20
ARCHITECTURE = 'cnn_lstm' # Use cnn_lstm or attention

def train_voice_model():
    """Load data, build, compile, and train the VoiceEmotionModel."""
    print("--- Starting Voice Emotion Model Training ---")
    
    # 1. Load Data (MFCC features extracted and normalized)
    try:
        # X_train shape will be (samples, n_mfcc, time_steps)
        X_train, y_train, X_test, y_test = load_voice_dataset(
            DATA_PATH, 
            sample_rate=SAMPLE_RATE, 
            duration=DURATION, 
            n_mfcc=N_MFCC, 
            emotions=EMOTIONS
        )
        
        # Add channel dimension (required for Conv2D layers in the model)
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)

        # Determine input shape dynamically
        _, n_mfcc, time_steps, channels = X_train.shape
        INPUT_SHAPE = (n_mfcc, time_steps, channels)

        print(f"\nCalculated Input Shape: {INPUT_SHAPE}")
        print(f"Training data shape: {X_train.shape}, Labels: {y_train.shape}")
        print(f"Testing data shape: {X_test.shape}, Labels: {y_test.shape}")
    except Exception as e:
        print(f"ERROR: Could not load voice dataset. Ensure data is in {DATA_PATH}.")
        print(f"Detail: {e}")
        return

    # 2. Build Model
    model_instance = VoiceEmotionModel(
        num_classes=NUM_CLASSES, 
        input_shape=INPUT_SHAPE
    )
    
    model = model_instance.build_model(architecture=ARCHITECTURE)
    model_instance.compile_model(learning_rate=LEARNING_RATE)
    
    print(f"\nModel Summary (Architecture: {ARCHITECTURE}):")
    model_instance.get_model_summary()

    # 3. Training
    print("\n--- Model Training Started ---")
    
    # Get callbacks
    callbacks = model_instance.get_callbacks(MODEL_SAVE_PATH, patience=PATIENCE)
    
    # Fit model
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    print("\n--- Training Complete ---")
    
    # 4. Evaluate and Save
    try:
        # Load the best model saved by the callback
        model_instance.load_model(MODEL_SAVE_PATH)
        loss, acc, top2_acc, precision, recall, auc = model.evaluate(X_test, y_test, verbose=0)
        
        print("\nFinal Evaluation on Test Data (using best model):")
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {acc*100:.2f}%")
        print(f"  Top-2 Accuracy: {top2_acc*100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  AUC: {auc:.4f}")
        print(f"\nBEST Model saved to: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"ERROR during evaluation or model loading: {e}")

if __name__ == '__main__':
    train_voice_model()