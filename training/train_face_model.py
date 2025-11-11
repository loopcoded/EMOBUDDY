import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models.face_emotion_model import FaceEmotionModel, get_data_augmentation
from utils.data_preprocessing import load_face_dataset
from config import config

# --- Configuration ---
DATA_PATH = 'datasets/face_emotions/'
MODEL_SAVE_PATH = config.FACE_MODEL_PATH
IMG_SIZE = config.FACE_IMAGE_SIZE
EMOTIONS = config.EMOTIONS
NUM_CLASSES = config.NUM_EMOTIONS

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 15

def train_face_model():
    """Load data, build, compile, and train the FaceEmotionModel."""
    print("--- Starting Face Emotion Model Training ---")
    
    # 1. Load Data (Preprocessed & Normalized)
    try:
        X_train, y_train, X_test, y_test = load_face_dataset(
            DATA_PATH, img_size=IMG_SIZE, emotions=EMOTIONS
        )
        
        print(f"\nTraining data shape: {X_train.shape}, Labels: {y_train.shape}")
        print(f"Testing data shape: {X_test.shape}, Labels: {y_test.shape}")
    except Exception as e:
        print(f"ERROR: Could not load dataset. Ensure data is in {DATA_PATH} with correct structure.")
        print(f"Detail: {e}")
        return

    # 2. Build Model
    model_instance = FaceEmotionModel(
        num_classes=NUM_CLASSES, 
        input_shape=(*IMG_SIZE, 3)
    )
    
    # Use transfer learning for best results on small, specialized datasets
    model = model_instance.build_model(use_transfer_learning=False)
    model_instance.compile_model(learning_rate=LEARNING_RATE)
    
    print("\nModel Summary:")
    model_instance.get_model_summary()

    # 3. Data Augmentation and Generator
    # Using ImageDataGenerator for simple flow, but custom tf.data pipeline 
    # with `get_data_augmentation()` is also recommended.
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # 4. Training
    print("\n--- Model Training Started ---")
    
    # Get callbacks
    callbacks = model_instance.get_callbacks(MODEL_SAVE_PATH, patience=PATIENCE)
    
    # Fit model using the generator
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        callbacks=callbacks
    )
    
    print("\n--- Training Complete ---")
    
    # 5. Evaluate and Save
    try:
        # Load the best model saved by the callback
        model_instance.load_model(MODEL_SAVE_PATH) 
        loss, acc, top2_acc, precision, recall = model.evaluate(X_test, y_test, verbose=0)
        
        print("\nFinal Evaluation on Test Data:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {acc*100:.2f}%")
        print(f"  Top-2 Accuracy: {top2_acc*100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"\nBEST Model saved to: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"ERROR during evaluation or model loading: {e}")

if __name__ == '__main__':
    train_face_model()