"""
Improved training script with better handling of overfitting
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
if "training" in ROOT_PATH:
    ROOT_PATH = os.path.abspath(os.path.join(ROOT_PATH, os.pardir))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)
from models.face_emotion_model import FaceEmotionModel, get_data_augmentation
from utils.data_preprocessing import load_face_dataset
from config import config

# --- GPU Configuration ---
def setup_gpu():
    """Configure GPU settings for optimal performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus, 'GPU')
            print(f"✅ GPU Setup Complete - Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                print(f"   {gpu}")
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
    else:
        print("⚠️ No GPU found - running on CPU")

# --- Configuration ---
DATA_PATH = 'datasets/face_emotions/'
MODEL_SAVE_PATH = config.FACE_MODEL_PATH
IMG_SIZE = config.FACE_IMAGE_SIZE
EMOTIONS = config.EMOTIONS
NUM_CLASSES = config.NUM_EMOTIONS

"""
Improved training script with better handling of overfitting
"""

import os
import sys
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

# --- Assume config.py and data_preprocessing.py exist ---
# Mocking config for demonstration
class MockConfig:
    FACE_MODEL_PATH = 'static/models/face_emotion_model_best.keras'
    FACE_IMAGE_SIZE = (48, 48)
    EMOTIONS = ['angry', 'happy', 'disgust', 'fear', 'sad', 'surprise', 'neutral']
    NUM_EMOTIONS = 7
config = MockConfig()
# --- End Mock ---

# We assume 'models' is in the same directory or accessible
from face_emotion_model import FaceEmotionModel, get_data_augmentation, FocalLoss
# from utils.data_preprocessing import load_face_dataset # Assuming this works

# --- GPU Configuration ---
def setup_gpu():
    """Configure GPU settings for optimal performance."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(gpus, 'GPU')
            print(f"✅ GPU Setup Complete - Found {len(gpus)} GPU(s)")
            for gpu in gpus:
                print(f"   {gpu}")
        except RuntimeError as e:
            print(f"❌ GPU configuration error: {e}")
    else:
        print("⚠️ No GPU found - running on CPU")

# --- Configuration ---
DATA_PATH = 'datasets/face_emotions/'
MODEL_SAVE_PATH = config.FACE_MODEL_PATH
IMG_SIZE = config.FACE_IMAGE_SIZE
EMOTIONS = config.EMOTIONS
NUM_CLASSES = config.NUM_EMOTIONS

# Hyperparameters - ADJUSTED
BATCH_SIZE = 64 # INCREASED for more stable training
EPOCHS = 100
LEARNING_RATE = 0.0005 
PATIENCE = 20
VALIDATION_SPLIT = 0.15 
USE_FOCAL_LOSS = True
LABEL_SMOOTHING = 0.1

def train_face_model():
    """Load data, build, compile, and train the FaceEmotionModel."""
    print("="*70)
    print("IMPROVED FACE EMOTION MODEL TRAINING")
    print("="*70)
    
    # Setup GPU
    setup_gpu()
    
    # Ensure model directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    # 1. Load Data
    print("\n📂 Loading dataset...")
    try:
        # Mock data loading since I don't have your function
        # X_train_full, y_train_full, X_test, y_test = load_face_dataset(...)
        
        # Using your provided stats to create mock data
        print("--- MOCK DATA GENERATION (using your stats) ---")
        train_full_size = 25109 + 4431
        test_size = 8009
        X_train_full = np.random.rand(train_full_size, 48, 48, 3).astype(np.float32)
        y_train_full = keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, train_full_size), num_classes=NUM_CLASSES)
        X_test = np.random.rand(test_size, 48, 48, 3).astype(np.float32)
        y_test = keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, test_size), num_classes=NUM_CLASSES)
        print("--- END MOCK DATA ---")

        
        # Create validation split from training data
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, 
            test_size=VALIDATION_SPLIT, 
            random_state=42,
            stratify=np.argmax(y_train_full, axis=1)
        )
        
        print(f"✅ Training data: {X_train.shape}")
        print(f"✅ Validation data: {X_val.shape}")
        print(f"✅ Test data: {X_test.shape}")
        
    except Exception as e:
        print(f"❌ ERROR: Could not load dataset from {DATA_PATH}")
        print(f"   Detail: {e}")
        return

    # 2. Build Model
    print("\n🏗️ Building simplified model...")
    model_instance = FaceEmotionModel(
        num_classes=NUM_CLASSES, 
        input_shape=(*IMG_SIZE, 3)
    )
    
    model = model_instance.build_model()
    model_instance.compile_model(
        learning_rate=LEARNING_RATE,
        label_smoothing=LABEL_SMOOTHING,
        use_focal_loss=USE_FOCAL_LOSS
    )
    
    print("\n📊 Model Summary:")
    model_instance.get_model_summary()
    
    # Count parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    print(f"\n   Total trainable parameters: {trainable_params:,} (Original was 13M+)")

    # 3. Data Augmentation using tf.data pipeline
    print("\n🔄 Setting up data augmentation...")
    
    augmentation = get_data_augmentation()
    
    def augment_data(images, labels):
        images = augmentation(images, training=True)
        return images, labels
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_dataset = (train_dataset
                    .shuffle(buffer_size=len(X_train))
                    .batch(BATCH_SIZE)
                    .map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
                    .prefetch(tf.data.AUTOTUNE))
    
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # 4. Training
    print("\n" + "="*70)
    print("🚀 TRAINING STARTED")
    print("="*70)
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Learning rate: {LEARNING_RATE}")
    print(f"   Use Focal Loss: {USE_FOCAL_LOSS}")
    print("="*70 + "\n")
    
    callbacks = model_instance.get_callbacks(MODEL_SAVE_PATH, patience=PATIENCE)
    
    try:
        fit_kwargs = {
            'epochs': EPOCHS,
            'validation_data': val_dataset,
            'callbacks': callbacks,
            'verbose': 1
        }
        
        # *** KEY CHANGE ***
        # Do NOT use class_weight if using FocalLoss. FocalLoss handles imbalance.
        if not USE_FOCAL_LOSS:
            print("⚖️ Using class weights (FocalLoss is OFF)")
            class_weights = compute_class_weights(y_train) # Your function
            fit_kwargs['class_weight'] = class_weights
        else:
            print("⚖️ Using FocalLoss (Class weights are OFF)")
            
        history = model.fit(
            train_dataset,
            **fit_kwargs
        )
        
        print("\n✅ Training Complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 6. Final Evaluation
    print("\n" + "="*70)
    print("📊 FINAL EVALUATION (Corrected Logic)")
    print("="*70)
    
    try:
        # *** KEY CHANGE: Fixed Evaluation Logic ***
        print(f"\nLoading best model from: {MODEL_SAVE_PATH}")
        
        # 1. Create a new model instance
        model_instance_eval = FaceEmotionModel(
            num_classes=NUM_CLASSES, 
            input_shape=(*IMG_SIZE, 3)
        )
        
        # 2. Load the *saved* model (this loads architecture, weights, and optimizer)
        model_instance_eval.load_model(MODEL_SAVE_PATH)
        
        # 3. Re-compile the *loaded* model. This is crucial to attach
        #    fresh, state-less metrics for evaluation.
        print("Re-compiling loaded model for evaluation...")
        model_instance_eval.compile_model(
            learning_rate=LEARNING_RATE, # LR doesn't matter for eval, but metrics do
            label_smoothing=LABEL_SMOOTHING,
            use_focal_loss=USE_FOCAL_LOSS
        )
        
        # Define metric names exactly as they appear in model.compile
        metric_names = ['loss', 'accuracy', 'top_2_accuracy'] 

        def print_evaluation(name, X, y):
            print(f"\n📈 {name} Set:")
            results = model_instance_eval.model.evaluate(X, y, verbose=0, batch_size=BATCH_SIZE)
            
            for name, value in zip(metric_names, results):
                if 'acc' in name.lower():
                    print(f"   {name.capitalize():20s}: {value*100:.2f}%")
                else:
                    print(f"   {name.capitalize():20s}: {value:.4f}")
            return results

        val_results = print_evaluation("Validation", X_val, y_val)
        test_results = print_evaluation("Test (Final Performance)", X_test, y_test)

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

# This is your function, included for completeness
def compute_class_weights(y_train):
    class_indices = np.argmax(y_train, axis=1)
    class_counts = Counter(class_indices)
    total = len(class_indices)
    class_weights = {}
    print("\n⚖️ Class Weights:")
    for class_idx in range(NUM_CLASSES):
        count = class_counts.get(class_idx, 1)
        weight = total / (NUM_CLASSES * count)
        class_weights[class_idx] = weight
        print(f"   {EMOTIONS[class_idx]:12s}: {weight:.3f}")
    return class_weights

if __name__ == '__main__':
    train_face_model()