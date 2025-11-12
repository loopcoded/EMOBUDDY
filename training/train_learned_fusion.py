"""
training/train_learned_fusion.py - Train learned fusion model
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
if "training" in ROOT_PATH:
    ROOT_PATH = os.path.abspath(os.path.join(ROOT_PATH, os.pardir))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from models.face_emotion_model import FaceEmotionModelTL
from models.voice_emotion_model import VoiceEmotionModel
from models.emotion_fusion_enhanced import LearnedFusionModel
from utils.data_preprocessing import load_face_dataset, load_voice_dataset
from config import config


def collect_predictions(face_model, voice_model, X_face, X_voice):
    """
    Collect predictions from face and voice models.
    
    Returns:
        face_preds, voice_preds, face_conf, voice_conf
    """
    print("\nCollecting predictions from models...")
    
    # Face predictions
    face_preds = []
    face_conf = []
    
    print(f"Processing {len(X_face)} face samples...")
    for i, img in enumerate(X_face):
        pred = face_model.predict(img)
        face_preds.append(pred)
        face_conf.append(np.max(pred))
        
        if (i + 1) % 500 == 0:
            print(f"  Face: {i+1}/{len(X_face)}")
    
    face_preds = np.array(face_preds)
    face_conf = np.array(face_conf).reshape(-1, 1)
    
    # Voice predictions
    voice_preds = []
    voice_conf = []
    
    print(f"Processing {len(X_voice)} voice samples...")
    for i, audio in enumerate(X_voice):
        pred = voice_model.predict(audio)
        voice_preds.append(pred)
        voice_conf.append(np.max(pred))
        
        if (i + 1) % 500 == 0:
            print(f"  Voice: {i+1}/{len(X_voice)}")
    
    voice_preds = np.array(voice_preds)
    voice_conf = np.array(voice_conf).reshape(-1, 1)
    
    print("✅ Predictions collected")
    return face_preds, voice_preds, face_conf, voice_conf


def train_learned_fusion():
    """Train learned fusion model"""
    
    print("=" * 60)
    print("TRAINING LEARNED FUSION MODEL")
    print("=" * 60)
    
    emotions = config.EMOTIONS
    num_emotions = len(emotions)
    
    # Model paths
    face_model_path = 'static/models/face_emotion_mobilenet.keras'
    voice_model_path = 'static/models/voice_emotion_model_best.keras'
    fusion_model_path = 'static/models/learned_fusion_model.keras'
    
    # Check if base models exist
    if not os.path.exists(face_model_path):
        print(f"❌ Face model not found: {face_model_path}")
        print("   Please train face model first!")
        return
    
    if not os.path.exists(voice_model_path):
        print(f"❌ Voice model not found: {voice_model_path}")
        print("   Please train voice model first!")
        return
    
    # Load base models
    print("\n📂 Loading base models...")
    
    face_model = FaceEmotionModelTL(num_classes=num_emotions, input_shape=(96, 96, 3))
    face_model.load_model(face_model_path)
    print("✅ Face model loaded")
    
    voice_model = VoiceEmotionModel(num_classes=num_emotions, input_shape=(40, 128, 1))
    voice_model.load_model(voice_model_path)
    print("✅ Voice model loaded")
    
    # Load datasets
    print("\n📂 Loading datasets...")
    
    print("Loading face dataset...")
    X_face_train, y_face_train, X_face_test, y_face_test = load_face_dataset(
        'datasets/face_emotions/',
        img_size=(96, 96)
    )
    
    print("\nLoading voice dataset...")
    X_voice_train, y_voice_train, X_voice_test, y_voice_test = load_voice_dataset(
        'datasets/voice_emotions/',
        sample_rate=22050,
        duration=3,
        n_mfcc=40,
        n_mels=128
    )
    
    # Ensure voice data has correct shape
    if X_voice_train.ndim == 2:
        X_voice_train = np.expand_dims(X_voice_train, axis=-1)
    if X_voice_test.ndim == 2:
        X_voice_test = np.expand_dims(X_voice_test, axis=-1)
    
    # Use minimum number of samples
    min_train = min(len(X_face_train), len(X_voice_train))
    min_test = min(len(X_face_test), len(X_voice_test))
    
    print(f"\n📊 Using {min_train} training and {min_test} test paired samples")
    
    X_face_train = X_face_train[:min_train]
    y_face_train = y_face_train[:min_train]
    X_voice_train = X_voice_train[:min_train]
    y_voice_train = y_voice_train[:min_train]
    
    X_face_test = X_face_test[:min_test]
    y_face_test = y_face_test[:min_test]
    X_voice_test = X_voice_test[:min_test]
    y_voice_test = y_voice_test[:min_test]
    
    # Collect predictions from base models
    print("\n" + "=" * 60)
    print("COLLECTING BASE MODEL PREDICTIONS")
    print("=" * 60)
    
    print("\n📊 Training set predictions...")
    face_train_preds, voice_train_preds, face_train_conf, voice_train_conf = \
        collect_predictions(face_model, voice_model, X_face_train, X_voice_train)
    
    print("\n📊 Test set predictions...")
    face_test_preds, voice_test_preds, face_test_conf, voice_test_conf = \
        collect_predictions(face_model, voice_model, X_face_test, X_voice_test)
    
    # Split training data into train/val
    indices = np.arange(len(face_train_preds))
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=0.15, 
        stratify=np.argmax(y_face_train, axis=1),
        random_state=42
    )
    
    face_train_split = face_train_preds[train_idx]
    voice_train_split = voice_train_preds[train_idx]
    face_conf_train_split = face_train_conf[train_idx]
    voice_conf_train_split = voice_train_conf[train_idx]
    y_train_split = y_face_train[train_idx]
    
    face_val_split = face_train_preds[val_idx]
    voice_val_split = voice_train_preds[val_idx]
    face_conf_val_split = face_train_conf[val_idx]
    voice_conf_val_split = voice_train_conf[val_idx]
    y_val_split = y_face_train[val_idx]
    
    print(f"\n📊 Fusion training split:")
    print(f"   Training: {len(train_idx)} samples")
    print(f"   Validation: {len(val_idx)} samples")
    print(f"   Test: {len(face_test_preds)} samples")
    
    # Build and compile fusion model
    print("\n" + "=" * 60)
    print("BUILDING LEARNED FUSION MODEL")
    print("=" * 60)
    
    fusion_model = LearnedFusionModel(num_emotions=num_emotions)
    fusion_model.build_model()
    fusion_model.compile_model(lr=1e-3)
    
    print("\n📝 Fusion model architecture:")
    fusion_model.model.summary()
    
    # Train fusion model
    print("\n" + "=" * 60)
    print("TRAINING FUSION MODEL")
    print("=" * 60)
    
    history = fusion_model.train(
        face_train_split,
        voice_train_split,
        face_conf_train_split,
        voice_conf_train_split,
        y_train_split,
        validation_data=(
            [face_val_split, voice_val_split, face_conf_val_split, voice_conf_val_split],
            y_val_split
        ),
        epochs=100,
        batch_size=64
    )
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING FUSION MODEL")
    print("=" * 60)
    
    test_results = fusion_model.model.evaluate(
        [face_test_preds, voice_test_preds, face_test_conf, voice_test_conf],
        y_face_test,
        verbose=0
    )
    
    print("\n🏆 Test Set Performance:")
    print(f"   Loss: {test_results[0]:.4f}")
    print(f"   Accuracy: {test_results[1]:.4f}")
    print(f"   Top-2 Accuracy: {test_results[2]:.4f}")
    print(f"   Precision: {test_results[3]:.4f}")
    print(f"   Recall: {test_results[4]:.4f}")
    
    # Compare with baseline strategies
    print("\n" + "=" * 60)
    print("COMPARING WITH BASELINE STRATEGIES")
    print("=" * 60)
    
    # Face only
    face_only_acc = np.mean(
        np.argmax(face_test_preds, axis=1) == np.argmax(y_face_test, axis=1)
    )
    print(f"\n📷 Face only: {face_only_acc:.4f}")
    
    # Voice only
    voice_only_acc = np.mean(
        np.argmax(voice_test_preds, axis=1) == np.argmax(y_face_test, axis=1)
    )
    print(f"🎤 Voice only: {voice_only_acc:.4f}")
    
    # Simple weighted average
    weighted_preds = 0.6 * face_test_preds + 0.4 * voice_test_preds
    weighted_acc = np.mean(
        np.argmax(weighted_preds, axis=1) == np.argmax(y_face_test, axis=1)
    )
    print(f"⚖️  Weighted average (0.6/0.4): {weighted_acc:.4f}")
    
    # Learned fusion
    learned_acc = test_results[1]
    print(f"🧠 Learned fusion: {learned_acc:.4f}")
    
    print(f"\n📈 Improvement over face only: {(learned_acc - face_only_acc):.4f}")
    print(f"📈 Improvement over weighted: {(learned_acc - weighted_acc):.4f}")
    
    # Save model
    os.makedirs('static/models', exist_ok=True)
    fusion_model.save(fusion_model_path)
    print(f"\n💾 Fusion model saved to: {fusion_model_path}")
    
    # Plot training history
    print("\n📊 Generating training plots...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax = axes[0]
    ax.plot(history.history['accuracy'], label='Training', linewidth=2)
    ax.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Accuracy', fontweight='bold')
    ax.set_title('Model Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Loss
    ax = axes[1]
    ax.plot(history.history['loss'], label='Training', linewidth=2)
    ax.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Loss', fontweight='bold')
    ax.set_title('Model Loss', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/fusion_training_history.png', dpi=300, bbox_inches='tight')
    print("✅ Training plot saved to results/fusion_training_history.png")
    
    plt.show()
    
    print("\n" + "=" * 60)
    print("✅ FUSION MODEL TRAINING COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    train_learned_fusion()