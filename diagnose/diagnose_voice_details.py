# diagnostics/diagnose_voice_detailed.py - Detailed voice model diagnostics
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
if "diagnostics" in ROOT_PATH:
    ROOT_PATH = os.path.abspath(os.path.join(ROOT_PATH, os.pardir))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from models.voice_emotion_model import VoiceEmotionModel
from utils.data_preprocessing import load_voice_dataset
from config import config


def analyze_predictions(y_true, y_pred, emotions):
    """Analyze prediction patterns in detail."""
    
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    
    print("\n" + "="*70)
    print("PREDICTION PATTERN ANALYSIS")
    print("="*70)
    
    # Overall accuracy
    accuracy = np.mean(true_labels == pred_labels)
    print(f"\n📊 Overall Accuracy: {accuracy*100:.2f}%")
    
    # Per-class analysis
    print(f"\n📊 Per-Class Performance:")
    print(f"{'Emotion':<12} {'Samples':>8} {'Correct':>8} {'Accuracy':>10} {'Status':>10}")
    print("-" * 60)
    
    for i, emotion in enumerate(emotions):
        mask = true_labels == i
        n_samples = np.sum(mask)
        if n_samples > 0:
            n_correct = np.sum(pred_labels[mask] == i)
            acc = n_correct / n_samples
            
            # Status indicator
            if acc > 0.7:
                status = "✅ Good"
            elif acc > 0.4:
                status = "⚠️  Low"
            else:
                status = "❌ Bad"
            
            print(f"{emotion:<12} {n_samples:>8} {n_correct:>8} {acc*100:>9.1f}% {status:>10}")
    
    # Check for collapsed predictions
    print(f"\n📊 Prediction Distribution:")
    unique, counts = np.unique(pred_labels, return_counts=True)
    total = len(pred_labels)
    
    collapsed = False
    for i, emotion in enumerate(emotions):
        count = counts[unique == i][0] if i in unique else 0
        pct = count / total * 100
        
        # Flag if one class dominates
        if pct > 50:
            print(f"   {emotion}: {count} ({pct:.1f}%) ❌ COLLAPSED!")
            collapsed = True
        elif pct > 30:
            print(f"   {emotion}: {count} ({pct:.1f}%) ⚠️  High")
        else:
            print(f"   {emotion}: {count} ({pct:.1f}%)")
    
    if collapsed:
        print("\n❌ MODEL HAS COLLAPSED - Predicting mostly one class!")
        print("   This indicates severe overfitting.")
        print("   Retrain with stronger regularization and data augmentation.")
    
    # Confidence analysis
    print(f"\n📊 Confidence Analysis:")
    confidences = np.max(y_pred, axis=1)
    print(f"   Mean confidence: {np.mean(confidences):.3f}")
    print(f"   Median confidence: {np.median(confidences):.3f}")
    print(f"   High confidence (>0.9): {np.sum(confidences > 0.9)} samples ({np.sum(confidences > 0.9)/len(confidences)*100:.1f}%)")
    print(f"   Very high (>0.99): {np.sum(confidences > 0.99)} samples ({np.sum(confidences > 0.99)/len(confidences)*100:.1f}%)")
    
    if np.mean(confidences) > 0.95:
        print("   ⚠️  WARNING: Model is overconfident - likely overfitting!")
    
    return accuracy, collapsed


def plot_confusion_matrix(y_true, y_pred, emotions, save_path='results/voice_confusion_matrix.png'):
    """Plot confusion matrix."""
    
    true_labels = np.argmax(y_true, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=emotions, yticklabels=emotions,
                cbar_kws={'label': 'Count'})
    plt.title('Voice Model Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Emotion', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Emotion', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrix saved to {save_path}")
    plt.close()


def diagnose_voice_model():
    """Complete diagnostic of voice model."""
    
    print("\n" + "="*70)
    print("VOICE MODEL DIAGNOSTIC TOOL")
    print("="*70)
    
    emotions = config.EMOTIONS
    model_path = config.VOICE_MODEL_PATH
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Model not found: {model_path}")
        print("   Please train the model first!")
        return
    
    # Load model
    print(f"\n📂 Loading model from: {model_path}")
    try:
        model = VoiceEmotionModel(num_classes=len(emotions), input_shape=(40, 128, 1))
        model.load_model(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test data
    print(f"\n📂 Loading test dataset...")
    try:
        _, _, X_test, y_test = load_voice_dataset(
            'datasets/voice_emotions/',
            sample_rate=22050,
            duration=3,
            n_mfcc=40,
            n_mels=128,
            emotions=emotions
        )
        
        # Ensure correct shape
        if X_test.ndim == 2:
            X_test = np.expand_dims(X_test, axis=-1)
        if X_test.ndim == 3:
            X_test = np.expand_dims(X_test, axis=0)
        
        print(f"✅ Loaded {len(X_test)} test samples")
        print(f"   Shape: {X_test.shape}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Get predictions
    print(f"\n🔮 Getting model predictions...")
    y_pred = model.model.predict(X_test, verbose=0)
    print("✅ Predictions obtained")
    
    # Analyze predictions
    accuracy, collapsed = analyze_predictions(y_test, y_pred, emotions)
    
    # Plot confusion matrix
    print(f"\n📊 Generating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, emotions)
    
    # Detailed classification report
    print(f"\n" + "="*70)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*70)
    true_labels = np.argmax(y_test, axis=1)
    pred_labels = np.argmax(y_pred, axis=1)
    print(classification_report(true_labels, pred_labels, target_names=emotions))
    
    # Final diagnosis
    print(f"\n" + "="*70)
    print("FINAL DIAGNOSIS")
    print("="*70)
    
    if accuracy > 0.7:
        print("✅ Model is working well!")
        print(f"   Test accuracy: {accuracy*100:.2f}%")
        print("   Ready for deployment.")
    elif accuracy > 0.4 and not collapsed:
        print("⚠️  Model has moderate performance")
        print(f"   Test accuracy: {accuracy*100:.2f}%")
        print("   Recommendations:")
        print("   - Collect more training data")
        print("   - Try different architectures")
        print("   - Fine-tune hyperparameters")
    elif collapsed:
        print("❌ MODEL IS BROKEN - COLLAPSED")
        print(f"   Test accuracy: {accuracy*100:.2f}%")
        print("   The model is predicting mostly one class.")
        print("\n   IMMEDIATE ACTIONS REQUIRED:")
        print("   1. Delete the current model file")
        print("   2. Use the fixed training script with:")
        print("      - Stronger regularization (L2=0.01, Dropout=0.6)")
        print("      - More data augmentation (augment_factor=5)")
        print("      - Class balancing (oversample minority classes)")
        print("      - Focal loss for class imbalance")
        print("      - Label smoothing (0.15)")
        print("   3. Use simpler architecture (simple_cnn)")
        print("   4. Train with lower learning rate (5e-5)")
        print("   5. Use larger batch size (32)")
    else:
        print("❌ Model has poor performance")
        print(f"   Test accuracy: {accuracy*100:.2f}%")
        print("   Please retrain with:")
        print("   - More training data")
        print("   - Stronger augmentation")
        print("   - Better regularization")
        print("   - Check data preprocessing")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    diagnose_voice_model()