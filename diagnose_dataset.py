"""
diagnostics/diagnose_voice_model.py - Diagnose voice model issues
"""
import os
import sys
import numpy as np
from collections import Counter

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
if "diagnostics" in ROOT_PATH:
    ROOT_PATH = os.path.abspath(os.path.join(ROOT_PATH, os.pardir))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from models.voice_emotion_model import VoiceEmotionModel
from utils.data_preprocessing import load_voice_dataset
from config import config


def diagnose_voice_model():
    """Diagnose issues with voice emotion model"""
    
    print("=" * 60)
    print("VOICE MODEL DIAGNOSTICS")
    print("=" * 60)
    
    emotions = config.EMOTIONS
    
    # Load model
    print("\n1. Loading voice model...")
    model_path = 'static/models/voice_emotion_model_best.keras'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    try:
        voice_model = VoiceEmotionModel(
            num_classes=len(emotions),
            input_shape=(40, 128, 1)
        )
        voice_model.load_model(model_path)
        print("✅ Model loaded")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load test data
    print("\n2. Loading test data...")
    _, _, X_test, y_test = load_voice_dataset(
        'datasets/voice_emotions/',
        sample_rate=22050,
        duration=3,
        n_mfcc=40,
        n_mels=128
    )
    
    if X_test.ndim == 2:
        X_test = np.expand_dims(X_test, axis=-1)
    
    print(f"   Test samples: {len(X_test)}")
    print(f"   Test shape: {X_test.shape}")
    
    # Check true label distribution
    print("\n3. True label distribution:")
    y_true_classes = np.argmax(y_test, axis=1)
    true_counts = Counter(y_true_classes)
    for cls_idx, count in sorted(true_counts.items()):
        print(f"   {emotions[cls_idx]:10s}: {count:4d} samples")
    
    # Get predictions
    print("\n4. Getting model predictions...")
    predictions = []
    raw_outputs = []
    
    for i in range(len(X_test)):
        pred = voice_model.predict(X_test[i])
        predictions.append(np.argmax(pred))
        raw_outputs.append(pred)
        
        if (i + 1) % 100 == 0:
            print(f"   Processed: {i+1}/{len(X_test)}")
    
    predictions = np.array(predictions)
    raw_outputs = np.array(raw_outputs)
    
    # Check prediction distribution
    print("\n5. Predicted label distribution:")
    pred_counts = Counter(predictions)
    for cls_idx, count in sorted(pred_counts.items()):
        print(f"   {emotions[cls_idx]:10s}: {count:4d} samples ({count/len(predictions)*100:.1f}%)")
    
    # Check if model is stuck
    unique_predictions = len(set(predictions))
    print(f"\n6. Unique predictions: {unique_predictions} out of {len(emotions)} possible")
    
    if unique_predictions == 1:
        print("   ⚠️ WARNING: Model is predicting ONLY ONE CLASS!")
        print("   This indicates severe model collapse.")
    elif unique_predictions < len(emotions) / 2:
        print(f"   ⚠️ WARNING: Model only uses {unique_predictions} out of {len(emotions)} classes")
    
    # Analyze raw outputs
    print("\n7. Raw output statistics:")
    print(f"   Min probability: {raw_outputs.min():.6f}")
    print(f"   Max probability: {raw_outputs.max():.6f}")
    print(f"   Mean probability: {raw_outputs.mean():.6f}")
    print(f"   Std probability: {raw_outputs.std():.6f}")
    
    # Check for extreme probabilities (softmax saturation)
    max_probs = np.max(raw_outputs, axis=1)
    print(f"\n8. Confidence distribution:")
    print(f"   Mean max prob: {max_probs.mean():.4f}")
    print(f"   Median max prob: {np.median(max_probs):.4f}")
    print(f"   Probabilities > 0.99: {np.sum(max_probs > 0.99)} samples")
    print(f"   Probabilities > 0.999: {np.sum(max_probs > 0.999)} samples")
    
    if max_probs.mean() > 0.99:
        print("   ⚠️ WARNING: Model is overconfident (softmax saturation)")
    
    # Sample predictions
    print("\n9. Sample predictions (first 10):")
    for i in range(min(10, len(X_test))):
        true_idx = y_true_classes[i]
        pred_idx = predictions[i]
        confidence = max_probs[i]
        
        status = "✓" if true_idx == pred_idx else "✗"
        print(f"   {status} Sample {i}: True={emotions[true_idx]:10s} | "
              f"Pred={emotions[pred_idx]:10s} | Conf={confidence:.4f}")
    
    # Accuracy per class
    print("\n10. Per-class accuracy:")
    for cls_idx, emotion in enumerate(emotions):
        mask = y_true_classes == cls_idx
        if np.sum(mask) == 0:
            continue
        
        class_acc = np.sum(predictions[mask] == cls_idx) / np.sum(mask)
        print(f"   {emotion:10s}: {class_acc:.4f} ({np.sum(mask)} samples)")
    
    # Overall accuracy
    accuracy = np.sum(predictions == y_true_classes) / len(y_true_classes)
    print(f"\n11. Overall accuracy: {accuracy:.4f}")
    
    # Diagnosis summary
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)
    
    issues = []
    
    if unique_predictions <= 2:
        issues.append("❌ CRITICAL: Model collapsed to predicting 1-2 classes only")
        issues.append("   → Retrain with class weights or balanced sampling")
    
    if max_probs.mean() > 0.99:
        issues.append("❌ CRITICAL: Softmax saturation (overconfidence)")
        issues.append("   → Add temperature scaling or label smoothing")
    
    if accuracy < 0.3:
        issues.append("❌ CRITICAL: Accuracy below random guessing")
        issues.append("   → Check data preprocessing and model architecture")
    
    # Check for class imbalance
    max_count = max(true_counts.values())
    min_count = min(true_counts.values())
    if max_count / min_count > 5:
        issues.append(f"⚠️  WARNING: Severe class imbalance ({max_count}/{min_count} = {max_count/min_count:.1f}x)")
        issues.append("   → Use class weights or oversampling")
    
    if not issues:
        print("✅ No critical issues detected!")
    else:
        print("\nIssues found:")
        for issue in issues:
            print(issue)
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    print("\n1. Immediate fixes:")
    print("   - Retrain voice model with balanced class weights")
    print("   - Add label smoothing (0.1)")
    print("   - Increase dropout (0.5+)")
    print("   - Use focal loss to handle class imbalance")
    
    print("\n2. Data improvements:")
    print("   - Balance dataset using oversampling for minority classes")
    print("   - Add data augmentation (noise, time stretch, pitch shift)")
    print("   - Verify audio preprocessing is consistent")
    
    print("\n3. Model improvements:")
    print("   - Reduce model capacity to prevent overfitting")
    print("   - Add batch normalization")
    print("   - Use learning rate scheduling")
    
    print("\n4. For fusion:")
    print("   - Until voice model is fixed, use face-only predictions")
    print("   - Or set face_weight=0.9, voice_weight=0.1 in fusion")


if __name__ == "__main__":
    diagnose_voice_model()