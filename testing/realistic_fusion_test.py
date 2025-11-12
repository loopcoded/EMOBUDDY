"""
testing/realistic_fusion_test.py - Test fusion with reliability detection
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
if "testing" in ROOT_PATH:
    ROOT_PATH = os.path.abspath(os.path.join(ROOT_PATH, os.pardir))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from models.face_emotion_model import FaceEmotionModelTL
from models.voice_emotion_model import VoiceEmotionModel
from utils.data_preprocessing import load_face_dataset, load_voice_dataset
from config import config


def detect_broken_modality(predictions, y_true):
    """
    Detect if a modality is broken/unreliable.
    
    Returns:
        dict with diagnostic info
    """
    y_pred = np.argmax(predictions, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    
    # Check accuracy
    accuracy = np.mean(y_pred == y_true_classes)
    
    # Check prediction diversity
    unique_predictions = len(np.unique(y_pred))
    total_classes = predictions.shape[1]
    diversity_ratio = unique_predictions / total_classes
    
    # Check confidence distribution
    max_confidences = np.max(predictions, axis=1)
    mean_confidence = np.mean(max_confidences)
    
    # Check for mode collapse (predicting mostly one class)
    from collections import Counter
    pred_counts = Counter(y_pred)
    most_common_count = pred_counts.most_common(1)[0][1]
    mode_collapse_ratio = most_common_count / len(y_pred)
    
    is_broken = False
    issues = []
    
    if accuracy < 0.25:  # Worse than random for 7 classes (1/7 ≈ 0.14)
        is_broken = True
        issues.append(f"Very low accuracy: {accuracy:.2%}")
    
    if diversity_ratio < 0.4:  # Using less than 40% of classes
        is_broken = True
        issues.append(f"Low prediction diversity: {diversity_ratio:.2%}")
    
    if mode_collapse_ratio > 0.7:  # More than 70% same prediction
        is_broken = True
        issues.append(f"Mode collapse: {mode_collapse_ratio:.2%} predict same class")
    
    if mean_confidence > 0.98:  # Overconfident
        is_broken = True
        issues.append(f"Overconfident: {mean_confidence:.2%}")
    
    return {
        'is_broken': is_broken,
        'accuracy': accuracy,
        'diversity': diversity_ratio,
        'mean_confidence': mean_confidence,
        'mode_collapse': mode_collapse_ratio,
        'issues': issues
    }


def adaptive_fusion(face_preds, voice_preds, face_info, voice_info, y_true):
    """
    Adaptive fusion that adjusts weights based on modality reliability.
    """
    # Determine weights based on reliability
    if face_info['is_broken'] and voice_info['is_broken']:
        print("⚠️ Both modalities broken! Using equal weights")
        face_weight = 0.5
        voice_weight = 0.5
    elif face_info['is_broken']:
        print("⚠️ Face broken, using voice only")
        face_weight = 0.0
        voice_weight = 1.0
    elif voice_info['is_broken']:
        print("⚠️ Voice broken, using face only")
        face_weight = 1.0
        voice_weight = 0.0
    else:
        # Both working, weight by accuracy
        total_acc = face_info['accuracy'] + voice_info['accuracy']
        face_weight = face_info['accuracy'] / total_acc
        voice_weight = voice_info['accuracy'] / total_acc
        print(f"✅ Both modalities working")
        print(f"   Face weight: {face_weight:.2%} (acc: {face_info['accuracy']:.2%})")
        print(f"   Voice weight: {voice_weight:.2%} (acc: {voice_info['accuracy']:.2%})")
    
    # Fuse predictions
    fused = face_weight * face_preds + voice_weight * voice_preds
    
    # Evaluate
    y_pred = np.argmax(fused, axis=1)
    y_true_classes = np.argmax(y_true, axis=1)
    accuracy = np.mean(y_pred == y_true_classes)
    
    return {
        'accuracy': accuracy,
        'face_weight': face_weight,
        'voice_weight': voice_weight,
        'predictions': fused
    }


def main():
    print("=" * 60)
    print("REALISTIC FUSION TEST WITH BROKEN MODALITY DETECTION")
    print("=" * 60)
    
    emotions = config.EMOTIONS
    
    # Load models
    print("\n1. Loading models...")
    
    face_model = FaceEmotionModelTL(num_classes=len(emotions), input_shape=(96, 96, 3))
    face_model.load_model('static/models/face_emotion_mobilenet.keras')
    print("✅ Face model loaded")
    
    voice_model = VoiceEmotionModel(num_classes=len(emotions), input_shape=(40, 128, 1))
    voice_model.load_model('static/models/voice_emotion_model_best.keras')
    print("✅ Voice model loaded")
    
    # Load test data
    print("\n2. Loading test data...")
    _, _, X_face_test, y_face_test = load_face_dataset(
        'datasets/face_emotions/', img_size=(96, 96)
    )
    _, _, X_voice_test, y_voice_test = load_voice_dataset(
        'datasets/voice_emotions/', sample_rate=22050, duration=3, n_mfcc=40, n_mels=128
    )
    
    if X_voice_test.ndim == 2:
        X_voice_test = np.expand_dims(X_voice_test, axis=-1)
    
    # Use minimum samples
    min_samples = min(len(X_face_test), len(X_voice_test))
    X_face_test = X_face_test[:min_samples]
    y_face_test = y_face_test[:min_samples]
    X_voice_test = X_voice_test[:min_samples]
    
    print(f"   Using {min_samples} paired samples")
    
    # Get predictions
    print("\n3. Getting predictions...")
    
    face_preds = []
    for i in range(len(X_face_test)):
        pred = face_model.predict(X_face_test[i])
        face_preds.append(pred)
        if (i + 1) % 100 == 0:
            print(f"   Face: {i+1}/{len(X_face_test)}")
    face_preds = np.array(face_preds)
    
    voice_preds = []
    for i in range(len(X_voice_test)):
        pred = voice_model.predict(X_voice_test[i])
        voice_preds.append(pred)
        if (i + 1) % 100 == 0:
            print(f"   Voice: {i+1}/{len(X_voice_test)}")
    voice_preds = np.array(voice_preds)
    
    # Diagnose modalities
    print("\n" + "=" * 60)
    print("4. MODALITY DIAGNOSTICS")
    print("=" * 60)
    
    print("\n📷 FACE MODALITY:")
    face_info = detect_broken_modality(face_preds, y_face_test)
    print(f"   Status: {'❌ BROKEN' if face_info['is_broken'] else '✅ WORKING'}")
    print(f"   Accuracy: {face_info['accuracy']:.2%}")
    print(f"   Diversity: {face_info['diversity']:.2%}")
    print(f"   Confidence: {face_info['mean_confidence']:.2%}")
    if face_info['issues']:
        print("   Issues:")
        for issue in face_info['issues']:
            print(f"     - {issue}")
    
    print("\n🎤 VOICE MODALITY:")
    voice_info = detect_broken_modality(voice_preds, y_face_test)
    print(f"   Status: {'❌ BROKEN' if voice_info['is_broken'] else '✅ WORKING'}")
    print(f"   Accuracy: {voice_info['accuracy']:.2%}")
    print(f"   Diversity: {voice_info['diversity']:.2%}")
    print(f"   Confidence: {voice_info['mean_confidence']:.2%}")
    if voice_info['issues']:
        print("   Issues:")
        for issue in voice_info['issues']:
            print(f"     - {issue}")
    
    # Adaptive fusion
    print("\n" + "=" * 60)
    print("5. ADAPTIVE FUSION")
    print("=" * 60)
    
    fusion_result = adaptive_fusion(
        face_preds, voice_preds, face_info, voice_info, y_face_test
    )
    
    print(f"\n🔀 Fusion result:")
    print(f"   Accuracy: {fusion_result['accuracy']:.2%}")
    print(f"   Face weight: {fusion_result['face_weight']:.2%}")
    print(f"   Voice weight: {fusion_result['voice_weight']:.2%}")
    
    # Compare
    print("\n" + "=" * 60)
    print("6. COMPARISON")
    print("=" * 60)
    
    print(f"\n📊 Results:")
    print(f"   Face only:     {face_info['accuracy']:.2%}")
    print(f"   Voice only:    {voice_info['accuracy']:.2%}")
    print(f"   Adaptive fusion: {fusion_result['accuracy']:.2%}")
    
    improvement = fusion_result['accuracy'] - max(face_info['accuracy'], voice_info['accuracy'])
    print(f"\n   Improvement: {improvement:+.2%}")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("7. RECOMMENDATIONS")
    print("=" * 60)
    
    if voice_info['is_broken']:
        print("\n⚠️ VOICE MODEL NEEDS FIXING:")
        print("   1. Run: python diagnostics/diagnose_voice_model.py")
        print("   2. Retrain with balanced class weights")
        print("   3. Add more data augmentation")
        print("   4. Use focal loss for class imbalance")
        print("\n   Until fixed:")
        print("   - Use face-only predictions")
        print("   - Or set fusion weights: face=0.95, voice=0.05")
    
    if face_info['is_broken']:
        print("\n⚠️ FACE MODEL NEEDS FIXING:")
        print("   - Retrain with more data or better augmentation")
    
    if not voice_info['is_broken'] and not face_info['is_broken']:
        print("\n✅ Both modalities working!")
        print("   - Fusion should improve performance")
        print("   - Consider training learned fusion model")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Face\nOnly', 'Voice\nOnly', 'Adaptive\nFusion']
    accuracies = [
        face_info['accuracy'],
        voice_info['accuracy'],
        fusion_result['accuracy']
    ]
    colors = ['#FF6B6B', '#4ECDC4', '#95E1D3']
    
    bars = ax.bar(methods, accuracies, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1%}',
               ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax.set_ylabel('Accuracy', fontweight='bold', fontsize=12)
    ax.set_title('Multi-Modal Fusion Performance\n(with Broken Modality Detection)',
                 fontweight='bold', fontsize=14)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add status indicators
    status_text = []
    if face_info['is_broken']:
        status_text.append("⚠️ Face: BROKEN")
    else:
        status_text.append("✅ Face: OK")
    
    if voice_info['is_broken']:
        status_text.append("⚠️ Voice: BROKEN")
    else:
        status_text.append("✅ Voice: OK")
    
    ax.text(0.02, 0.98, '\n'.join(status_text),
           transform=ax.transAxes,
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10)
    
    plt.tight_layout()
    
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/realistic_fusion_test.png', dpi=300, bbox_inches='tight')
    print("\n📊 Plot saved to: results/realistic_fusion_test.png")
    
    plt.show()


if __name__ == "__main__":
    main()