"""
Dataset diagnostic tool to identify issues causing overfitting
"""

import os
import sys
from pathlib import Path
import numpy as np
import cv2
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import config

def analyze_dataset(dataset_path, emotions):
    """Analyze dataset for common issues."""
    
    print("="*70)
    print("DATASET DIAGNOSTIC TOOL")
    print("="*70)
    
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
    
    # Check if paths exist
    if not os.path.exists(train_path):
        print(f"❌ Training path not found: {train_path}")
        return
    if not os.path.exists(test_path):
        print(f"❌ Test path not found: {test_path}")
        return
    
    print(f"✅ Dataset path: {dataset_path}\n")
    
    # Analyze train and test splits
    train_stats = analyze_split(train_path, emotions, "TRAIN")
    test_stats = analyze_split(test_path, emotions, "TEST")
    
    # Check for issues
    print("\n" + "="*70)
    print("🔍 ISSUE DETECTION")
    print("="*70)
    
    issues_found = False
    
    # 1. Class imbalance
    print("\n1️⃣ Class Imbalance Check:")
    train_counts = train_stats['counts']
    test_counts = test_stats['counts']
    
    if train_counts:
        max_count = max(train_counts.values())
        min_count = min(train_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 3:
            print(f"   ⚠️  SEVERE imbalance detected! Ratio: {imbalance_ratio:.1f}:1")
            print(f"   Most common: {max(train_counts, key=train_counts.get)} ({max_count})")
            print(f"   Least common: {min(train_counts, key=train_counts.get)} ({min_count})")
            print("\n   💡 Solutions:")
            print("      - Use class_weights in training")
            print("      - Augment minority classes more")
            print("      - Collect more balanced data")
            issues_found = True
        else:
            print(f"   ✅ Acceptable balance (ratio: {imbalance_ratio:.1f}:1)")
    
    # 2. Dataset size
    print("\n2️⃣ Dataset Size Check:")
    total_train = sum(train_counts.values())
    total_test = sum(test_counts.values())
    
    if total_train < 1000:
        print(f"   ⚠️  VERY SMALL training set: {total_train} images")
        print("   💡 Minimum recommended: 1000+ images")
        print("      - Use heavy data augmentation")
        print("      - Consider transfer learning")
        print("      - Reduce model complexity")
        issues_found = True
    elif total_train < 5000:
        print(f"   ⚠️  Small training set: {total_train} images")
        print("   💡 Consider using data augmentation and regularization")
        issues_found = True
    else:
        print(f"   ✅ Good training set size: {total_train} images")
    
    if total_test < 200:
        print(f"   ⚠️  Small test set: {total_test} images")
        print("   💡 Recommended: 20-30% of training size")
        issues_found = True
    
    # 3. Train/Test split ratio
    print("\n3️⃣ Train/Test Split Check:")
    if total_train > 0 and total_test > 0:
        split_ratio = total_test / (total_train + total_test)
        if split_ratio < 0.1 or split_ratio > 0.4:
            print(f"   ⚠️  Unusual split ratio: {split_ratio*100:.1f}% test")
            print("   💡 Recommended: 15-30% test")
            issues_found = True
        else:
            print(f"   ✅ Good split ratio: {split_ratio*100:.1f}% test")
    
    # 4. Image quality issues
    print("\n4️⃣ Image Quality Check:")
    quality_issues = check_image_quality(train_path, emotions)
    if quality_issues:
        print("   ⚠️  Quality issues detected:")
        for issue in quality_issues:
            print(f"      - {issue}")
        issues_found = True
    else:
        print("   ✅ No obvious quality issues")
    
    # 5. Data leakage check
    print("\n5️⃣ Potential Data Leakage Check:")
    if check_potential_duplicates(train_path, test_path, emotions):
        print("   ⚠️  Possible duplicate images between train/test")
        print("   💡 This can cause overfitting - review your split")
        issues_found = True
    else:
        print("   ✅ No obvious duplicates detected")
    
    # Summary
    print("\n" + "="*70)
    if issues_found:
        print("⚠️  ISSUES FOUND - See recommendations above")
        print("\n📋 Quick Fixes to Try:")
        print("   1. Use the improved model (face_emotion_model_improved.py)")
        print("   2. Reduce batch size to 32 or 16")
        print("   3. Use stronger regularization (already in improved model)")
        print("   4. Reduce model complexity")
        print("   5. Add more data augmentation")
        print("   6. Use class weights for imbalanced classes")
    else:
        print("✅ Dataset looks good! If still overfitting:")
        print("   - Reduce model complexity")
        print("   - Increase dropout rates")
        print("   - Use stronger L2 regularization")
    print("="*70)


def analyze_split(split_path, emotions, split_name):
    """Analyze a single split (train or test)."""
    print(f"\n📊 {split_name} SET ANALYSIS:")
    print("-" * 70)
    
    counts = {}
    total = 0
    
    for emotion in emotions:
        emotion_path = os.path.join(split_path, emotion)
        if os.path.exists(emotion_path):
            files = [f for f in os.listdir(emotion_path) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            count = len(files)
            counts[emotion] = count
            total += count
            print(f"   {emotion:12s}: {count:5d} images")
        else:
            counts[emotion] = 0
            print(f"   {emotion:12s}: ❌ folder missing")
    
    print(f"   {'TOTAL':12s}: {total:5d} images")
    
    return {'counts': counts, 'total': total}


def check_image_quality(split_path, emotions):
    """Check for common image quality issues."""
    issues = []
    
    # Sample check on first 10 images of each emotion
    for emotion in emotions:
        emotion_path = os.path.join(split_path, emotion)
        if not os.path.exists(emotion_path):
            continue
        
        files = [f for f in os.listdir(emotion_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:10]
        
        for file in files:
            img_path = os.path.join(emotion_path, file)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    issues.append(f"Corrupted image: {emotion}/{file}")
                    continue
                
                # Check if image is too small
                if img.shape[0] < 48 or img.shape[1] < 48:
                    issues.append(f"Very small image: {emotion}/{file} ({img.shape})")
                
                # Check if image is grayscale when expecting color
                if len(img.shape) == 2:
                    issues.append(f"Grayscale image found: {emotion}/{file}")
                
            except Exception as e:
                issues.append(f"Error reading {emotion}/{file}: {e}")
    
    return issues[:5]  # Return first 5 issues


def check_potential_duplicates(train_path, test_path, emotions):
    """Simple check for potential duplicates (by filename)."""
    train_files = set()
    test_files = set()
    
    for emotion in emotions:
        train_emotion = os.path.join(train_path, emotion)
        test_emotion = os.path.join(test_path, emotion)
        
        if os.path.exists(train_emotion):
            train_files.update(os.listdir(train_emotion))
        
        if os.path.exists(test_emotion):
            test_files.update(os.listdir(test_emotion))
    
    duplicates = train_files.intersection(test_files)
    return len(duplicates) > 0


if __name__ == "__main__":
    # Analyze face dataset
    dataset_path = 'datasets/face_emotions/'
    emotions = config.EMOTIONS
    
    analyze_dataset(dataset_path, emotions)