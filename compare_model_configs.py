"""
compare_model_configs.py - Compare Different Model Configurations

This script helps you find the optimal configuration by testing multiple
settings and comparing their performance on your dataset.
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
if "training" in ROOT_PATH:
    ROOT_PATH = os.path.abspath(os.path.join(ROOT_PATH, os.pardir))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from models.voice_emotion_model import VoiceEmotionModel
from utils.data_preprocessing import load_voice_dataset

# ============================================
# CONFIGURATION PRESETS
# ============================================
CONFIGURATIONS = {
    'ultra_conservative': {
        'name': 'Ultra Conservative (Smallest Dataset)',
        'model_type': 'simple',
        'architecture': 'ultra_light',
        'l2': 0.005,
        'dropout': 0.7,
        'learning_rate': 0.0003,
        'label_smoothing': 0.25,
        'batch_size': 16,
        'augmentation_factor': 5
    },
    
    'conservative': {
        'name': 'Conservative (Small Dataset)',
        'model_type': 'simple',
        'architecture': 'simple',
        'l2': 0.003,
        'dropout': 0.6,
        'learning_rate': 0.0005,
        'label_smoothing': 0.2,
        'batch_size': 32,
        'augmentation_factor': 3
    },
    
    'moderate': {
        'name': 'Moderate (Medium Dataset)',
        'model_type': 'standard',
        'architecture': 'cnn',
        'l2': 0.002,
        'dropout': 0.5,
        'learning_rate': 0.001,
        'label_smoothing': 0.15,
        'batch_size': 32,
        'augmentation_factor': 2
    },
    
    'aggressive': {
        'name': 'Aggressive (Large Dataset)',
        'model_type': 'standard',
        'architecture': 'cnn_lstm',
        'l2': 0.001,
        'dropout': 0.4,
        'learning_rate': 0.001,
        'label_smoothing': 0.1,
        'batch_size': 64,
        'augmentation_factor': 1
    }
}

# ============================================
# QUICK TRAINING FUNCTION
# ============================================
def quick_train_test(config, X_train, y_train, X_val, y_val, X_test, y_test, epochs=30):
    """
    Quick training run to test a configuration.
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*80}")
    print(f"Testing Configuration: {config['name']}")
    print(f"{'='*80}")
    
    # Build model
    if config['model_type'] == 'simple':
        model_builder = VoiceEmotionModel(num_classes=7, input_shape=(40, 128, 1))
        
        if config['architecture'] == 'ultra_light':
            model = model_builder.build_ultra_light_model(
                l2=config['l2'],
                dropout=config['dropout']
            )
        else:
            model = model_builder.build_model(
                l2=config['l2'],
                dropout=config['dropout']
            )
    else:
        model_builder = VoiceEmotionModel(num_classes=7, input_shape=(40, 128, 1))
        model = model_builder.build_model(
            architecture=config['architecture'],
            l2=config['l2'],
            dropout=config['dropout']
        )
    
    # Compile
    model_builder.compile_model(
        learning_rate=config['learning_rate'],
        label_smoothing=config['label_smoothing']
    )
    
    # Count parameters
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    print(f"\nModel Parameters: {trainable_params:,}")
    
    # Setup callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            mode='max',
            verbose=0
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=0
        )
    ]
    
    # Train
    print(f"\nTraining for max {epochs} epochs...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=config['batch_size'],
        epochs=epochs,
        callbacks=callbacks,
        verbose=0
    )
    
    # Evaluate
    val_results = model.evaluate(X_val, y_val, verbose=0)
    test_results = model.evaluate(X_test, y_test, verbose=0)
    
    # Calculate overfitting metrics
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    final_gap = train_acc[-1] - val_acc[-1]
    max_gap = max([t - v for t, v in zip(train_acc, val_acc)])
    
    results = {
        'config_name': config['name'],
        'trainable_params': trainable_params,
        'epochs_trained': len(train_acc),
        'final_train_acc': train_acc[-1],
        'final_val_acc': val_acc[-1],
        'best_val_acc': max(val_acc),
        'test_acc': test_results[1],  # Assuming accuracy is second metric
        'final_gap': final_gap,
        'max_gap': max_gap,
        'test_precision': test_results[3] if len(test_results) > 3 else 0,
        'test_recall': test_results[4] if len(test_results) > 4 else 0,
        'history': history.history
    }
    
    # Print summary
    print(f"\nResults:")
    print(f"  Epochs Trained: {results['epochs_trained']}")
    print(f"  Best Val Acc: {results['best_val_acc']:.4f}")
    print(f"  Test Acc: {results['test_acc']:.4f}")
    print(f"  Final Gap: {results['final_gap']:.4f}")
    print(f"  Overfitting: {'🔴 High' if final_gap > 0.15 else '🟡 Moderate' if final_gap > 0.10 else '🟢 Low'}")
    
    return results

# ============================================
# COMPARISON RUNNER
# ============================================
def run_comparison(dataset_path, configs_to_test=None, quick_epochs=30):
    """
    Run comparison across multiple configurations.
    
    Args:
        dataset_path: Path to voice dataset
        configs_to_test: List of config keys to test (default: all)
        quick_epochs: Number of epochs for quick test
    
    Returns:
        DataFrame with comparison results
    """
    print("="*80)
    print("MODEL CONFIGURATION COMPARISON")
    print("="*80)
    
    # Load dataset
    print("\nLoading dataset...")
    X_train, y_train, X_test, y_test = load_voice_dataset(
        dataset_path,
        sample_rate=22050,
        duration=3,
        n_mfcc=40,
        n_mels=128
    )
    
    # Add channel dimension
    if X_train.ndim == 3:
        X_train = X_train[..., np.newaxis]
        X_test = X_test[..., np.newaxis]
    
    # Create validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.15,
        stratify=np.argmax(y_train, axis=1),
        random_state=42
    )
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Determine which configs to test
    if configs_to_test is None:
        configs_to_test = list(CONFIGURATIONS.keys())
    
    # Run tests
    results = []
    for config_key in configs_to_test:
        if config_key not in CONFIGURATIONS:
            print(f"Warning: Unknown config '{config_key}', skipping...")
            continue
        
        config = CONFIGURATIONS[config_key]
        result = quick_train_test(
            config, X_train, y_train, X_val, y_val, X_test, y_test,
            epochs=quick_epochs
        )
        results.append(result)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame([{
        'Configuration': r['config_name'],
        'Parameters': r['trainable_params'],
        'Epochs': r['epochs_trained'],
        'Best Val Acc': f"{r['best_val_acc']:.4f}",
        'Test Acc': f"{r['test_acc']:.4f}",
        'Overfitting Gap': f"{r['final_gap']:.4f}",
        'Precision': f"{r['test_precision']:.4f}",
        'Recall': f"{r['test_recall']:.4f}"
    } for r in results])
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}\n")
    print(comparison_df.to_string(index=False))
    print(f"\n{'='*80}")
    
    # Find best configuration
    best_idx = np.argmax([r['test_acc'] for r in results])
    best_config = results[best_idx]
    
    print(f"\n🏆 BEST CONFIGURATION: {best_config['config_name']}")
    print(f"   Test Accuracy: {best_config['test_acc']:.4f}")
    print(f"   Overfitting Gap: {best_config['final_gap']:.4f}")
    
    # Plot comparison
    plot_comparison(results)
    
    return comparison_df, results

def plot_comparison(results):
    """Create visualization comparing all configurations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    config_names = [r['config_name'] for r in results]
    
    # 1. Test Accuracy Comparison
    test_accs = [r['test_acc'] for r in results]
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(config_names)), test_accs, color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(config_names)))
    ax1.set_xticklabels(config_names, rotation=45, ha='right')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy Comparison', fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Overfitting Gap
    gaps = [r['final_gap'] for r in results]
    ax2 = axes[0, 1]
    colors = ['red' if g > 0.15 else 'orange' if g > 0.10 else 'green' for g in gaps]
    bars = ax2.bar(range(len(config_names)), gaps, color=colors, alpha=0.7)
    ax2.axhline(y=0.15, color='red', linestyle='--', label='High Overfitting', linewidth=2)
    ax2.axhline(y=0.10, color='orange', linestyle='--', label='Moderate Overfitting', linewidth=2)
    ax2.set_xticks(range(len(config_names)))
    ax2.set_xticklabels(config_names, rotation=45, ha='right')
    ax2.set_ylabel('Train-Val Accuracy Gap')
    ax2.set_title('Overfitting Analysis', fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, gap in zip(bars, gaps):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{gap:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Parameter Count
    params = [r['trainable_params'] for r in results]
    ax3 = axes[1, 0]
    bars = ax3.bar(range(len(config_names)), params, color='mediumpurple', alpha=0.7)
    ax3.set_xticks(range(len(config_names)))
    ax3.set_xticklabels(config_names, rotation=45, ha='right')
    ax3.set_ylabel('Trainable Parameters')
    ax3.set_title('Model Complexity', fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    for bar, param in zip(bars, params):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{param/1000:.0f}K', ha='center', va='bottom', fontsize=9)
    
    # 4. Learning Curves (best config)
    best_idx = np.argmax(test_accs)
    best_result = results[best_idx]
    
    ax4 = axes[1, 1]
    ax4.plot(best_result['history']['accuracy'], label='Train', linewidth=2)
    ax4.plot(best_result['history']['val_accuracy'], label='Val', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.set_title(f'Learning Curve: {best_result["config_name"]}', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'model_comparison_{timestamp}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\n✅ Comparison plot saved: {filename}")

# ============================================
# MAIN EXECUTION
# ============================================
if __name__ == "__main__":
    # Configuration
    DATASET_PATH = 'datasets/voice_emotions/'
    
    # Test specific configurations (or None for all)
    # Options: 'ultra_conservative', 'conservative', 'moderate', 'aggressive'
    configs_to_test = ['ultra_conservative', 'conservative', 'moderate']
    
    # Number of epochs for quick testing
    QUICK_EPOCHS = 30
    
    # Run comparison
    print("\n🎤 Starting Model Configuration Comparison...")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Configurations to test: {configs_to_test}")
    print(f"Quick test epochs: {QUICK_EPOCHS}\n")
    
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Error: Dataset path '{DATASET_PATH}' not found!")
        print("   Please update DATASET_PATH in the script.")
    else:
        df, results = run_comparison(
            DATASET_PATH,
            configs_to_test=configs_to_test,
            quick_epochs=QUICK_EPOCHS
        )
        
        print("\n" + "="*80)
        print("✅ COMPARISON COMPLETE!")
        print("="*80)
        print("\n💡 Next Steps:")
        print("   1. Review the comparison plot (model_comparison_*.png)")
        print("   2. Choose the configuration with best test accuracy and low overfitting")
        print("   3. Run full training with the chosen configuration")
        print("   4. If all configurations overfit, try 'ultra_conservative' or collect more data")
        print("="*80)