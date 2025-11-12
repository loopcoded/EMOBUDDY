"""
testing/test_fusion_system.py - Test and evaluate multi-modal fusion
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from pathlib import Path

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
if "testing" in ROOT_PATH:
    ROOT_PATH = os.path.abspath(os.path.join(ROOT_PATH, os.pardir))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from models.face_emotion_model import FaceEmotionModelTL
from models.voice_emotion_model import VoiceEmotionModel
from models.emotion_fusion_enhanced import (
    EmotionFusionSystem, LearnedFusionModel, MultiModalEmotionSystem
)
from utils.data_preprocessing import load_face_dataset, load_voice_dataset
from config import config


class FusionEvaluator:
    """Evaluate multi-modal fusion performance"""
    
    def __init__(self, emotions, face_model_path, voice_model_path):
        self.emotions = emotions
        self.num_emotions = len(emotions)
        
        # Load models
        print("Loading face model...")
        self.face_model = FaceEmotionModelTL(
            num_classes=self.num_emotions,
            input_shape=(96, 96, 3)
        )
        self.face_model.load_model(face_model_path)
        print("✅ Face model loaded")
        
        print("Loading voice model...")
        self.voice_model = VoiceEmotionModel(
            num_classes=self.num_emotions,
            input_shape=(40, 128, 1)
        )
        self.voice_model.load_model(voice_model_path)
        print("✅ Voice model loaded")
    
    def get_predictions(self, face_data, voice_data):
        """
        Get predictions from both models.
        
        Args:
            face_data: Face images (N, 96, 96, 3)
            voice_data: Voice features (N, 40, 128, 1)
        
        Returns:
            face_preds, voice_preds, face_conf, voice_conf
        """
        print("\nGenerating predictions...")
        
        # Face predictions
        face_preds = []
        face_conf = []
        for i in range(len(face_data)):
            pred = self.face_model.predict(face_data[i])
            face_preds.append(pred)
            face_conf.append(float(np.max(pred)))
            
            if (i + 1) % 100 == 0:
                print(f"  Face: {i+1}/{len(face_data)}")
        
        face_preds = np.array(face_preds)
        face_conf = np.array(face_conf).reshape(-1, 1)
        
        # Voice predictions
        voice_preds = []
        voice_conf = []
        for i in range(len(voice_data)):
            pred = self.voice_model.predict(voice_data[i])
            voice_preds.append(pred)
            voice_conf.append(float(np.max(pred)))
            
            if (i + 1) % 100 == 0:
                print(f"  Voice: {i+1}/{len(voice_data)}")
        
        voice_preds = np.array(voice_preds)
        voice_conf = np.array(voice_conf).reshape(-1, 1)
        
        print("✅ Predictions generated")
        return face_preds, voice_preds, face_conf, voice_conf
    
    def evaluate_strategy(self, face_preds, voice_preds, face_conf, 
                         voice_conf, y_true, strategy='confidence_weighted'):
        """
        Evaluate a specific fusion strategy.
        
        Returns:
            dict with evaluation metrics
        """
        fusion = EmotionFusionSystem(
            self.emotions, 
            fusion_strategy=strategy
        )
        
        # Fuse predictions
        fused_preds = []
        for i in range(len(face_preds)):
            result = fusion.fuse(
                face_preds[i],
                voice_preds[i],
                face_conf[i, 0],
                voice_conf[i, 0]
            )
            
            fused_pred = np.array([
                result['all_predictions'][e] for e in self.emotions
            ])
            fused_preds.append(fused_pred)
        
        fused_preds = np.array(fused_preds)
        
        # Get predicted classes
        y_pred = np.argmax(fused_preds, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true_classes, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_classes, y_pred, average='weighted', zero_division=0
        )
        
        # Per-class metrics
        cm = confusion_matrix(y_true_classes, y_pred)
        
        # Calculate top-2 accuracy
        top2_preds = np.argsort(fused_preds, axis=1)[:, -2:]
        top2_accuracy = np.mean([
            y_true_classes[i] in top2_preds[i] 
            for i in range(len(y_true_classes))
        ])
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'top2_accuracy': top2_accuracy,
            'confusion_matrix': cm,
            'predictions': fused_preds
        }
    
    def compare_all_strategies(self, face_preds, voice_preds, face_conf, 
                               voice_conf, y_true):
        """
        Compare all fusion strategies.
        
        Returns:
            dict with results for each strategy
        """
        strategies = [
            'weighted',
            'confidence_weighted',
            'voting',
            'attention'
        ]
        
        # Also evaluate individual modalities
        results = {}
        
        print("\n" + "=" * 60)
        print("EVALUATING FUSION STRATEGIES")
        print("=" * 60)
        
        # Face only
        print("\n📷 Face only...")
        y_pred_face = np.argmax(face_preds, axis=1)
        y_true_classes = np.argmax(y_true, axis=1)
        results['face_only'] = {
            'accuracy': accuracy_score(y_true_classes, y_pred_face),
            'precision': precision_recall_fscore_support(
                y_true_classes, y_pred_face, average='weighted', zero_division=0
            )[0],
            'recall': precision_recall_fscore_support(
                y_true_classes, y_pred_face, average='weighted', zero_division=0
            )[1],
            'f1': precision_recall_fscore_support(
                y_true_classes, y_pred_face, average='weighted', zero_division=0
            )[2]
        }
        print(f"  Accuracy: {results['face_only']['accuracy']:.4f}")
        
        # Voice only
        print("\n🎤 Voice only...")
        y_pred_voice = np.argmax(voice_preds, axis=1)
        results['voice_only'] = {
            'accuracy': accuracy_score(y_true_classes, y_pred_voice),
            'precision': precision_recall_fscore_support(
                y_true_classes, y_pred_voice, average='weighted', zero_division=0
            )[0],
            'recall': precision_recall_fscore_support(
                y_true_classes, y_pred_voice, average='weighted', zero_division=0
            )[1],
            'f1': precision_recall_fscore_support(
                y_true_classes, y_pred_voice, average='weighted', zero_division=0
            )[2]
        }
        print(f"  Accuracy: {results['voice_only']['accuracy']:.4f}")
        
        # Fusion strategies
        for strategy in strategies:
            print(f"\n🔀 Testing {strategy} fusion...")
            results[strategy] = self.evaluate_strategy(
                face_preds, voice_preds, face_conf, voice_conf,
                y_true, strategy
            )
            print(f"  Accuracy: {results[strategy]['accuracy']:.4f}")
            print(f"  F1 Score: {results[strategy]['f1']:.4f}")
            print(f"  Top-2 Accuracy: {results[strategy]['top2_accuracy']:.4f}")
        
        return results
    
    def plot_comparison(self, results, save_path=None):
        """Plot comparison of fusion strategies"""
        strategies = [k for k in results.keys() if k not in ['face_only', 'voice_only']]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        ax = axes[0]
        all_strategies = ['face_only', 'voice_only'] + strategies
        accuracies = [results[s]['accuracy'] for s in all_strategies]
        
        colors = ['#FF6B6B', '#4ECDC4'] + ['#95E1D3'] * len(strategies)
        bars = ax.bar(range(len(all_strategies)), accuracies, color=colors)
        
        ax.set_xticks(range(len(all_strategies)))
        ax.set_xticklabels([s.replace('_', '\n') for s in all_strategies], 
                           rotation=45, ha='right')
        ax.set_ylabel('Accuracy')
        ax.set_title('Fusion Strategy Comparison', fontweight='bold')
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)
        
        # Multi-metric comparison
        ax = axes[1]
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(strategies))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results[s][metric] for s in strategies]
            ax.bar(x + i*width, values, width, label=metric.capitalize())
        
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([s.replace('_', '\n') for s in strategies], 
                          rotation=45, ha='right')
        ax.set_ylabel('Score')
        ax.set_title('Fusion Strategies - Multiple Metrics', fontweight='bold')
        ax.legend()
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 Comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, results, strategy='confidence_weighted', 
                             save_path=None):
        """Plot confusion matrix for best strategy"""
        cm = results[strategy]['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.emotions,
            yticklabels=self.emotions,
            cbar_kws={'label': 'Count'}
        )
        plt.xlabel('Predicted Emotion', fontweight='bold')
        plt.ylabel('True Emotion', fontweight='bold')
        plt.title(f'Confusion Matrix - {strategy.replace("_", " ").title()}',
                 fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Confusion matrix saved to {save_path}")
        
        plt.show()


def main():
    """Main testing function"""
    print("=" * 60)
    print("MULTI-MODAL EMOTION FUSION EVALUATION")
    print("=" * 60)
    
    emotions = config.EMOTIONS
    
    # Model paths
    face_model_path = 'static/models/face_emotion_mobilenet.keras'
    voice_model_path = 'static/models/voice_emotion_model_best.keras'
    
    # Check if models exist
    if not os.path.exists(face_model_path):
        print(f"❌ Face model not found: {face_model_path}")
        return
    
    if not os.path.exists(voice_model_path):
        print(f"❌ Voice model not found: {voice_model_path}")
        return
    
    # Initialize evaluator
    evaluator = FusionEvaluator(emotions, face_model_path, voice_model_path)
    
    # Load test data
    print("\n" + "=" * 60)
    print("LOADING TEST DATA")
    print("=" * 60)
    
    print("\nLoading face test data...")
    _, _, X_face_test, y_face_test = load_face_dataset(
        'datasets/face_emotions/',
        img_size=(96, 96)
    )
    
    print("\nLoading voice test data...")
    _, _, X_voice_test, y_voice_test = load_voice_dataset(
        'datasets/voice_emotions/',
        sample_rate=22050,
        duration=3,
        n_mfcc=40,
        n_mels=128
    )
    
    # Ensure both have same number of samples
    min_samples = min(len(X_face_test), len(X_voice_test))
    print(f"\n📊 Using {min_samples} paired samples for evaluation")
    
    X_face_test = X_face_test[:min_samples]
    y_face_test = y_face_test[:min_samples]
    X_voice_test = X_voice_test[:min_samples]
    y_voice_test = y_voice_test[:min_samples]
    
    # Ensure voice data has correct shape
    if X_voice_test.ndim == 2:
        X_voice_test = np.expand_dims(X_voice_test, axis=-1)
    
    # Get predictions
    face_preds, voice_preds, face_conf, voice_conf = evaluator.get_predictions(
        X_face_test, X_voice_test
    )
    
    # Evaluate all strategies
    results = evaluator.compare_all_strategies(
        face_preds, voice_preds, face_conf, voice_conf, y_face_test
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    for strategy, metrics in results.items():
        print(f"\n{strategy.upper()}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        if 'precision' in metrics:
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1 Score:  {metrics['f1']:.4f}")
        if 'top2_accuracy' in metrics:
            print(f"  Top-2 Acc: {metrics['top2_accuracy']:.4f}")
    
    # Find best strategy
    fusion_strategies = [k for k in results.keys() 
                        if k not in ['face_only', 'voice_only']]
    best_strategy = max(fusion_strategies, 
                       key=lambda s: results[s]['accuracy'])
    
    print(f"\n🏆 BEST FUSION STRATEGY: {best_strategy}")
    print(f"   Accuracy: {results[best_strategy]['accuracy']:.4f}")
    print(f"   Improvement over face only: "
          f"{(results[best_strategy]['accuracy'] - results['face_only']['accuracy']):.4f}")
    print(f"   Improvement over voice only: "
          f"{(results[best_strategy]['accuracy'] - results['voice_only']['accuracy']):.4f}")
    
    # Plot results
    print("\n" + "=" * 60)
    print("GENERATING VISUALIZATIONS")
    print("=" * 60)
    
    os.makedirs('results', exist_ok=True)
    
    evaluator.plot_comparison(
        results,
        save_path='results/fusion_comparison.png'
    )
    
    evaluator.plot_confusion_matrix(
        results,
        strategy=best_strategy,
        save_path=f'results/confusion_matrix_{best_strategy}.png'
    )
    
    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()