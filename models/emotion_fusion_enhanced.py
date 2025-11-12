"""
models/emotion_fusion_enhanced.py - Enhanced Multi-Modal Emotion Fusion
Combines face and voice emotion predictions with learned fusion weights
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import time
import pickle


class LearnedFusionModel:
    """
    Learned fusion model that trains optimal weights for combining
    face and voice predictions based on actual performance data.
    """
    
    def __init__(self, num_emotions=7):
        self.num_emotions = num_emotions
        self.model = None
        
    def build_model(self):
        """Build a simple MLP for learning fusion weights"""
        # Input: [face_probs (7), voice_probs (7), face_confidence (1), voice_confidence (1)]
        # Output: fused emotion probabilities (7)
        
        face_input = keras.Input(shape=(self.num_emotions,), name='face_probs')
        voice_input = keras.Input(shape=(self.num_emotions,), name='voice_probs')
        face_conf = keras.Input(shape=(1,), name='face_confidence')
        voice_conf = keras.Input(shape=(1,), name='voice_confidence')
        
        # Concatenate all inputs
        concat = keras.layers.Concatenate()([
            face_input, voice_input, face_conf, voice_conf
        ])
        
        # Learn fusion weights
        x = keras.layers.Dense(64, activation='relu')(concat)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        
        # Output fused probabilities
        output = keras.layers.Dense(self.num_emotions, activation='softmax')(x)
        
        self.model = keras.Model(
            inputs=[face_input, voice_input, face_conf, voice_conf],
            outputs=output,
            name='learned_fusion'
        )
        
        return self.model
    
    def compile_model(self, lr=1e-3):
        self.model.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
    
    def train(self, face_probs, voice_probs, face_conf, voice_conf, y_true, 
              validation_data=None, epochs=50, batch_size=32):
        """
        Train fusion model on paired predictions.
        
        Args:
            face_probs: Face model predictions (N, num_emotions)
            voice_probs: Voice model predictions (N, num_emotions)
            face_conf: Face confidence scores (N, 1)
            voice_conf: Voice confidence scores (N, 1)
            y_true: Ground truth labels (N, num_emotions)
            validation_data: Tuple of validation data
            epochs: Training epochs
            batch_size: Batch size
        """
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        history = self.model.fit(
            [face_probs, voice_probs, face_conf, voice_conf],
            y_true,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, face_probs, voice_probs, face_conf=None, voice_conf=None):
        """Predict fused emotion probabilities"""
        if face_conf is None:
            face_conf = np.max(face_probs, axis=-1, keepdims=True)
        if voice_conf is None:
            voice_conf = np.max(voice_probs, axis=-1, keepdims=True)
        
        if face_probs.ndim == 1:
            face_probs = face_probs.reshape(1, -1)
            voice_probs = voice_probs.reshape(1, -1)
            face_conf = np.array([[face_conf]]) if isinstance(face_conf, (int, float)) else face_conf.reshape(1, 1)
            voice_conf = np.array([[voice_conf]]) if isinstance(voice_conf, (int, float)) else voice_conf.reshape(1, 1)
        
        return self.model.predict(
            [face_probs, voice_probs, face_conf, voice_conf],
            verbose=0
        )
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = keras.models.load_model(path)


class EmotionFusionSystem:
    """
    Complete multi-modal emotion fusion system with multiple strategies.
    """
    
    def __init__(self, emotions, fusion_strategy='learned', learned_model_path=None):
        """
        Initialize fusion system.
        
        Args:
            emotions: List of emotion labels
            fusion_strategy: 'weighted', 'confidence_weighted', 'learned', 'voting', 'attention'
            learned_model_path: Path to trained fusion model (if using 'learned')
        """
        self.emotions = emotions
        self.num_emotions = len(emotions)
        self.fusion_strategy = fusion_strategy
        
        # Default weights
        self.face_weight = 0.6
        self.voice_weight = 0.4
        
        # Learned fusion model
        self.learned_model = None
        if fusion_strategy == 'learned' and learned_model_path:
            self.learned_model = LearnedFusionModel(self.num_emotions)
            self.learned_model.load(learned_model_path)
        
        # Confidence history for dynamic weighting
        self.face_confidence_history = deque(maxlen=20)
        self.voice_confidence_history = deque(maxlen=20)
        
    def fuse(self, face_prediction, voice_prediction, 
             face_confidence=None, voice_confidence=None):
        """
        Fuse face and voice predictions.
        
        Args:
            face_prediction: Face emotion probabilities (array or dict)
            voice_prediction: Voice emotion probabilities (array or dict)
            face_confidence: Face prediction confidence
            voice_confidence: Voice prediction confidence
        
        Returns:
            dict: Fused emotion result
        """
        # Convert to arrays
        face_probs = self._to_array(face_prediction)
        voice_probs = self._to_array(voice_prediction)
        
        # Handle missing modalities
        if face_probs is None and voice_probs is None:
            return None
        
        if face_probs is None:
            return self._create_result(voice_probs, 'voice_only')
        
        if voice_probs is None:
            return self._create_result(face_probs, 'face_only')
        
        # Extract confidences if not provided
        if face_confidence is None:
            face_confidence = float(np.max(face_probs))
        if voice_confidence is None:
            voice_confidence = float(np.max(voice_probs))
        
        # Update history
        self.face_confidence_history.append(face_confidence)
        self.voice_confidence_history.append(voice_confidence)
        
        # Choose fusion strategy
        if self.fusion_strategy == 'weighted':
            fused_probs = self._weighted_fusion(face_probs, voice_probs)
        elif self.fusion_strategy == 'confidence_weighted':
            fused_probs = self._confidence_weighted_fusion(
                face_probs, voice_probs, face_confidence, voice_confidence
            )
        elif self.fusion_strategy == 'learned':
            fused_probs = self._learned_fusion(
                face_probs, voice_probs, face_confidence, voice_confidence
            )
        elif self.fusion_strategy == 'voting':
            fused_probs = self._voting_fusion(face_probs, voice_probs)
        elif self.fusion_strategy == 'attention':
            fused_probs = self._attention_fusion(face_probs, voice_probs)
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        return self._create_result(fused_probs, f'fused_{self.fusion_strategy}')
    
    def _to_array(self, prediction):
        """Convert prediction to numpy array"""
        if prediction is None:
            return None
        
        if isinstance(prediction, dict):
            return np.array([prediction.get(e, 0.0) for e in self.emotions])
        
        return np.array(prediction)
    
    def _create_result(self, probabilities, source):
        """Create result dictionary"""
        emotion_idx = np.argmax(probabilities)
        emotion = self.emotions[emotion_idx]
        confidence = float(probabilities[emotion_idx])
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'source': source,
            'all_predictions': {
                self.emotions[i]: float(probabilities[i])
                for i in range(self.num_emotions)
            }
        }
    
    def _weighted_fusion(self, face_probs, voice_probs):
        """Simple weighted average fusion"""
        fused = self.face_weight * face_probs + self.voice_weight * voice_probs
        return fused / np.sum(fused)
    
    def _confidence_weighted_fusion(self, face_probs, voice_probs, 
                                    face_conf, voice_conf):
        """Confidence-based weighted fusion"""
        # Exponential weighting based on confidence
        face_weight = np.exp(face_conf * 2)
        voice_weight = np.exp(voice_conf * 2)
        
        total = face_weight + voice_weight
        face_weight /= total
        voice_weight /= total
        
        fused = face_weight * face_probs + voice_weight * voice_probs
        return fused / np.sum(fused)
    
    def _learned_fusion(self, face_probs, voice_probs, face_conf, voice_conf):
        """Learned fusion using trained neural network"""
        if self.learned_model is None:
            # Fallback to confidence weighted
            return self._confidence_weighted_fusion(
                face_probs, voice_probs, face_conf, voice_conf
            )
        
        fused = self.learned_model.predict(
            face_probs, voice_probs, face_conf, voice_conf
        )[0]
        
        return fused
    
    def _voting_fusion(self, face_probs, voice_probs):
        """Hard voting fusion"""
        face_top = np.argmax(face_probs)
        voice_top = np.argmax(voice_probs)
        
        votes = np.zeros(self.num_emotions)
        votes[face_top] += 1.0
        votes[voice_top] += 1.0
        
        # If both agree, give strong signal
        if face_top == voice_top:
            votes[face_top] = 3.0
        
        return votes / np.sum(votes)
    
    def _attention_fusion(self, face_probs, voice_probs):
        """Attention-based fusion (lower entropy = higher attention)"""
        face_entropy = -np.sum(face_probs * np.log(face_probs + 1e-10))
        voice_entropy = -np.sum(voice_probs * np.log(voice_probs + 1e-10))
        
        # Lower entropy = more confident = higher weight
        face_attention = 1.0 / (face_entropy + 0.1)
        voice_attention = 1.0 / (voice_entropy + 0.1)
        
        total = face_attention + voice_attention
        face_weight = face_attention / total
        voice_weight = voice_attention / total
        
        fused = face_weight * face_probs + voice_weight * voice_probs
        return fused / np.sum(fused)
    
    def set_weights(self, face_weight, voice_weight):
        """Set fusion weights manually"""
        total = face_weight + voice_weight
        self.face_weight = face_weight / total
        self.voice_weight = voice_weight / total


class TemporalSmoother:
    """Temporal smoothing for emotion predictions"""
    
    def __init__(self, window_size=5, decay_factor=0.9):
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.history = deque(maxlen=window_size)
    
    def update(self, prediction):
        """Update with new prediction and return smoothed result"""
        self.history.append(prediction)
        
        if len(self.history) == 0:
            return prediction
        
        # Apply exponential decay
        weights = np.array([
            self.decay_factor ** (len(self.history) - i - 1)
            for i in range(len(self.history))
        ])
        weights /= np.sum(weights)
        
        smoothed = np.zeros_like(prediction)
        for i, pred in enumerate(self.history):
            smoothed += weights[i] * pred
        
        return smoothed
    
    def reset(self):
        self.history.clear()


class MultiModalEmotionSystem:
    """
    Complete multi-modal emotion recognition system with fusion and smoothing.
    """
    
    def __init__(self, emotions, fusion_strategy='confidence_weighted',
                 temporal_smoothing=True, learned_model_path=None):
        """
        Initialize the complete system.
        
        Args:
            emotions: List of emotion labels
            fusion_strategy: Fusion method to use
            temporal_smoothing: Whether to apply temporal smoothing
            learned_model_path: Path to learned fusion model
        """
        self.emotions = emotions
        
        self.fusion_system = EmotionFusionSystem(
            emotions, 
            fusion_strategy=fusion_strategy,
            learned_model_path=learned_model_path
        )
        
        self.temporal_smoothing = temporal_smoothing
        if temporal_smoothing:
            self.smoother = TemporalSmoother(window_size=5, decay_factor=0.9)
        else:
            self.smoother = None
    
    def process(self, face_result, voice_result):
        """
        Process face and voice results through complete pipeline.
        
        Args:
            face_result: Dict with face emotion results
            voice_result: Dict with voice emotion results
        
        Returns:
            Final fused and smoothed emotion result
        """
        # Extract predictions and confidences
        face_pred = None
        face_conf = None
        if face_result:
            face_pred = face_result.get('all_predictions')
            face_conf = face_result.get('confidence')
        
        voice_pred = None
        voice_conf = None
        if voice_result:
            voice_pred = voice_result.get('all_predictions')
            voice_conf = voice_result.get('confidence')
        
        # Fuse predictions
        fused_result = self.fusion_system.fuse(
            face_pred, voice_pred, face_conf, voice_conf
        )
        
        if fused_result is None:
            return None
        
        # Apply temporal smoothing
        if self.smoother:
            fused_probs = self.fusion_system._to_array(
                fused_result['all_predictions']
            )
            smoothed_probs = self.smoother.update(fused_probs)
            
            fused_result = self.fusion_system._create_result(
                smoothed_probs,
                fused_result['source'] + '_smoothed'
            )
        
        return fused_result
    
    def reset(self):
        """Reset system state"""
        if self.smoother:
            self.smoother.reset()
        
        self.fusion_system.face_confidence_history.clear()
        self.fusion_system.voice_confidence_history.clear()


if __name__ == "__main__":
    print("Enhanced Multi-Modal Fusion System")
    print("=" * 60)
    
    emotions = ['angry', 'happy', 'disgust', 'fear', 'sad', 'surprise', 'neutral']
    
    # Test different fusion strategies
    strategies = ['weighted', 'confidence_weighted', 'voting', 'attention']
    
    for strategy in strategies:
        print(f"\nTesting {strategy} fusion...")
        fusion = EmotionFusionSystem(emotions, fusion_strategy=strategy)
        
        face_pred = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.03, 0.02])
        voice_pred = np.array([0.15, 0.6, 0.1, 0.05, 0.05, 0.03, 0.02])
        
        result = fusion.fuse(face_pred, voice_pred, 0.7, 0.6)
        print(f"  Result: {result['emotion']} (conf: {result['confidence']:.3f})")
    
    print("\n✅ All fusion strategies tested successfully!")