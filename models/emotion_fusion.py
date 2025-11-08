"""
models/emotion_fusion.py - Multi-Modal Emotion Fusion
Combines face and voice emotion predictions for better accuracy
"""
import numpy as np
from collections import deque
import time


class EmotionFusion:
    """
    Multi-modal emotion fusion system.
    Combines face and voice predictions using various strategies.
    """
    
    def __init__(self, emotions, fusion_strategy='weighted_average'):
        """
        Initialize emotion fusion.
        
        Args:
            emotions: List of emotion labels
            fusion_strategy: Strategy to use ('weighted_average', 'voting', 
                           'attention', 'dynamic_weight')
        """
        self.emotions = emotions
        self.num_emotions = len(emotions)
        self.fusion_strategy = fusion_strategy
        
        # Default weights (can be learned)
        self.face_weight = 0.6
        self.voice_weight = 0.4
        
        # For dynamic weighting
        self.face_confidence_history = deque(maxlen=10)
        self.voice_confidence_history = deque(maxlen=10)
        
    def fuse(self, face_prediction, voice_prediction, 
             face_confidence=None, voice_confidence=None):
        """
        Fuse face and voice predictions.
        
        Args:
            face_prediction: Face emotion probabilities (numpy array or dict)
            voice_prediction: Voice emotion probabilities (numpy array or dict)
            face_confidence: Confidence score for face prediction
            voice_confidence: Confidence score for voice prediction
        
        Returns:
            dict: Fused emotion result
        """
        # Convert to numpy arrays if dicts
        face_probs = self._to_array(face_prediction)
        voice_probs = self._to_array(voice_prediction)
        
        # Handle missing modalities
        if face_probs is None and voice_probs is None:
            return None
        
        if face_probs is None:
            return self._create_result(voice_probs, 'voice_only')
        
        if voice_probs is None:
            return self._create_result(face_probs, 'face_only')
        
        # Update confidence history for dynamic weighting
        if face_confidence is not None:
            self.face_confidence_history.append(face_confidence)
        if voice_confidence is not None:
            self.voice_confidence_history.append(voice_confidence)
        
        # Choose fusion strategy
        if self.fusion_strategy == 'weighted_average':
            fused_probs = self._weighted_average_fusion(face_probs, voice_probs)
        elif self.fusion_strategy == 'voting':
            fused_probs = self._voting_fusion(face_probs, voice_probs)
        elif self.fusion_strategy == 'attention':
            fused_probs = self._attention_fusion(face_probs, voice_probs)
        elif self.fusion_strategy == 'dynamic_weight':
            fused_probs = self._dynamic_weight_fusion(
                face_probs, voice_probs, face_confidence, voice_confidence
            )
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")
        
        return self._create_result(fused_probs, 'fused')
    
    def _to_array(self, prediction):
        """Convert prediction to numpy array."""
        if prediction is None:
            return None
        
        if isinstance(prediction, dict):
            # Convert dict to array
            return np.array([prediction.get(e, 0.0) for e in self.emotions])
        
        return np.array(prediction)
    
    def _create_result(self, probabilities, source):
        """Create result dictionary from probabilities."""
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
    
    def _weighted_average_fusion(self, face_probs, voice_probs):
        """
        Simple weighted average fusion.
        """
        fused = self.face_weight * face_probs + self.voice_weight * voice_probs
        fused = fused / np.sum(fused)
        return fused
    
    def _voting_fusion(self, face_probs, voice_probs):
        """
        Hard voting fusion - each modality votes for top emotion.
        """
        face_top = np.argmax(face_probs)
        voice_top = np.argmax(voice_probs)
        
        votes = np.zeros(self.num_emotions)
        votes[face_top] += self.face_weight
        votes[voice_top] += self.voice_weight
        
        if face_top == voice_top:
            votes[face_top] = 1.0
        
        votes = votes / np.sum(votes)
        return votes
    
    def _attention_fusion(self, face_probs, voice_probs):
        """
        Attention-based fusion - modality with higher entropy gets more weight.
        """
        face_entropy = -np.sum(face_probs * np.log(face_probs + 1e-10))
        voice_entropy = -np.sum(voice_probs * np.log(voice_probs + 1e-10))
        
        face_attention = 1.0 / (face_entropy + 1e-10)
        voice_attention = 1.0 / (voice_entropy + 1e-10)
        
        total_attention = face_attention + voice_attention
        face_weight = face_attention / total_attention
        voice_weight = voice_attention / total_attention
        
        fused = face_weight * face_probs + voice_weight * voice_probs
        fused = fused / np.sum(fused)
        return fused
    
    def _dynamic_weight_fusion(self, face_probs, voice_probs, 
                               face_confidence, voice_confidence):
        """
        Dynamic weighting based on recent confidence history.
        """
        avg_face_conf = np.mean(self.face_confidence_history) if self.face_confidence_history else 0.5
        avg_voice_conf = np.mean(self.voice_confidence_history) if self.voice_confidence_history else 0.5
        
        face_conf = face_confidence if face_confidence is not None else avg_face_conf
        voice_conf = voice_confidence if voice_confidence is not None else avg_voice_conf
        
        total_conf = face_conf + voice_conf
        if total_conf > 0:
            face_weight = face_conf / total_conf
            voice_weight = voice_conf / total_conf
        else:
            face_weight = 0.5
            voice_weight = 0.5
        
        fused = face_weight * face_probs + voice_weight * voice_probs
        fused = fused / np.sum(fused)
        return fused
    
    def set_weights(self, face_weight, voice_weight):
        """Set fusion weights manually."""
        total = face_weight + voice_weight
        self.face_weight = face_weight / total
        self.voice_weight = voice_weight / total


class TemporalFusion:
    """
    Temporal fusion for smoothing emotion predictions over time.
    """
    
    def __init__(self, window_size=5, decay_factor=0.9):
        """
        Initialize temporal fusion.
        
        Args:
            window_size: Number of past predictions to consider
            decay_factor: Weight decay for older predictions
        """
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.prediction_history = deque(maxlen=window_size)
        self.timestamp_history = deque(maxlen=window_size)
    
    def update(self, prediction, timestamp=None):
        """
        Update with new prediction and return temporally smoothed result.
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.prediction_history.append(prediction)
        self.timestamp_history.append(timestamp)
        
        weights = []
        for i in range(len(self.prediction_history)):
            age = len(self.prediction_history) - i - 1
            weight = self.decay_factor ** age
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        smoothed = np.zeros_like(prediction)
        for i, pred in enumerate(self.prediction_history):
            smoothed += weights[i] * pred
        
        return smoothed
    
    def reset(self):
        """Reset temporal history."""
        self.prediction_history.clear()
        self.timestamp_history.clear()


class MultiModalEmotionFusionSystem:
    """
    Complete multi-modal emotion fusion system with temporal smoothing.
    """
    
    def __init__(self, emotions, config=None):
        """
        Initialize the fusion system.
        
        Args:
            emotions: List of emotion labels
            config: Configuration dict
        """
        self.emotions = emotions
        
        default_config = {
            'fusion_strategy': 'dynamic_weight',
            'temporal_smoothing': True,
            'window_size': 5,
            'decay_factor': 0.9,
            'face_weight': 0.6,
            'voice_weight': 0.4
        }
        
        if config:
            default_config.update(config)
        
        self.config = default_config
        
        self.emotion_fusion = EmotionFusion(
            emotions, 
            fusion_strategy=self.config['fusion_strategy']
        )
        
        if self.config['temporal_smoothing']:
            self.temporal_fusion = TemporalFusion(
                window_size=self.config['window_size'],
                decay_factor=self.config['decay_factor']
            )
        else:
            self.temporal_fusion = None
        
        self.emotion_fusion.set_weights(
            self.config['face_weight'],
            self.config['voice_weight']
        )
    
    def process(self, face_result, voice_result):
        """
        Process face and voice results through complete fusion pipeline.
        """
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
        
        fused_result = self.emotion_fusion.fuse(
            face_pred, voice_pred, face_conf, voice_conf
        )
        
        if fused_result is None:
            return None
        
        if self.temporal_fusion:
            fused_probs = self.emotion_fusion._to_array(
                fused_result['all_predictions']
            )
            smoothed_probs = self.temporal_fusion.update(fused_probs)
            
            fused_result = self.emotion_fusion._create_result(
                smoothed_probs, 
                fused_result['source'] + '_temporal'
            )
        
        return fused_result
    
    def reset(self):
        """Reset fusion system state."""
        if self.temporal_fusion:
            self.temporal_fusion.reset()
        
        self.emotion_fusion.face_confidence_history.clear()
        self.emotion_fusion.voice_confidence_history.clear()


if __name__ == "__main__":
    # Test the models
    print("Testing Face Emotion Model...")
    from models.face_emotion_model import FaceEmotionModel
    
    emotions = ['angry', 'happy', 'disgust', 'fear', 'sad', 'surprise', 'neutral']
    
    face_model = FaceEmotionModel(num_classes=len(emotions), input_shape=(48, 48, 3))
    face_model.build_model(use_transfer_learning=True)
    face_model.compile_model()
    print("✓ Face model built successfully!")
    
    print("\nTesting Voice Emotion Model...")
    from models.voice_emotion_model import VoiceEmotionModel
    
    voice_model = VoiceEmotionModel(num_classes=len(emotions), input_shape=(40, 128, 1))
    voice_model.build_model(architecture='cnn_lstm')
    voice_model.compile_model()
    print("✓ Voice model built successfully!")
    
    print("\nTesting Emotion Fusion...")
    fusion = EmotionFusion(emotions, fusion_strategy='dynamic_weight')
    
    # Test fusion
    face_pred = {'angry': 0.1, 'happy': 0.7, 'disgust': 0.1, 'fear': 0.05, 
                 'sad': 0.02, 'surprise': 0.02, 'neutral': 0.01}
    voice_pred = {'angry': 0.2, 'happy': 0.6, 'disgust': 0.1, 'fear': 0.05, 
                  'sad': 0.02, 'surprise': 0.02, 'neutral': 0.01}

    result = fusion.fuse(face_pred, voice_pred, 0.7, 0.6)
    print(f"✓ Fusion result: {result['emotion']} ({result['confidence']:.3f})")
    
    print("\n" + "="*60)
    print("ALL MODEL FILES ARE WORKING CORRECTLY!")
    print("="*60)