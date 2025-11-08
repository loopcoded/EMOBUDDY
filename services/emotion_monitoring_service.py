"""
services/emotion_monitoring_service.py - Real-time Emotion Monitoring Service
Monitors child's emotional state and triggers interventions
"""
import numpy as np
import cv2
import time
from collections import deque
from datetime import datetime
import threading
import logging

from models.face_emotion_model import FaceEmotionModel
from models.voice_emotion_model import VoiceEmotionModel
from utils.data_preprocessing import (
    preprocess_face_image, detect_faces_in_image,
    extract_features_from_audio_buffer, EmotionSmoother
)


class EmotionMonitoringService:
    """
    Real-time emotion monitoring service for autistic children.
    Combines face and voice emotion recognition.
    """
    
    def __init__(self, config):
        """
        Initialize monitoring service.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load models
        self.face_model = None
        self.voice_model = None
        self._load_models()
        
        # Emotion smoothers
        self.face_smoother = EmotionSmoother(config.EMOTION_SMOOTHING_WINDOW)
        self.voice_smoother = EmotionSmoother(config.EMOTION_SMOOTHING_WINDOW)
        
        # Monitoring state
        self.is_monitoring = False
        self.current_emotion = 'calm'
        self.current_confidence = 0.0
        self.intervention_level = 'green'
        
        # History
        self.emotion_history = deque(maxlen=config.MAX_EMOTION_HISTORY)
        
        # Timing
        self.last_face_detection = 0
        self.last_voice_detection = 0
        
        # Thread locks
        self.lock = threading.Lock()
        
    def _load_models(self):
        """Load pre-trained emotion recognition models."""
        try:
            self.logger.info("Loading face emotion model...")
            self.face_model = FaceEmotionModel(
                num_classes=len(self.config.EMOTIONS),
                input_shape=(*self.config.FACE_IMAGE_SIZE, 3)
            )
            self.face_model.load_model(self.config.FACE_MODEL_PATH)
            self.logger.info("Face model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load face model: {e}")
            self.face_model = None
        
        try:
            self.logger.info("Loading voice emotion model...")
            self.voice_model = VoiceEmotionModel(
                num_classes=len(self.config.EMOTIONS),
                input_shape=(self.config.VOICE_N_MFCC, 128, 1)
            )
            self.voice_model.load_model(self.config.VOICE_MODEL_PATH)
            self.logger.info("Voice model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load voice model: {e}")
            self.voice_model = None
    
    def start_monitoring(self):
        """Start emotion monitoring."""
        with self.lock:
            self.is_monitoring = True
            self.face_smoother.reset()
            self.voice_smoother.reset()
            self.logger.info("Emotion monitoring started")
    
    def stop_monitoring(self):
        """Stop emotion monitoring."""
        with self.lock:
            self.is_monitoring = False
            self.logger.info("Emotion monitoring stopped")
    
    def process_face_frame(self, frame):
        """
        Process a video frame for face emotion recognition.
        
        Args:
            frame: Video frame (numpy array)
        
        Returns:
            dict with emotion prediction results
        """
        current_time = time.time()
        
        # Check if enough time has passed since last detection
        if current_time - self.last_face_detection < self.config.FACE_DETECTION_INTERVAL:
            return None
        
        self.last_face_detection = current_time
        
        if self.face_model is None:
            self.logger.warning("Face model not loaded")
            return None
        
        try:
            # Detect faces
            faces = detect_faces_in_image(frame)
            
            if len(faces) == 0:
                self.logger.debug("No faces detected in frame")
                return {
                    'emotion': self.current_emotion,
                    'confidence': 0.0,
                    'face_detected': False
                }
            
            # Use the largest face
            face_bbox = max(faces, key=lambda f: f[2] * f[3])
            
            # Extract and preprocess face
            x, y, w, h = face_bbox
            margin = 20
            x1, y1 = max(0, x - margin), max(0, y - margin)
            x2, y2 = min(frame.shape[1], x + w + margin), min(frame.shape[0], y + h + margin)
            face_region = frame[y1:y2, x1:x2]
            
            preprocessed_face = preprocess_face_image(
                face_region, 
                self.config.FACE_IMAGE_SIZE
            )
            
            # Predict emotion
            prediction = self.face_model.predict(preprocessed_face)
            
            # Smooth prediction
            smoothed_prediction = self.face_smoother.update(prediction)
            
            # Get emotion and confidence
            emotion_idx = np.argmax(smoothed_prediction)
            confidence = float(smoothed_prediction[emotion_idx])
            emotion = self.config.EMOTIONS[emotion_idx]
            
            result = {
                'emotion': emotion,
                'confidence': confidence,
                'face_detected': True,
                'all_predictions': {
                    self.config.EMOTIONS[i]: float(smoothed_prediction[i])
                    for i in range(len(self.config.EMOTIONS))
                },
                'face_bbox': face_bbox.tolist()
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing face frame: {e}")
            return None
    
    def process_voice_sample(self, audio_buffer, sample_rate=22050):
        """
        Process audio sample for voice emotion recognition.
        
        Args:
            audio_buffer: Audio samples (numpy array)
            sample_rate: Sample rate
        
        Returns:
            dict with emotion prediction results
        """
        current_time = time.time()
        
        # Check if enough time has passed
        if current_time - self.last_voice_detection < self.config.VOICE_DETECTION_INTERVAL:
            return None
        
        self.last_voice_detection = current_time
        
        if self.voice_model is None:
            self.logger.warning("Voice model not loaded")
            return None
        
        try:
            # Extract features
            features = extract_features_from_audio_buffer(
                audio_buffer,
                sample_rate,
                self.config.VOICE_N_MFCC,
                self.config.VOICE_N_MEL
            )
            
            if features is None:
                return None
            
            # Ensure correct shape
            features = np.expand_dims(features, axis=-1)  # Add channel dimension
            
            # Predict emotion
            prediction = self.voice_model.predict(features)
            
            # Smooth prediction
            smoothed_prediction = self.voice_smoother.update(prediction)
            
            # Get emotion and confidence
            emotion_idx = np.argmax(smoothed_prediction)
            confidence = float(smoothed_prediction[emotion_idx])
            emotion = self.config.EMOTIONS[emotion_idx]
            
            result = {
                'emotion': emotion,
                'confidence': confidence,
                'all_predictions': {
                    self.config.EMOTIONS[i]: float(smoothed_prediction[i])
                    for i in range(len(self.config.EMOTIONS))
                }
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing voice sample: {e}")
            return None
    
    def fuse_emotions(self, face_result, voice_result, face_weight=0.6, voice_weight=0.4):
        """
        Fuse face and voice emotion predictions.
        
        Args:
            face_result: Face emotion result
            voice_result: Voice emotion result
            face_weight: Weight for face prediction
            voice_weight: Weight for voice prediction
        
        Returns:
            Fused emotion result
        """
        if face_result is None and voice_result is None:
            return None
        
        if face_result is None:
            return voice_result
        
        if voice_result is None:
            return face_result
        
        # Weighted fusion
        face_pred = np.array([
            face_result['all_predictions'][e] 
            for e in self.config.EMOTIONS
        ])
        voice_pred = np.array([
            voice_result['all_predictions'][e] 
            for e in self.config.EMOTIONS
        ])
        
        fused_pred = face_weight * face_pred + voice_weight * voice_pred
        
        emotion_idx = np.argmax(fused_pred)
        confidence = float(fused_pred[emotion_idx])
        emotion = self.config.EMOTIONS[emotion_idx]
        
        return {
            'emotion': emotion,
            'confidence': confidence,
            'source': 'fused',
            'all_predictions': {
                self.config.EMOTIONS[i]: float(fused_pred[i])
                for i in range(len(self.config.EMOTIONS))
            }
        }
    
    def determine_intervention_level(self, emotion, confidence):
        """
        Determine intervention level based on emotion and confidence.
        
        Args:
            emotion: Detected emotion
            confidence: Prediction confidence
        
        Returns:
            Intervention level ('green', 'yellow', 'orange', 'red')
        """
        if confidence < self.config.CONFIDENCE_THRESHOLD:
            return 'green'  # Low confidence, continue normally
        
        for level, info in self.config.INTERVENTION_LEVELS.items():
            if emotion in info['emotions'] and confidence >= info['threshold']:
                return level
        
        return 'green'
    
    def update_emotion_state(self, emotion_result):
        """
        Update current emotion state and trigger interventions if needed.
        
        Args:
            emotion_result: Emotion detection result
        
        Returns:
            dict with updated state and intervention info
        """
        if emotion_result is None:
            return None
        
        with self.lock:
            emotion = emotion_result['emotion']
            confidence = emotion_result['confidence']
            
            # Determine intervention level
            level = self.determine_intervention_level(emotion, confidence)
            
            # Check if intervention level changed
            level_changed = level != self.intervention_level
            
            # Update state
            self.current_emotion = emotion
            self.current_confidence = confidence
            self.intervention_level = level
            
            # Add to history
            self.emotion_history.append({
                'timestamp': datetime.now().isoformat(),
                'emotion': emotion,
                'confidence': confidence,
                'level': level
            })
            
            # Get intervention action
            intervention = self.config.INTERVENTION_LEVELS[level]
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'level': level,
                'level_changed': level_changed,
                'action': intervention['action'],
                'duration': intervention.get('duration', 0),
                'all_predictions': emotion_result.get('all_predictions', {})
            }
    
    def get_emotion_statistics(self):
        """
        Get statistics from emotion history.
        
        Returns:
            dict with emotion statistics
        """
        if not self.emotion_history:
            return None
        
        emotions = [entry['emotion'] for entry in self.emotion_history]
        levels = [entry['level'] for entry in self.emotion_history]
        
        # Count emotions
        emotion_counts = {e: emotions.count(e) for e in self.config.EMOTIONS}
        
        # Count levels
        level_counts = {
            'green': levels.count('green'),
            'yellow': levels.count('yellow'),
            'orange': levels.count('orange'),
            'red': levels.count('red')
        }
        
        return {
            'total_samples': len(self.emotion_history),
            'current_emotion': self.current_emotion,
            'current_confidence': self.current_confidence,
            'current_level': self.intervention_level,
            'emotion_distribution': emotion_counts,
            'level_distribution': level_counts,
            'recent_history': list(self.emotion_history)[-10:]  # Last 10 entries
        }


if __name__ == "__main__":
    print("Emotion Monitoring Service ready!")
    print("\nThis service provides:")
    print("  - Real-time face emotion recognition")
    print("  - Real-time voice emotion recognition")
    print("  - Multi-modal emotion fusion")
    print("  - Automatic intervention level determination")
    print("  - Emotion history tracking")