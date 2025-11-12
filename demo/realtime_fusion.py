"""
demo/realtime_fusion_demo.py - Real-time multi-modal emotion recognition demo
"""
import os
import sys
import cv2
import numpy as np
import time
from collections import deque

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
if "demo" in ROOT_PATH:
    ROOT_PATH = os.path.abspath(os.path.join(ROOT_PATH, os.pardir))
if ROOT_PATH not in sys.path:
    sys.path.append(ROOT_PATH)

from models.face_emotion_model import FaceEmotionModelTL
from models.voice_emotion_model import VoiceEmotionModel
from models.emotion_fusion_enhanced import MultiModalEmotionSystem
from utils.data_preprocessing import preprocess_face_image, detect_faces_in_image
from config import config


class RealtimeFusionDemo:
    """Real-time multi-modal emotion recognition demo"""
    
    def __init__(self):
        self.emotions = config.EMOTIONS
        self.running = False
        
        # Load models
        print("Loading models...")
        self.face_model = self._load_face_model()
        self.voice_model = self._load_voice_model()
        
        # Initialize fusion system
        fusion_strategy = 'confidence_weighted'  # Can be changed
        self.fusion_system = MultiModalEmotionSystem(
            self.emotions,
            fusion_strategy=fusion_strategy,
            temporal_smoothing=True
        )
        
        print(f"✅ Using {fusion_strategy} fusion strategy")
        
        # Emotion history for visualization
        self.emotion_history = deque(maxlen=100)
        
        # Colors for emotions
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'happy': (0, 255, 0),      # Green
            'sad': (255, 0, 0),        # Blue
            'surprise': (0, 255, 255), # Yellow
            'fear': (128, 0, 128),     # Purple
            'disgust': (0, 128, 128),  # Teal
            'neutral': (128, 128, 128) # Gray
        }
    
    def _load_face_model(self):
        """Load face emotion model"""
        model_path = 'static/models/face_emotion_mobilenet.keras'
        
        if not os.path.exists(model_path):
            print(f"⚠️ Face model not found: {model_path}")
            return None
        
        try:
            model = FaceEmotionModelTL(
                num_classes=len(self.emotions),
                input_shape=(96, 96, 3)
            )
            model.load_model(model_path)
            print("✅ Face model loaded")
            return model
        except Exception as e:
            print(f"❌ Error loading face model: {e}")
            return None
    
    def _load_voice_model(self):
        """Load voice emotion model"""
        model_path = 'static/models/voice_emotion_model_best.keras'
        
        if not os.path.exists(model_path):
            print(f"⚠️ Voice model not found: {model_path}")
            return None
        
        try:
            model = VoiceEmotionModel(
                num_classes=len(self.emotions),
                input_shape=(40, 128, 1)
            )
            model.load_model(model_path)
            print("✅ Voice model loaded")
            return model
        except Exception as e:
            print(f"❌ Error loading voice model: {e}")
            return None
    
    def process_face(self, frame):
        """Process frame for face emotion recognition"""
        if self.face_model is None:
            return None
        
        try:
            # Detect faces
            faces = detect_faces_in_image(frame)
            
            if len(faces) == 0:
                return None
            
            # Use largest face
            face_bbox = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face_bbox
            
            # Extract face region
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            face_region = frame[y1:y2, x1:x2]
            
            # Preprocess
            preprocessed = preprocess_face_image(face_region, (96, 96))
            
            # Predict
            prediction = self.face_model.predict(preprocessed)
            
            emotion_idx = np.argmax(prediction)
            confidence = float(prediction[emotion_idx])
            emotion = self.emotions[emotion_idx]
            
            return {
                'emotion': emotion,
                'confidence': confidence,
                'all_predictions': {
                    self.emotions[i]: float(prediction[i])
                    for i in range(len(self.emotions))
                },
                'bbox': (x, y, w, h)
            }
            
        except Exception as e:
            print(f"Error processing face: {e}")
            return None
    
    def draw_results(self, frame, face_result, fused_result):
        """Draw results on frame"""
        height, width = frame.shape[:2]
        
        # Draw face detection box
        if face_result and 'bbox' in face_result:
            x, y, w, h = face_result['bbox']
            emotion = face_result['emotion']
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Draw emotion label
            label = f"Face: {emotion} ({face_result['confidence']:.2f})"
            cv2.putText(frame, label, (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw fusion result
        if fused_result:
            emotion = fused_result['emotion']
            confidence = fused_result['confidence']
            source = fused_result['source']
            
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Main emotion display
            text = f"Emotion: {emotion.upper()}"
            cv2.putText(frame, text, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            text = f"Confidence: {confidence:.2%}"
            cv2.putText(frame, text, (20, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            text = f"Source: {source}"
            cv2.putText(frame, text, (20, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Add to history
            self.emotion_history.append(emotion)
            
            # Draw emotion bar chart
            self._draw_emotion_bars(frame, fused_result['all_predictions'])
            
            # Draw emotion timeline
            self._draw_emotion_timeline(frame)
    
    def _draw_emotion_bars(self, frame, predictions):
        """Draw emotion probability bars"""
        height, width = frame.shape[:2]
        
        bar_width = 150
        bar_height = 20
        x_start = width - bar_width - 20
        y_start = 20
        
        # Sort emotions by probability
        sorted_emotions = sorted(
            predictions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        for i, (emotion, prob) in enumerate(sorted_emotions):
            y = y_start + i * (bar_height + 5)
            
            # Draw background
            cv2.rectangle(frame,
                         (x_start, y),
                         (x_start + bar_width, y + bar_height),
                         (50, 50, 50), -1)
            
            # Draw filled bar
            fill_width = int(bar_width * prob)
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            cv2.rectangle(frame,
                         (x_start, y),
                         (x_start + fill_width, y + bar_height),
                         color, -1)
            
            # Draw text
            text = f"{emotion}: {prob:.2%}"
            cv2.putText(frame, text,
                       (x_start + 5, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       (255, 255, 255), 1)
    
    def _draw_emotion_timeline(self, frame):
        """Draw emotion history timeline"""
        if len(self.emotion_history) < 2:
            return
        
        height, width = frame.shape[:2]
        timeline_height = 30
        timeline_y = height - timeline_height - 10
        
        # Draw background
        cv2.rectangle(frame,
                     (10, timeline_y),
                     (width - 10, timeline_y + timeline_height),
                     (30, 30, 30), -1)
        
        # Draw emotion segments
        segment_width = (width - 20) / len(self.emotion_history)
        
        for i, emotion in enumerate(self.emotion_history):
            x1 = int(10 + i * segment_width)
            x2 = int(10 + (i + 1) * segment_width)
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            cv2.rectangle(frame,
                         (x1, timeline_y),
                         (x2, timeline_y + timeline_height),
                         color, -1)
    
    def run(self):
        """Run real-time demo"""
        print("\n" + "=" * 60)
        print("REAL-TIME MULTI-MODAL EMOTION RECOGNITION DEMO")
        print("=" * 60)
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to take screenshot")
        print("  - Press 'r' to reset emotion history")
        print("\nStarting webcam...")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Webcam opened successfully")
        print("\nPress 'q' to quit...")
        
        self.running = True
        frame_count = 0
        fps_start_time = time.time()
        fps = 0
        
        while self.running:
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Error: Failed to capture frame")
                break
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_start_time)
                fps_start_time = time.time()
            
            # Process every 5 frames for performance
            if frame_count % 5 == 0:
                # Process face
                face_result = self.process_face(frame)
                
                # For demo, we don't have voice input
                # In real application, you would process audio here
                voice_result = None
                
                # Fuse results
                fused_result = self.fusion_system.process(
                    face_result, voice_result
                )
                
                # Draw results
                self.draw_results(frame, face_result, fused_result)
            
            # Draw FPS
            cv2.putText(frame, f"FPS: {fps:.1f}",
                       (frame.shape[1] - 120, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Multi-Modal Emotion Recognition', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                self.running = False
            elif key == ord('s'):
                filename = f"screenshot_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('r'):
                self.emotion_history.clear()
                self.fusion_system.reset()
                print("🔄 Emotion history reset")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n✅ Demo ended")


def main():
    demo = RealtimeFusionDemo()
    demo.run()


if __name__ == "__main__":
    main()