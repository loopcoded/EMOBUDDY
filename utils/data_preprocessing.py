"""
utils/data_preprocessing.py - Data Preprocessing Utilities
Handles face and voice data preprocessing for emotion recognition
"""
import os
import cv2
import numpy as np
import librosa
import soundfile as sf
from tensorflow import keras
from pathlib import Path
from config import config # Import config to access MODEL_EMOTIONS


# ============================================
# FACE PREPROCESSING
# ============================================

def preprocess_face_image(image, target_size=(48, 48)):
    """
    Preprocess a face image for emotion recognition.
    
    Args:
        image: Input image (BGR or grayscale)
        target_size: Target size for the image
    
    Returns:
        Preprocessed image
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Resize
    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to 3 channels (for models expecting RGB)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    
    # Normalize to [0, 1]
    normalized = rgb.astype(np.float32) / 255.0
    
    return normalized


def detect_faces_in_image(image, face_cascade_path=None):
    """
    Detect faces in an image using Haar Cascade.
    
    Args:
        image: Input image
        face_cascade_path: Path to Haar Cascade XML file
    
    Returns:
        List of face bounding boxes (x, y, w, h)
    """
    if face_cascade_path is None:
        # Use OpenCV's default Haar Cascade
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    else:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
    
    # Convert to grayscale for detection
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    return faces


def extract_face_from_frame(frame, face_bbox, margin=0.2):
    """
    Extract and preprocess a face from a frame.
    
    Args:
        frame: Input video frame
        face_bbox: Face bounding box (x, y, w, h)
        margin: Additional margin around face (proportion of size)
    
    Returns:
        Extracted and preprocessed face image
    """
    x, y, w, h = face_bbox
    
    # Add margin
    margin_x = int(w * margin)
    margin_y = int(h * margin)
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame.shape[1], x + w + margin_x)
    y2 = min(frame.shape[0], y + h + margin_y)
    
    # Extract face region
    face = frame[y1:y2, x1:x2]
    
    # Preprocess
    preprocessed = preprocess_face_image(face)
    
    return preprocessed


def load_face_dataset(dataset_path, img_size=(48, 48), emotions=None):
    """
    Load face emotion dataset from directory structure.
    
    Uses config.MODEL_EMOTIONS if 'emotions' is None.
    
    Args:
        dataset_path: Path to dataset directory
        img_size: Target image size
        emotions: List of emotion labels (defaults to config.MODEL_EMOTIONS)
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    if emotions is None:
        emotions = config.MODEL_EMOTIONS
    
    def load_images_from_folder(folder_path):
        images = []
        labels = []
        
        for emotion_idx, emotion in enumerate(emotions):
            emotion_path = os.path.join(folder_path, emotion)
            
            if not os.path.exists(emotion_path):
                print(f"Warning: {emotion_path} not found. Ensure folder name matches config.")
                continue
            
            print(f"Loading {emotion} images from {emotion_path}...")
            
            for img_file in os.listdir(emotion_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(emotion_path, img_file)
                    
                    try:
                        # Read image
                        img = cv2.imread(img_path)
                        
                        if img is None:
                            continue
                        
                        # Preprocess
                        preprocessed = preprocess_face_image(img, img_size)
                        
                        images.append(preprocessed)
                        labels.append(emotion_idx)
                        
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
        
        return np.array(images), np.array(labels)
    
    # Load training data
    train_path = os.path.join(dataset_path, 'train')
    X_train, y_train = load_images_from_folder(train_path)
    
    # Load test data
    test_path = os.path.join(dataset_path, 'test')
    X_test, y_test = load_images_from_folder(test_path)
    
    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=len(emotions))
    y_test = keras.utils.to_categorical(y_test, num_classes=len(emotions))
    
    return X_train, y_train, X_test, y_test


# ============================================
# VOICE PREPROCESSING
# ============================================

def extract_voice_features(audio_path, sample_rate=22050, duration=3, 
                           n_mfcc=40, n_mels=128):
    """
    Extract features from audio file for emotion recognition.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate
        duration: Duration to extract (seconds)
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of Mel bands
    
    Returns:
        Feature matrix (n_mfcc, time_steps)
    """
    try:
        # Load audio
        audio, sr = librosa.load(audio_path, sr=sample_rate, duration=duration)
        
        # Pad if too short
        target_length = sample_rate * duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Extract Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Resize mel spectrogram to match MFCC time steps
        if mel_spec_db.shape[1] != mfcc.shape[1]:
            mel_spec_db = cv2.resize(
                mel_spec_db, 
                (mfcc.shape[1], n_mels), 
                interpolation=cv2.INTER_LINEAR
            )
        
        # Combine features (use MFCC for now, can combine later)
        features = mfcc
        
        # Normalize
        features = (features - np.mean(features)) / np.std(features)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None


def extract_features_from_audio_buffer(audio_buffer, sample_rate=22050, 
                                       n_mfcc=40, n_mels=128):
    """
    Extract features from audio buffer (for real-time processing).
    
    Args:
        audio_buffer: Audio samples as numpy array
        sample_rate: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of Mel bands
    
    Returns:
        Feature matrix
    """
    try:
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio_buffer, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Normalize
        features = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from buffer: {e}")
        return None


def load_voice_dataset(dataset_path, sample_rate=22050, duration=3,
                       n_mfcc=40, n_mels=128, emotions=None):
    """
    Load voice emotion dataset from directory structure.
    
    Uses config.MODEL_EMOTIONS if 'emotions' is None.
    
    Args:
        dataset_path: Path to dataset directory
        sample_rate: Target sample rate
        duration: Audio duration
        n_mfcc: Number of MFCC coefficients
        n_mels: Number of Mel bands
        emotions: List of emotion labels (defaults to config.MODEL_EMOTIONS)
    
    Returns:
        X_train, y_train, X_test, y_test
    """
    if emotions is None:
        emotions = config.MODEL_EMOTIONS
    
    def load_audio_from_folder(folder_path):
        features_list = []
        labels = []
        
        for emotion_idx, emotion in enumerate(emotions):
            emotion_path = os.path.join(folder_path, emotion)
            
            if not os.path.exists(emotion_path):
                print(f"Warning: {emotion_path} not found. Ensure folder name matches config.")
                continue
            
            print(f"Loading {emotion} audio from {emotion_path}...")
            
            for audio_file in os.listdir(emotion_path):
                if audio_file.lower().endswith(('.wav', '.mp3', '.flac')):
                    audio_path = os.path.join(emotion_path, audio_file)
                    
                    features = extract_voice_features(
                        audio_path, sample_rate, duration, n_mfcc, n_mels
                    )
                    
                    if features is not None:
                        features_list.append(features)
                        labels.append(emotion_idx)
        
        return np.array(features_list), np.array(labels)
    
    # Load training data
    train_path = os.path.join(dataset_path, 'train')
    X_train, y_train = load_audio_from_folder(train_path)
    
    # Load test data
    test_path = os.path.join(dataset_path, 'test')
    X_test, y_test = load_audio_from_folder(test_path)
    
    # Convert labels to one-hot encoding
    y_train = keras.utils.to_categorical(y_train, num_classes=len(emotions))
    y_test = keras.utils.to_categorical(y_test, num_classes=len(emotions))
    
    return X_train, y_train, X_test, y_test


# ============================================
# EMOTION SMOOTHING
# ============================================

class EmotionSmoother:
    """Smooth emotion predictions over time using moving average."""
    
    def __init__(self, window_size=5):
        """
        Initialize emotion smoother.
        
        Args:
            window_size: Number of predictions to average
        """
        self.window_size = window_size
        self.predictions_history = []
    
    def update(self, prediction):
        """
        Update with new prediction and return smoothed result.
        
        Args:
            prediction: New emotion prediction (array of probabilities)
        
        Returns:
            Smoothed prediction
        """
        self.predictions_history.append(prediction)
        
        # Keep only recent predictions
        if len(self.predictions_history) > self.window_size:
            self.predictions_history.pop(0)
        
        # Return average
        return np.mean(self.predictions_history, axis=0)
    
    def reset(self):
        """Reset the prediction history."""
        self.predictions_history = []


if __name__ == "__main__":
    print("Data preprocessing utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  - preprocess_face_image()")
    print("  - detect_faces_in_image()")
    print("  - load_face_dataset()")
    print("  - extract_voice_features()")
    print("  - load_voice_dataset()")
    print("  - EmotionSmoother class")