"""
utils/data_preprocessing.py - Data Preprocessing Utilities
Handles face and voice data preprocessing for emotion recognition
"""

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import cv2
import numpy as np
import soundfile as sf
import torchaudio
import torch
from tensorflow import keras
from pathlib import Path
from config import config


# ============================================
# FACE PREPROCESSING
# ============================================

def preprocess_face_image(image, target_size=(48, 48)):
    """Preprocess a face image for emotion recognition."""
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    resized = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return normalized


def detect_faces_in_image(image, face_cascade_path=None):
    """Detect faces in an image using Haar Cascade."""
    if face_cascade_path is None:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
    else:
        face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )
    return faces


def extract_face_from_frame(frame, face_bbox, margin=0.2):
    """Extract and preprocess a face from a frame."""
    x, y, w, h = face_bbox
    margin_x = int(w * margin)
    margin_y = int(h * margin)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(frame.shape[1], x + w + margin_x)
    y2 = min(frame.shape[0], y + h + margin_y)

    face = frame[y1:y2, x1:x2]
    return preprocess_face_image(face)


def load_face_dataset(dataset_path, img_size=(48, 48), emotions=None):
    """Load face emotion dataset from directory structure."""
    if emotions is None:
        emotions = config.MODEL_EMOTIONS

    def load_images_from_folder(folder_path):
        images, labels = [], []

        for emotion_idx, emotion in enumerate(emotions):
            emotion_path = os.path.join(folder_path, emotion)
            if not os.path.exists(emotion_path):
                print(f"Warning: {emotion_path} not found.")
                continue

            print(f"Loading {emotion} images from {emotion_path}...")

            for img_file in os.listdir(emotion_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(emotion_path, img_file)

                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue

                        preprocessed = preprocess_face_image(img, img_size)
                        images.append(preprocessed)
                        labels.append(emotion_idx)

                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")

        return np.array(images), np.array(labels)

    X_train, y_train = load_images_from_folder(os.path.join(dataset_path, 'train'))
    X_test, y_test = load_images_from_folder(os.path.join(dataset_path, 'test'))

    y_train = keras.utils.to_categorical(y_train, len(emotions))
    y_test = keras.utils.to_categorical(y_test, len(emotions))

    return X_train, y_train, X_test, y_test


# ============================================
# VOICE PREPROCESSING (TORCHAUDIO ONLY)
# ============================================

def extract_voice_features(
    audio_path, sample_rate=22050, duration=3, n_mfcc=40, n_mels=128
):
    """Extract MFCC features using torchaudio (no librosa, no numba)."""
    try:
        waveform, sr = torchaudio.load(audio_path)

        # Ensure mono
        waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)

        # Ensure fixed duration
        target_len = sample_rate * duration
        if waveform.shape[1] > target_len:
            waveform = waveform[:, :target_len]
        else:
            pad_len = target_len - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad_len))

        # Extract MFCC
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_mels": n_mels}
        )

        mfcc = mfcc_transform(waveform)
        mfcc = mfcc.squeeze(0).numpy()

        # Normalize
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)
        return mfcc

    except Exception as e:
        print(f"[ERROR] Failed audio extraction: {audio_path}")
        print(f"Detail: {e}")
        return None


def extract_features_from_audio_buffer(
    audio_buffer, sample_rate=22050, n_mfcc=40, n_mels=128
):
    """Extract MFCC from raw audio buffer for real-time inference."""
    try:
        waveform = torch.tensor(audio_buffer).float().unsqueeze(0)

        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_mels": n_mels}
        )

        mfcc = mfcc_transform(waveform).squeeze(0).numpy()
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

        return mfcc

    except Exception as e:
        print(f"Error extracting features from buffer: {e}")
        return None


def load_voice_dataset(dataset_path, sample_rate=22050, duration=3,
                       n_mfcc=40, n_mels=128, emotions=None):

    if emotions is None:
        emotions = config.MODEL_EMOTIONS

    def load_audio_from_folder(folder_path):
        features, labels = [], []

        for emotion_idx, emotion in enumerate(emotions):
            emotion_path = os.path.join(folder_path, emotion)

            if not os.path.exists(emotion_path):
                print(f"Warning: {emotion_path} missing.")
                continue

            print(f"Loading {emotion} audio from {emotion_path}...")

            for file in os.listdir(emotion_path):
                if file.lower().endswith((".wav", ".mp3", ".flac")):
                    feat = extract_voice_features(
                        os.path.join(emotion_path, file),
                        sample_rate, duration, n_mfcc, n_mels
                    )

                    if feat is not None:
                        features.append(feat)
                        labels.append(emotion_idx)

        return np.array(features), np.array(labels)

    X_train, y_train = load_audio_from_folder(os.path.join(dataset_path, "train"))
    X_test, y_test = load_audio_from_folder(os.path.join(dataset_path, "test"))

    y_train = keras.utils.to_categorical(y_train, len(emotions))
    y_test = keras.utils.to_categorical(y_test, len(emotions))

    return X_train, y_train, X_test, y_test


# ============================================
# EMOTION SMOOTHING
# ============================================

class EmotionSmoother:
    """Smooth emotion predictions over time using moving average."""
    
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.predictions_history = []
    
    def update(self, prediction):
        self.predictions_history.append(prediction)
        if len(self.predictions_history) > self.window_size:
            self.predictions_history.pop(0)
        return np.mean(self.predictions_history, axis=0)
    
    def reset(self):
        self.predictions_history = []


if __name__ == "__main__":
    print("Data preprocessing utilities loaded successfully!")
