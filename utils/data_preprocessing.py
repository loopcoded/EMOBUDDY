"""
utils/data_preprocessing.py
FINAL CLEAN VERSION
Fully fixed for voice + face preprocessing.
Compatible with train_voice_model.py and train_clean_face_model.py
"""

import os
os.environ["NUMBA_DISABLE_JIT"] = "1"

import cv2
import numpy as np
import random
import librosa  # REQUIRED for audio_to_melspec
import torch
import torchaudio

from tensorflow import keras
from config import config


# ===========================================================
# FACE PREPROCESSING
# ===========================================================

def preprocess_face_image(image, target_size=(96, 96)):
    if image is None:
        raise ValueError("Input image is None")

    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    resized = cv2.resize(gray, target_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    return rgb.astype(np.float32)


def detect_faces_in_image(image, face_cascade_path=None):
    if face_cascade_path is None:
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
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
    x, y, w, h = face_bbox

    mx, my = int(w * margin), int(h * margin)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(frame.shape[1], x + w + mx)
    y2 = min(frame.shape[0], y + h + my)

    face = frame[y1:y2, x1:x2]
    return preprocess_face_image(face)


def load_clean_face_dataset(dataset_path, img_size=(96, 96), emotions=None):
    if emotions is None:
        emotions = config.EMOTIONS

    def load_split(split):
        X, y = [], []
        split_path = os.path.join(dataset_path, split)

        for idx, emotion in enumerate(emotions):
            folder = os.path.join(split_path, emotion)
            if not os.path.exists(folder):
                print(f"[WARN] Missing: {folder}")
                continue

            for f in os.listdir(folder):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    img = cv2.imread(os.path.join(folder, f))
                    if img is None:
                        continue

                    X.append(preprocess_face_image(img, img_size))
                    y.append(idx)

        return np.array(X), keras.utils.to_categorical(y, len(emotions))

    print("🔄 Loading CLEAN dataset...")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    print(f"TRAIN: {len(X_train)} | VAL: {len(X_val)} | TEST: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ===========================================================
# VOICE PREPROCESSING (librosa mel-spectrogram)
# ===========================================================

SAMPLE_RATE = 16000
AUDIO_DURATION = 3
SAMPLES = SAMPLE_RATE * AUDIO_DURATION


# ---------------------------
# SAFE AUDIO AUGMENTATION
# ---------------------------
def augment_audio(y, sr):

    # Pitch Shift
    if random.random() < 0.3:
        try:
            y = librosa.effects.pitch_shift(
                y=y,
                sr=sr,
                n_steps=random.uniform(-2, 2)
            )
        except:
            pass

    # Time Stretch (safe implementation)
    if random.random() < 0.3:
        rate = random.uniform(0.8, 1.2)
        try:
            y = librosa.effects.time_stretch(y, rate)
        except:
            pass

    # Add noise
    if random.random() < 0.3:
        noise = np.random.randn(len(y)) * 0.005
        y = y + noise

    return y


# ---------------------------
# MEL-SPECTROGRAM
# ---------------------------
def audio_to_melspec(path, augment=False):
    y, sr = librosa.load(path, sr=SAMPLE_RATE)

    # Pad or trim to fixed length
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]

    # Gentle augmentation (fixing your overly aggressive version)
    if augment and random.random() < 0.5:
        y = augment_audio(y, sr)

    # Compute mel (safe parameters)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        hop_length=256,
        power=2.0
    )

    mel = librosa.power_to_db(mel, ref=np.max)

    # Normalize between 0–1 (CRITICAL FIX)
    mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

    # Resize
    mel = cv2.resize(mel, (128, 128))

    return mel.astype("float32").reshape(128, 128, 1)



# ===========================================================
# VOICE FEATURE EXTRACTION (torchaudio MFCC)
# ===========================================================

def extract_voice_features(audio_path, sample_rate=22050, duration=3,
                           n_mfcc=40, n_mels=128):

    try:
        wav, sr = torchaudio.load(audio_path)

        wav = torch.mean(wav, dim=0, keepdim=True)

        if sr != sample_rate:
            wav = torchaudio.transforms.Resample(sr, sample_rate)(wav)

        target = sample_rate * duration
        L = wav.shape[1]

        if L > target:
            wav = wav[:, :target]
        else:
            wav = torch.nn.functional.pad(wav, (0, target - L))

        mfcc = torchaudio.transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_mels": n_mels}
        )(wav)

        mfcc = mfcc.squeeze(0).numpy()
        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

        return mfcc

    except Exception as e:
        print(f"[ERROR] MFCC failed for: {audio_path}")
        print(e)
        return None


# ===========================================================
# LOAD VOICE DATASET
# ===========================================================
def load_voice_dataset(dataset_path, emotions=None, augment=False):

    if emotions is None:
        emotions = config.MODEL_EMOTIONS

    X, Y = [], []

    print(f"📌 Loading voice dataset from: {dataset_path}")

    for idx, emo in enumerate(emotions):
        emo_path = os.path.join(dataset_path, emo)
        if not os.path.exists(emo_path):
            print(f"[WARN] Missing folder: {emo_path}")
            continue

        print(f"→ Loading {emo}...")

        for f in os.listdir(emo_path):
            if not f.lower().endswith((".wav", ".mp3", ".flac")):
                continue

            file_path = os.path.join(emo_path, f)

            # training script uses mel-spectrogram
            mel = audio_to_melspec(file_path, augment=augment)

            X.append(mel)
            Y.append(idx)

    X = np.array(X)
    Y = keras.utils.to_categorical(Y, len(emotions))

    return X, Y


# ===========================================================
# EMOTION SMOOTHER
# ===========================================================
class EmotionSmoother:

    def __init__(self, window_size=5):
        self.window = window_size
        self.history = []

    def update(self, pred):
        self.history.append(pred)
        if len(self.history) > self.window:
            self.history.pop(0)
        return np.mean(self.history, axis=0)

    def reset(self):
        self.history = []
