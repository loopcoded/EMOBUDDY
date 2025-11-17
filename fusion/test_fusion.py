import os
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

from config import config
from models.voice_emotion_model import AttentionLayer
from utils.data_preprocessing import preprocess_face_image, audio_to_melspec

FACE_MODEL_PATH = "static/models/face_emotion_cnn_96_improved.keras"
VOICE_MODEL_PATH = "static/models/voice_emotion_model_full.keras"

FACE_TEST_ROOT = "datasets/clean_face_emotions/test"
VOICE_TEST_ROOT = "datasets/voice_emotions/test"

EMOTIONS = config.EMOTIONS

VOICE_WEIGHT = 0.7
FACE_WEIGHT = 0.3


def pick_random_file(root, emotion):
    emo_dir = os.path.join(root, emotion)
    files = [
        f for f in os.listdir(emo_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".wav"))
    ]
    return os.path.join(emo_dir, random.choice(files))


def load_face_sample(emotion):
    path = pick_random_file(FACE_TEST_ROOT, emotion)
    print("🖼 Face:", path)

    img = cv2.imread(path)
    face = preprocess_face_image(img, target_size=(96, 96)) / 255.0
    return np.expand_dims(face, 0)


def load_voice_sample(emotion):
    path = pick_random_file(VOICE_TEST_ROOT, emotion)
    print("🎤 Voice:", path)

    mel = audio_to_melspec(path, augment=False)
    return np.expand_dims(mel, 0)


def main():
    print("📌 Loading face model...")
    face_model = keras.models.load_model(FACE_MODEL_PATH)

    print("📌 Loading voice model...")
    voice_model = keras.models.load_model(
        VOICE_MODEL_PATH,
        custom_objects={"AttentionLayer": AttentionLayer}
    )

    target_emotion = random.choice(EMOTIONS)
    print("\n🎯 Testing on:", target_emotion)

    x_face = load_face_sample(target_emotion)
    x_voice = load_voice_sample(target_emotion)

    face_probs = face_model.predict(x_face)[0]
    voice_probs = voice_model.predict(x_voice)[0]

    print("\nFace →", EMOTIONS[np.argmax(face_probs)], face_probs.max())
    print("Voice →", EMOTIONS[np.argmax(voice_probs)], voice_probs.max())

    fused = (VOICE_WEIGHT * voice_probs) + (FACE_WEIGHT * face_probs)
    fused /= fused.sum()

    print("\nFusion →", EMOTIONS[np.argmax(fused)], fused.max())
    print("\n✅ Simple fusion successful.")


if __name__ == "__main__":
    main()
