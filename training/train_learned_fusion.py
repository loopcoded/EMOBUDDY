# training/train_fusion_model.py

import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split

from models.emotion_fusion_enhanced import FusionModel
from models.face_emotion_model import FaceEmotionModelTL
from models.voice_emotion_model import VoiceEmotionModel
from utils.data_preprocessing import load_clean_face_dataset, audio_to_melspec
from config import config


FACE_MODEL_PATH = "static/models/face_emotion_final"
VOICE_MODEL_PATH = "static/models/voice_emotion_model_best_weights.h5"
FUSION_SAVE_PATH = "static/models/emotion_fusion_model"


# --------------------------------------------------
# LOAD TRAINED SUB-MODELS (face + voice)
# --------------------------------------------------
def load_face_embedding_model():
    face_model = tf.keras.models.load_model(FACE_MODEL_PATH)
    embedding_layer = face_model.get_layer(index=-2)
    return tf.keras.Model(inputs=face_model.input, outputs=embedding_layer.output)


def load_voice_embedding_model():
    voice_model = tf.keras.models.load_model(VOICE_MODEL_PATH)
    embedding_layer = voice_model.get_layer(index=-2)
    return tf.keras.Model(inputs=voice_model.input, outputs=embedding_layer.output)


# --------------------------------------------------
# BUILD DATASET FOR FUSION TRAINING
# --------------------------------------------------
def load_fusion_dataset():
    X_train_f, y_train_f, X_val_f, y_val_f, X_test_f, y_test_f = \
        load_clean_face_dataset("datasets/clean_face_emotions", img_size=(96, 96))

    X_train_v, y_train_v = [], []
    voice_root = "datasets/voice_emotions/train"

    for label_idx, emotion in enumerate(config.EMOTIONS):
        path = os.path.join(voice_root, emotion)
        if not os.path.isdir(path): continue

        for file in os.listdir(path):
            if file.endswith((".wav", ".mp3", ".m4a")):
                mel = audio_to_melspec(os.path.join(path, file))
                X_train_v.append(mel)
                y_train_v.append(label_idx)

    return (
        X_train_f, y_train_f,
        np.array(X_train_v), tf.keras.utils.to_categorical(y_train_v, len(config.EMOTIONS))
    )


# --------------------------------------------------
# MAIN TRAINING
# --------------------------------------------------
def train_fusion():
    print("📌 Loading trained models...")
    face_embed_model = load_face_embedding_model()
    voice_embed_model = load_voice_embedding_model()

    print("📌 Preparing fusion dataset...")
    X_face, y_face, X_voice, y_voice = load_fusion_dataset()

    # For simplicity, assume equal counts → random match (you can sync by filename later)
    N = min(len(X_face), len(X_voice))
    X_face = X_face[:N]
    y_face = y_face[:N]
    X_voice = X_voice[:N]
    y_voice = y_voice[:N]

    # Generate embeddings
    face_embeddings = face_embed_model.predict(X_face, batch_size=32)
    voice_embeddings = voice_embed_model.predict(X_voice, batch_size=32)

    # MASK = ones (means modality is available)
    face_mask = np.ones((N, 1))
    voice_mask = np.ones((N, 1))

    # Train-val split
    X_train, X_val, y_train, y_val = train_test_split(
        np.arange(N), y_face, test_size=0.15, random_state=42
    )

    fusion = FusionModel(
        face_embedding_dim=face_embeddings.shape[1],
        voice_embedding_dim=voice_embeddings.shape[1],
        num_classes=len(config.EMOTIONS)
    )

    model = fusion.build_model()
    fusion.compile_model(lr=1e-3)

    print("📌 Training fusion model...")

    model.fit(
        {
            "face_embedding": face_embeddings[X_train],
            "voice_embedding": voice_embeddings[X_train],
            "face_mask": face_mask[X_train],
            "voice_mask": voice_mask[X_train],
        },
        y_train,
        validation_data=(
            {
                "face_embedding": face_embeddings[X_val],
                "voice_embedding": voice_embeddings[X_val],
                "face_mask": face_mask[X_val],
                "voice_mask": voice_mask[X_val],
            },
            y_val
        ),
        epochs=20,
        batch_size=32,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                FUSION_SAVE_PATH,
                save_best_only=True,
                monitor="val_accuracy",
                mode="max"
            )
        ]
    )

    print("✅ Fusion model saved at:", FUSION_SAVE_PATH)


if __name__ == "__main__":
    train_fusion()
