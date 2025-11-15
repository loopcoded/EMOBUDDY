# training/train_fusion_model.py

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from models.emotion_fusion_enhanced import FusionModel
from models.face_emotion_model import FaceEmotionModelTL
from models.voice_emotion_model import VoiceEmotionModel
from utils.data_preprocessing import load_clean_face_dataset, audio_to_melspec
from config import config

# -------------------------
# PATHS
# -------------------------
FACE_MODEL_PATH = "static/models/face_emotion_final"              # SavedModel folder
VOICE_WEIGHTS_PATH = "static/models/voice_emotion_model_best_weights.h5"
FUSION_SAVE_PATH = "static/models/emotion_fusion_model"           # SavedModel folder

FACE_DATASET_ROOT = "datasets/clean_face_emotions"
VOICE_TRAIN_ROOT = "datasets/voice_emotions/train"


# -------------------------
# GPU MEMORY GROWTH
# -------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        print(f"GPUs detected: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠ No GPU detected. Using CPU.")


# --------------------------------------------------
# LOAD TRAINED SUB-MODELS (face + voice) AS EMBEDDERS
# --------------------------------------------------
def load_face_embedding_model():
    """
    Load the trained face model and return a model that outputs
    the penultimate layer (embedding).
    """
    print("📌 Loading face model for embeddings...")
    face_inst = FaceEmotionModelTL(
        num_classes=len(config.EMOTIONS),
        input_shape=(224, 224, 3),
        backbone="mobilenetv2"
    )
    # This will load SavedModel + custom FocalLoss internally
    face_inst.load_model(FACE_MODEL_PATH)
    face_model = face_inst.model

    # Take the layer before final 'emotion' Dense
    # In your FaceEmotionModelTL:
    # ... Dense(128) -> BN -> ReLU -> Dropout -> Dense(num_classes, name='emotion')
    # So index -2 = Dropout(…) output (dim 128)
    embedding_layer = face_model.get_layer(index=-2)
    print("Face embedding layer:", embedding_layer.name,
          "dim =", embedding_layer.output_shape[-1])

    embedder = tf.keras.Model(
        inputs=face_model.input,
        outputs=embedding_layer.output,
        name="face_embedding_model"
    )
    return embedder


def load_voice_embedding_model():
    """
    Rebuild VoiceEmotionModel architecture and load best weights.
    Then return a model that outputs the penultimate layer (embedding).
    """
    print("📌 Loading voice model for embeddings...")
    voice_inst = VoiceEmotionModel(
        num_classes=len(config.EMOTIONS),
        input_shape=(128, 128, 1)
    )
    voice_model = voice_inst.build_model()
    voice_inst.compile_model(learning_rate=1e-3)

    # Load weights-only checkpoint
    voice_inst.load_weights(VOICE_WEIGHTS_PATH)

    # In VoiceEmotionModel:
    # ... Dense(256) -> Dropout -> Dense(128) -> Dropout -> Dense(num_classes, 'emotion')
    # So index -2 = Dropout after Dense(128) (dim 128)
    embedding_layer = voice_model.get_layer(index=-2)
    print("Voice embedding layer:", embedding_layer.name,
          "dim =", embedding_layer.output_shape[-1])

    embedder = tf.keras.Model(
        inputs=voice_model.input,
        outputs=embedding_layer.output,
        name="voice_embedding_model"
    )
    return embedder


# --------------------------------------------------
# BUILD DATASET FOR FUSION TRAINING
# --------------------------------------------------
def load_fusion_dataset():
    """
    Returns:
        X_face: np.array of face images
        y_face: one-hot labels (same order as X_face)
        X_voice: np.array of mel-spectrograms
        y_voice: one-hot labels (same order as X_voice)
    """
    print("📌 Loading face dataset for fusion...")
    # Adjust img_size to 224x224 to match face model
    X_train_f, y_train_f, X_val_f, y_val_f, X_test_f, y_test_f = \
        load_clean_face_dataset(FACE_DATASET_ROOT, img_size=(224, 224))

    # For now we will use only train split for fusion training
    X_face = X_train_f
    y_face = y_train_f

    print("📌 Loading voice dataset for fusion...")
    X_voice_list, y_voice_list = [], []
    emotions = config.EMOTIONS

    for label_idx, emotion in enumerate(emotions):
        emo_dir = os.path.join(VOICE_TRAIN_ROOT, emotion)
        if not os.path.isdir(emo_dir):
            continue

        for file in os.listdir(emo_dir):
            if file.lower().endswith((".wav", ".mp3", ".m4a")):
                file_path = os.path.join(emo_dir, file)
                mel = audio_to_melspec(file_path, augment=False)
                X_voice_list.append(mel)
                y_voice_list.append(label_idx)

    X_voice = np.array(X_voice_list)
    y_voice = tf.keras.utils.to_categorical(y_voice_list, len(emotions))

    print("Face samples:", X_face.shape, "| Voice samples:", X_voice.shape)
    return X_face, y_face, X_voice, y_voice


# --------------------------------------------------
# MAIN TRAINING
# --------------------------------------------------
def train_fusion():
    # 1. Load embedder models
    face_embed_model = load_face_embedding_model()
    voice_embed_model = load_voice_embedding_model()

    # 2. Load raw data
    X_face, y_face, X_voice, y_voice = load_fusion_dataset()

    # 3. Align counts (simple index-based pairing, can be improved later)
    N = min(len(X_face), len(X_voice))
    X_face = X_face[:N]
    y_face = y_face[:N]
    X_voice = X_voice[:N]
    y_voice = y_voice[:N]

    print("📌 Using N =", N, "paired samples for fusion training.")

    # 4. Generate embeddings
    print("📌 Generating face embeddings...")
    face_embeddings = face_embed_model.predict(X_face, batch_size=32, verbose=1)

    print("📌 Generating voice embeddings...")
    voice_embeddings = voice_embed_model.predict(X_voice, batch_size=32, verbose=1)

    # 5. Masks = 1 (both modalities available in training)
    face_mask = np.ones((N, 1), dtype=np.float32)
    voice_mask = np.ones((N, 1), dtype=np.float32)

    # 6. Train-val split based on indices
    idx = np.arange(N)
    X_train_idx, X_val_idx, y_train, y_val = train_test_split(
        idx, y_face, test_size=0.15, random_state=42
    )

    # 7. Build fusion model with correct embedding dims
    fusion = FusionModel(
        face_embedding_dim=face_embeddings.shape[1],
        voice_embedding_dim=voice_embeddings.shape[1],
        num_classes=len(config.EMOTIONS)
    )
    model = fusion.build_model()
    fusion.compile_model(lr=1e-3)

    model.summary()

    # 8. Train
    print("📌 Training fusion model...")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=FUSION_SAVE_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False  # Save full SavedModel
        )
    ]

    model.fit(
        {
            "face_embedding": face_embeddings[X_train_idx],
            "voice_embedding": voice_embeddings[X_train_idx],
            "face_mask": face_mask[X_train_idx],
            "voice_mask": voice_mask[X_train_idx],
        },
        y_train,
        validation_data=(
            {
                "face_embedding": face_embeddings[X_val_idx],
                "voice_embedding": voice_embeddings[X_val_idx],
                "face_mask": face_mask[X_val_idx],
                "voice_mask": voice_mask[X_val_idx],
            },
            y_val,
        ),
        epochs=20,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    print("✅ Fusion model saved at:", FUSION_SAVE_PATH)


if __name__ == "__main__":
    train_fusion()
