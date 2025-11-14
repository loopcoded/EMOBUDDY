import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from models.voice_emotion_model import VoiceEmotionModel
from utils.data_preprocessing import audio_to_melspec
from config import config


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)


def load_voice_dataset(root, augment=False):
    X, y = [], []
    class_names = config.EMOTIONS

    for label_idx, emotion in enumerate(class_names):
        emotion_path = os.path.join(root, emotion)
        if not os.path.isdir(emotion_path):
            continue

        for f in os.listdir(emotion_path):
            if f.endswith((".wav", ".mp3", ".m4a")):
                file_path = os.path.join(emotion_path, f)
                mel = audio_to_melspec(file_path, augment=augment)
                X.append(mel)
                y.append(label_idx)

    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, len(class_names))
    return X, y


def train():
    print("📌 Loading voice dataset...")
    X_train, y_train = load_voice_dataset("datasets/voice_emotions/train", augment=True)
    X_test, y_test = load_voice_dataset("datasets/voice_emotions/test")

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42
    )

    print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

    model_inst = VoiceEmotionModel(
        num_classes=len(config.EMOTIONS),
        input_shape=(128, 128, 1)
    )
    model = model_inst.build_model()
    model_inst.compile_model(learning_rate=1e-3)


    # =====================================================
    # FIXED CALLBACKS — SAVING WEIGHTS ONLY (STABLE)
    # =====================================================
    checkpoint_path = "static/models/voice_emotion_model_best_weights.h5"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=True,
        save_weights_only=True,   # <<--- important
        monitor="val_accuracy",
        mode="max"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=8,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            verbose=1
        )
    ]


    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=40,
        batch_size=32,
        callbacks=callbacks
    )

    print("📌 Evaluating...")
    print(model.evaluate(X_test, y_test))


if __name__ == "__main__":
    train()
