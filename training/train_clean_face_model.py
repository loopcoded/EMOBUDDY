# training/train_face_cnn_96_improved.py

import os
import tensorflow as tf
from tensorflow import keras
import numpy as np

from models.face_emotion_model import ImprovedFaceCNN
from utils.data_preprocessing import load_clean_face_dataset
from config import config


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


# -------------------------
# CONFIG
# -------------------------
DATASET_PATH = "datasets/clean_face_emotions/"
IMG_SIZE = (96, 96)
BATCH_SIZE = 64
EPOCHS = 40

NUM_CLASSES = len(config.EMOTIONS)
MODEL_SAVE_PATH = "static/models/face_emotion_cnn_96_improved.keras"


def train():
    # -------------------------
    # LOAD DATA
    # -------------------------
    print("🔄 Loading CLEAN dataset (face)...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_clean_face_dataset(
        DATASET_PATH,
        img_size=IMG_SIZE
    )

    print("TRAIN:", X_train.shape, "| VAL:", X_val.shape, "| TEST:", X_test.shape)

    # NOTE: No /255.0 here because the model has a Rescaling(1/255)

    # -------------------------
    # BUILD MODEL
    # -------------------------
    print("📌 Building ImprovedFaceCNN model...")
    model_builder = ImprovedFaceCNN(
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        num_classes=NUM_CLASSES
    )
    model = model_builder.build()
    model.summary()

    # -------------------------
    # COMPILE
    # -------------------------
    optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=[
            "accuracy",
            keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
        ],
    )

    # -------------------------
    # CALLBACKS
    # -------------------------
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            mode="max",
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    # -------------------------
    # TRAIN
    # -------------------------
    print("\n===== TRAINING ImprovedFaceCNN =====")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,
    )

    # -------------------------
    # EVALUATE
    # -------------------------
    print("\n📊 Final Test Evaluation (using best weights in memory):")
    test_metrics = model.evaluate(X_test, y_test, batch_size=BATCH_SIZE)
    print(test_metrics)

    # Optional: explicitly save final best model again
    print(f"\n💾 Saving final model to: {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)


if __name__ == "__main__":
    train()
