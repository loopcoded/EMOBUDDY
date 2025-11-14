# training/train_clean_face_model.py

import os
import tensorflow as tf
from tensorflow import keras
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from models.face_emotion_model import FaceEmotionModelTL


# -------------------------
# GPU MEMORY GROWTH
# -------------------------
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
    except:
        pass


# -------------------------
# CONFIG
# -------------------------
DATASET_PATH = "datasets/clean_face_emotions/"
SAVE_PATH = "static/models/face_emotion_final"  # SavedModel folder

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS_HEAD = 6
EPOCHS_FINE = 18

LR_HEAD = 1e-3
LR_FINE = 5e-5
LABEL_SMOOTHING = 0.05

NUM_CLASSES = 7


# -------------------------
# LOAD DATASET FROM DISK
# -------------------------
def load_dataset_from_disk():
    """
    Loads train/val/test datasets using tf.image_dataset_from_directory
    (streams from disk → no RAM explosion)
    """

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH + "train",
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH + "val",
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATASET_PATH + "test",
        labels="inferred",
        label_mode="categorical",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Performance optimizations
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds


# -------------------------------------------------
# COMPUTE CLASS WEIGHTS (for imbalanced datasets)
# -------------------------------------------------
def compute_class_weights_from_dataset(train_ds):
    labels = []
    for _, y in train_ds.unbatch().take(20000):  # sample 20k only to speed up
        labels.append(np.argmax(y.numpy()))

    labels = np.array(labels)
    classes = np.unique(labels)

    cw = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    cw = {int(c): float(w) for c, w in zip(classes, cw)}
    print("Class Weights:", cw)
    return cw


# -------------------------
# TRAINING PIPELINE
# -------------------------
def train():

    print("🔄 Loading dataset from disk...")
    train_ds, val_ds, test_ds = load_dataset_from_disk()

    # Compute weights
    class_weight = compute_class_weights_from_dataset(train_ds)

    print("📌 Building model...")
    model_inst = FaceEmotionModelTL(
        num_classes=NUM_CLASSES,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        backbone="mobilenetv2"
    )

    model = model_inst.build_model(backbone_trainable=False)

    # -------------------------
    # PHASE 1: Train Head
    # -------------------------
    print("\n===== PHASE 1: Train Head =====")
    model_inst.compile_model(
        lr=LR_HEAD,
        label_smoothing=LABEL_SMOOTHING,
        use_focal_loss=False
    )

    callbacks = model_inst.get_callbacks(
        model_path=SAVE_PATH,
        patience=8
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        class_weight=class_weight,
        callbacks=callbacks
    )

    # -------------------------
    # PHASE 2: Fine-Tune Backbone
    # -------------------------
    print("\n===== PHASE 2: Fine-Tune Backbone =====")

    # Unfreeze last ~30 layers
    model_inst.unfreeze_backbone()

    model_inst.compile_model(
        lr=LR_FINE,
        label_smoothing=LABEL_SMOOTHING,
        use_focal_loss=True
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE,
        class_weight=class_weight,
        callbacks=callbacks
    )

    # -------------------------
    # FINAL EVAL
    # -------------------------
    print("\n📌 Loading Best Model...")
    model_inst.load_model(SAVE_PATH)

    print("\n📊 Validation Evaluation:")
    print(model_inst.model.evaluate(val_ds))

    print("\n📊 Test Evaluation:")
    print(model_inst.model.evaluate(test_ds))


if __name__ == "__main__":
    train()
