# models/fusion_model.py

import tensorflow as tf
from tensorflow.keras import layers, Model

class FusionModel:
    def __init__(self, face_embedding_dim=256, voice_embedding_dim=256, num_classes=7):
        self.face_embedding_dim = face_embedding_dim
        self.voice_embedding_dim = voice_embedding_dim
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        # ---------- FACE INPUT ----------
        face_in = layers.Input(shape=(self.face_embedding_dim,), name="face_embedding")

        # ---------- VOICE INPUT ----------
        voice_in = layers.Input(shape=(self.voice_embedding_dim,), name="voice_embedding")

        # ---------- MASKS (to handle missing modalities) ----------
        face_mask = layers.Input(shape=(1,), name="face_mask")
        voice_mask = layers.Input(shape=(1,), name="voice_mask")

        # Apply masks: if missing → zero out embedding
        face_vec = layers.Multiply()([face_in, face_mask])
        voice_vec = layers.Multiply()([voice_in, voice_mask])

        # ---------- CONCAT ----------
        combined = layers.Concatenate(name="concat_face_voice")([face_vec, voice_vec])

        # ---------- FUSION FC HEAD ----------
        x = layers.Dense(512, activation="relu")(combined)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(128, activation="relu")(x)

        # ---------- FINAL EMOTION OUTPUT ----------
        out = layers.Dense(self.num_classes, activation="softmax", name="emotion")(x)

        self.model = Model(
            inputs=[face_in, voice_in, face_mask, voice_mask],
            outputs=out,
            name="face_voice_fusion_model"
        )

        return self.model

    def compile_model(self, lr=1e-3):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)

    def save_model(self, path):
        self.model.save(path)
