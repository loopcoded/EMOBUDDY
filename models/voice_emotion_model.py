# models/voice_emotion_model.py
"""
Stable Voice Emotion Model with safe Attention layer and clear save/load API.
Designed for inputs shaped (128,128,1) (mel-spectrograms).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


class AttentionLayer(layers.Layer):
    """
    Simple, stable attention over time-steps.
    Creates variables in build() (so they are created only once).
    Input: (batch, timesteps, features)
    Output: (batch, features) -- weighted sum across timesteps
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        # input_shape = (batch, timesteps, features)
        features = int(input_shape[-1])
        # weight vector for scoring (features -> 1)
        self.W = self.add_weight(
            name="att_W",
            shape=(features, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        # optional bias (broadcastable across timesteps)
        self.b = self.add_weight(
            name="att_b",
            shape=(1,),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x, mask=None):
        # x: (batch, timesteps, features)
        # score: (batch, timesteps, 1)
        # Use tensordot to avoid shape confusion
        score = tf.tensordot(x, self.W, axes=[[2], [0]])  # (batch, timesteps, 1)
        score = score + self.b  # bias broadcast
        score = tf.nn.tanh(score)
        att_weights = tf.nn.softmax(score, axis=1)  # softmax over timesteps
        # weighted sum
        weighted = x * att_weights  # (batch, timesteps, features)
        context = tf.reduce_sum(weighted, axis=1)  # (batch, features)
        return context

    def get_config(self):
        cfg = super().get_config()
        return cfg


class VoiceEmotionModel:
    """
    CNN + Bi-GRU + Attention model for mel-spectrogram inputs.
    Save/load methods use weights-only to avoid SavedModel 'options' errors.
    """

    def __init__(self, num_classes=7, input_shape=(128, 128, 1)):
        self.num_classes = int(num_classes)
        self.input_shape = tuple(input_shape)
        self.model = None

    def build_model(self):
        # Input shape e.g. (128,128,1)
        inputs = layers.Input(shape=self.input_shape, name="mel_input")

        # CNN feature extractor (keeps spatial dims so we can form a sequence)
        x = layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
        x = layers.MaxPooling2D((2, 2))(x)  # 128 -> 64

        x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)  # 64 -> 32

        x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
        x = layers.MaxPooling2D((2, 2))(x)  # 32 -> 16

        # Compute sequence length from known input shape.
        # After three (2x2) pools spatial dims are input_shape[0]//8, input_shape[1]//8
        h = self.input_shape[0] // 8
        w = self.input_shape[1] // 8
        seq_len = h * w  # e.g. (16*16)=256 for 128x128 input

        # Reshape into (batch, timesteps, features)
        # last channel is 128 (from last Conv2D)
        x = layers.Reshape((seq_len, 128))(x)

        # Bi-GRU for temporal modelling
        x = layers.Bidirectional(layers.GRU(64, return_sequences=True))(x)

        # Attention layer reduces (batch, seq_len, 256) -> (batch, 256)
        x = AttentionLayer()(x)

        # Classification head
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation="relu")(x)
        x = layers.Dropout(0.4)(x)

        outputs = layers.Dense(self.num_classes, activation="softmax", name="emotion")(x)

        self.model = models.Model(inputs=inputs, outputs=outputs, name="VoiceEmotionModel")
        return self.model

    def compile_model(self, learning_rate=1e-3):
        if self.model is None:
            raise ValueError("Call build_model() before compile_model()")
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    # weights-only save/load to avoid TensorFlow SavedModel 'options' incompat issues
    def save_weights(self, path):
        # ensure directory exists
        osdir = tf.io.gfile.exists(path) and path.endswith(".h5")
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def summary(self):
        if self.model is None:
            return "Model not built"
        return self.model.summary()
