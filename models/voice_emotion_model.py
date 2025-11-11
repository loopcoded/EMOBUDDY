"""
models/voice_emotion_model.py – Voice Emotion Recognition Model
Optimized for atypical prosody & short utterances
"""

import tensorflow as tf
from tensorflow import keras

layers = keras.layers
models = keras.models
regularizers = keras.regularizers
K = keras.backend
import numpy as np


class VoiceEmotionModel:
    """
    Voice emotion recognition model.
    Expects MFCC tensors shaped (n_mfcc, time_steps, 1) per sample.
    """

    def __init__(self, num_classes=7, input_shape=(40, 128, 1)):
        """
        Args:
            num_classes: number of emotion classes
            input_shape: (n_mfcc, time_steps, channels) – e.g., (40, 331, 1)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None

    def build_model(self, architecture='cnn_lstm', l2=1e-4, dropout=0.3):
        """
        Build the emotion recognition model.

        Args:
            architecture: 'cnn', 'cnn_lstm', or 'attention'
            l2: L2 weight decay
            dropout: default dropout rate
        """
        inputs = layers.Input(shape=self.input_shape, name="mfcc_input")

        # Normalize per sample - use BatchNormalization instead of LayerNorm
        # to avoid shape mismatch issues when loading
        x = layers.BatchNormalization(name="input_bn")(inputs)

        if architecture == 'cnn':
            x = self._build_cnn_layers(x, l2=l2, dropout=dropout)
        elif architecture == 'cnn_lstm':
            x = self._build_cnn_lstm_layers(x, l2=l2, dropout=dropout)
        elif architecture == 'attention':
            x = self._build_attention_layers(x, l2=l2, dropout=dropout)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Classification head
        x = layers.Dense(128, activation='relu',
                         kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)

        outputs = layers.Dense(self.num_classes, activation='softmax',
                               name="emotion")(x)

        self.model = models.Model(inputs=inputs, outputs=outputs, name=f"voice_{architecture}")
        return self.model

    # ------------------------- sub-architectures ------------------------- #
    def _build_cnn_layers(self, x, l2=1e-4, dropout=0.3):
        x = layers.Conv2D(64, 3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, 3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)

        x = layers.Conv2D(128, 3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, 3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)

        x = layers.Conv2D(256, 3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, 3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        return x

    def _build_cnn_lstm_layers(self, x, l2=1e-4, dropout=0.3):
        # CNN feature extractor over (freq, time)
        x = layers.Conv2D(64, 3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)  # -> (freq/2, time/2)
        x = layers.Dropout(dropout)(x)

        x = layers.Conv2D(128, 3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)  # -> (freq/4, time/4)
        x = layers.Dropout(dropout)(x)

        # Get the shape after convolutions
        # Shape is now (batch, freq', time', channels)
        shape = K.int_shape(x)
        
        # Permute to (batch, time', freq', channels)
        x = layers.Permute((2, 1, 3), name="to_time_major")(x)
        
        # Reshape to (batch, time', freq'*channels)
        # Calculate the feature dimension
        freq_dim = shape[1]  # freq' after pooling
        channel_dim = shape[3]  # number of channels
        feature_dim = freq_dim * channel_dim
        
        # Use Reshape instead of Lambda for better shape inference
        x = layers.Reshape((-1, feature_dim), name="time_step_flatten")(x)

        # Temporal modeling with LSTM
        x = layers.LSTM(128, return_sequences=True, name="lstm_1")(x)
        x = layers.Dropout(dropout)(x)
        x = layers.LSTM(64, name="lstm_2")(x)
        x = layers.Dropout(dropout)(x)
        return x

    def _build_attention_layers(self, x, l2=1e-4, dropout=0.3):
        x = layers.Conv2D(64, 3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, 3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        # Get the shape after convolutions
        shape = K.int_shape(x)
        
        # Permute to (batch, time', freq', channels)
        x = layers.Permute((2, 1, 3), name="to_time_major")(x)
        
        # Reshape to (batch, time', freq'*channels)
        freq_dim = shape[1]
        channel_dim = shape[3]
        feature_dim = freq_dim * channel_dim
        
        x = layers.Reshape((-1, feature_dim), name="time_step_flatten")(x)

        attn = layers.MultiHeadAttention(num_heads=4, key_dim=32, dropout=dropout)
        y = attn(x, x)
        x = layers.Add()([x, y])
        x = layers.LayerNormalization()(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.5)(x)
        return x

    # ------------------------------ compile ------------------------------ #
    def compile_model(self, learning_rate=5e-4, label_smoothing=0.05):
        """
        Use a float LR so ReduceLROnPlateau can adjust it.
        Label smoothing improves generalization on imbalanced speech data.
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing),
            metrics=[
                "accuracy",
                keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy"),
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc", multi_label=True, num_labels=self.num_classes),
            ],
        )

    # ---------------------------- callbacks ------------------------------ #
    def get_callbacks(self, model_path, patience=12):
        return [
            keras.callbacks.ModelCheckpoint(
                model_path, monitor="val_loss", save_best_only=True, mode="min", verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True, verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-7, verbose=1
            ),
            keras.callbacks.TensorBoard(log_dir="./logs/voice_model", histogram_freq=1),
            keras.callbacks.CSVLogger("training_voice.csv"),
        ]

    # ------------------------------ I/O ---------------------------------- #
    def predict(self, voice_features):
        # Accept (n_mfcc, T) or (n_mfcc, T, 1) or (B, n_mfcc, T, 1)
        arr = np.array(voice_features)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        return self.model.predict(arr, verbose=0)[0]

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path)

    def save_model(self, model_path):
        self.model.save(model_path)

    def get_model_summary(self):
        if self.model is not None:
            return self.model.summary()
        print("Model not built yet. Call build_model() first.")