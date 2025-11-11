"""
models/voice_emotion_model.py - Fixed Voice Emotion Recognition Model

Key fixes:
- Simplified architecture to prevent overfitting
- Better regularization
- Fixed shape handling
- Proper normalization
"""

import tensorflow as tf
from tensorflow import keras
layers = keras.layers
models = keras.models
regularizers = keras.regularizers
K = keras.backend
import numpy as np


class VoiceEmotionModel:
    """Voice emotion recognition model for MFCC features."""

    def __init__(self, num_classes=7, input_shape=(40, 128, 1)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None

    def build_model(self, architecture='cnn_lstm', l2=0.001, dropout=0.4):
        """
        Build model with specified architecture.
        
        Args:
            architecture: 'cnn', 'cnn_lstm', or 'attention'
            l2: L2 regularization strength
            dropout: Dropout rate
        """
        inputs = layers.Input(shape=self.input_shape, name="mfcc_input")
        
        # Normalization
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
        
        self.model = models.Model(inputs=inputs, outputs=outputs,
                                 name=f"voice_{architecture}")
        return self.model

    def _build_cnn_layers(self, x, l2=0.001, dropout=0.4):
        """CNN-only architecture."""
        # Block 1
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)
        
        # Block 2
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)
        
        # Block 3
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        return x

    def _build_cnn_lstm_layers(self, x, l2=0.001, dropout=0.4):
        """CNN + LSTM for temporal modeling."""
        # CNN feature extraction
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout * 0.5)(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout * 0.5)(x)
        
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout * 0.5)(x)
        
        # Reshape for LSTM: (batch, time, features)
        shape = K.int_shape(x)
        x = layers.Permute((2, 1, 3))(x)  # (batch, time, freq, channels)
        
        # Flatten frequency and channel dimensions
        freq_dim = shape[1]
        channel_dim = shape[3]
        x = layers.Reshape((-1, freq_dim * channel_dim))(x)
        
        # LSTM layers
        x = layers.LSTM(128, return_sequences=True,
                       kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.Dropout(dropout)(x)
        x = layers.LSTM(64, kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.Dropout(dropout)(x)
        
        return x

    def _build_attention_layers(self, x, l2=0.001, dropout=0.4):
        """CNN + Multi-head attention."""
        # CNN feature extraction
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout * 0.5)(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(l2))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(dropout * 0.5)(x)
        
        # Reshape for attention
        shape = K.int_shape(x)
        x = layers.Permute((2, 1, 3))(x)
        freq_dim = shape[1]
        channel_dim = shape[3]
        x = layers.Reshape((-1, freq_dim * channel_dim))(x)
        
        # Multi-head attention
        attn = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=64,
            dropout=dropout
        )
        y = attn(x, x)
        x = layers.Add()([x, y])
        x = layers.LayerNormalization()(x)
        
        # Pool
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(dropout)(x)
        
        return x

    def compile_model(self, learning_rate=0.0001, label_smoothing=0.1):
        """Compile model with optimizer and loss."""
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss=keras.losses.CategoricalCrossentropy(
                label_smoothing=label_smoothing
            ),
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc', multi_label=True)
            ]
        )

    def get_callbacks(self, model_path, patience=15):
        """Training callbacks."""
        return [
            keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=patience,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir='./logs/voice_model',
                histogram_freq=1
            )
        ]

    def predict(self, voice_features):
        """Predict emotion from voice features."""
        arr = np.array(voice_features)
        if arr.ndim == 2:
            arr = arr[..., np.newaxis]
        if arr.ndim == 3:
            arr = arr[np.newaxis, ...]
        return self.model.predict(arr, verbose=0)[0]

    def load_model(self, model_path):
        """Load saved model."""
        self.model = keras.models.load_model(model_path)

    def save_model(self, model_path):
        """Save model."""
        self.model.save(model_path)

    def get_model_summary(self):
        """Print model summary."""
        if self.model is not None:
            return self.model.summary()
        print("Model not built yet. Call build_model() first.")