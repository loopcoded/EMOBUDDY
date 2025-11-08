"""
models/voice_emotion_model.py - Voice Emotion Recognition Model
Optimized for autistic children's speech patterns and vocalizations
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np


class VoiceEmotionModel:
    """
    Voice emotion recognition model designed for autistic children.
    Considers unique speech patterns, prosody variations, and non-verbal vocalizations.
    """
    
    def __init__(self, num_classes=7, input_shape=(40, 128, 1)):
        """
        Initialize the voice emotion model.
        
        Args:
            num_classes: Number of emotion categories
            input_shape: Input spectrogram shape (n_mfcc, time_steps, channels)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self, architecture='cnn_lstm'):
        """
        Build the emotion recognition model.
        
        Args:
            architecture: 'cnn', 'cnn_lstm', or 'attention'
        """
        inputs = keras.Input(shape=self.input_shape)
        
        if architecture == 'cnn':
            x = self._build_cnn_layers(inputs)
        elif architecture == 'cnn_lstm':
            x = self._build_cnn_lstm_layers(inputs)
        elif architecture == 'attention':
            x = self._build_attention_layers(inputs)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Output layer
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def _build_cnn_layers(self, inputs):
        """Build CNN-based architecture."""
        # Normalization
        x = layers.BatchNormalization()(inputs)
        
        # Block 1
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 2
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 3
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        
        # Dense layers
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        return x
    
    def _build_cnn_lstm_layers(self, inputs):
        """Build CNN-LSTM architecture for temporal patterns."""
        # Normalization
        x = layers.BatchNormalization()(inputs)
        
        # CNN for feature extraction
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Reshape for LSTM
        shape = x.shape
        x = layers.Reshape((shape[1], shape[2] * shape[3]))(x)
        
        # LSTM for temporal modeling
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64)(x)
        x = layers.Dropout(0.3)(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        return x
    
    def _build_attention_layers(self, inputs):
        """Build attention-based architecture."""
        # Normalization
        x = layers.BatchNormalization()(inputs)
        
        # CNN for initial feature extraction
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Reshape for attention
        shape = x.shape
        x = layers.Reshape((shape[1] * shape[2], shape[3]))(x)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=4,
            key_dim=32,
            dropout=0.3
        )(x, x)
        
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.5)(x)
        
        # Dense layers
        x = layers.Dense(128, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        return x
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        # Learning rate schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )
        
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Compile with weighted categorical crossentropy
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )
    
    def get_callbacks(self, model_path, patience=15):
        """
        Get training callbacks.
        
        Args:
            model_path: Path to save best model
            patience: Early stopping patience
        """
        callbacks = [
            # Save best model
            keras.callbacks.ModelCheckpoint(
                model_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir='./logs/voice_model',
                histogram_freq=1
            ),
            
            # Custom callback for logging
            keras.callbacks.CSVLogger('training_voice.csv')
        ]
        
        return callbacks
    
    def predict(self, voice_features):
        """
        Predict emotion from voice features.
        
        Args:
            voice_features: Preprocessed voice spectrogram
        
        Returns:
            predictions: Array of probabilities for each emotion
        """
        if len(voice_features.shape) == 3:
            voice_features = np.expand_dims(voice_features, axis=0)
        
        predictions = self.model.predict(voice_features, verbose=0)
        return predictions[0]
    
    def load_model(self, model_path):
        """Load a trained model from file."""
        self.model = keras.models.load_model(model_path)
    
    def save_model(self, model_path):
        """Save the model to file."""
        self.model.save(model_path)
    
    def get_model_summary(self):
        """Print model architecture summary."""
        if self.model:
            return self.model.summary()
        else:
            return "Model not built yet. Call build_model() first."

