"""
models/face_emotion_model.py - Refined Face Emotion Recognition Model
Key fixes:
- Simplified model architecture to prevent overfitting on 48x48 images.
- Fixed compile_model to ACTUALLY use FocalLoss.
- Removed confusing metrics (Precision/Recall) that were reporting 0.0.
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np


class FocalLoss(keras.losses.Loss):
    """Focal Loss to handle class imbalance better than simple weighting."""
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
    
    def call(self, y_true, y_pred):
        # Apply label smoothing
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / tf.cast(tf.shape(y_true)[-1], dtype=tf.float32)
        
        # Compute focal loss
        epsilon = keras.backend.epsilon()
        y_pred = keras.backend.clip(y_pred, epsilon, 1. - epsilon)
        
        # Calculate cross-entropy
        cross_entropy = -y_true * keras.backend.log(y_pred)
        
        # Calculate focal loss weight (pt = y_pred for true class)
        # This is a common implementation: (1-pt)^gamma
        p_t = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal_weight = tf.pow(1. - p_t, self.gamma)
        
        # Apply alpha (weighting for positive class)
        alpha_factor = self.alpha
        
        # Full focal loss
        # Note: We apply alpha to the summed cross-entropy for the true class
        # A more direct formula: -alpha * (1-pt)^gamma * log(pt)
        # We'll use the cross_entropy calculated with label smoothing
        # and weigh it by the focal_weight
        
        loss = alpha_factor * focal_weight * tf.reduce_sum(cross_entropy, axis=-1)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha,
            'label_smoothing': self.label_smoothing
        })
        return config


class FaceEmotionModel:
    """Face emotion recognition model optimized for FER2013-style data."""
    
    def __init__(self, num_classes=7, input_shape=(48, 48, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self):
        """
        Build a *simplified* CNN to reduce overfitting.
        """
        inputs = keras.Input(shape=self.input_shape)
        
        # Convert RGB to grayscale if needed
        if self.input_shape[-1] == 3:
            x = layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x))(inputs)
        else:
            x = inputs
        
        # Normalize
        x = layers.Rescaling(1./255)(x)
        
        # Moderate L2 regularization
        l2_reg = keras.regularizers.l2(0.001) # Slightly increased L2
        
        # Block 1:
        x = layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 2:
        x = layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Block 3:
        x = layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layer
        x = layers.Dense(256, kernel_regularizer=l2_reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.5)(x) # Increased dropout
        
        # Output
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = models.Model(inputs=inputs, outputs=outputs)
        return self.model
    
    def compile_model(self, learning_rate=1e-3, label_smoothing=0.0, use_focal_loss=True):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    
        if use_focal_loss:
            print("✅ Compiling with FocalLoss.")
            loss_fn = FocalLoss(label_smoothing=label_smoothing, name='focal_loss')
        else:
            print("✅ Compiling with CategoricalCrossentropy.")
            loss_fn = keras.losses.CategoricalCrossentropy(
                label_smoothing=label_smoothing
            )
    
        self.model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=[
                keras.metrics.CategoricalAccuracy(name='accuracy'),
                keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
                # Removed Precision/Recall to avoid confusion
            ]
        )

    
    def get_callbacks(self, model_path, patience=15):
        """Get training callbacks."""
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
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir='./logs/face_model',
                histogram_freq=1
            )
        ]
    
    def predict(self, face_image):
        """Predict emotion from face image."""
        if len(face_image.shape) == 3:
            face_image = np.expand_dims(face_image, axis=0)
        predictions = self.model.predict(face_image, verbose=0)
        return predictions[0]
    
    def load_model(self, model_path):
        """Load model with custom objects."""
        # This is CRUCIAL for loading a model trained with custom loss
        custom_objects = {'FocalLoss': FocalLoss}
        self.model = keras.models.load_model(model_path, custom_objects=custom_objects)
    
    def save_model(self, model_path):
        """Save model."""
        self.model.save(model_path)
    
    def get_model_summary(self):
        """Print model summary."""
        if self.model:
            return self.model.summary()
        return "Model not built yet. Call build_model() first."


def get_data_augmentation():
    """Strong augmentation for emotion recognition."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.15),
        layers.RandomTranslation(0.1, 0.1),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
    ])