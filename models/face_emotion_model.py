import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.applications import MobileNetV2
import numpy as np


class FaceEmotionModel:
    """
    Face emotion recognition model specifically designed for autistic children.
    Uses transfer learning with MobileNetV2 for efficiency.
    """
    
    def __init__(self, num_classes=7, input_shape=(48, 48, 3)):
        """
        Initialize the face emotion model.
        
        Args:
            num_classes: Number of emotion categories
            input_shape: Input image shape (height, width, channels)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None
        
    def build_model(self, use_transfer_learning=True):
        """
        Build the emotion recognition model.
        
        Args:
            use_transfer_learning: Whether to use pre-trained MobileNetV2
        """
        if use_transfer_learning:
            # Use MobileNetV2 as base (efficient for real-time processing)
            base_model = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet',
                pooling='avg'
            )
            
            # Freeze early layers, fine-tune later layers
            for layer in base_model.layers[:-30]:
                layer.trainable = False
            
            inputs = keras.Input(shape=self.input_shape)
            
            # Preprocessing
            x = layers.Rescaling(1./255)(inputs)
            
            # Base model
            x = base_model(x, training=False)
            
            # Custom head for emotion detection
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            # Output layer
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
            
        else:
            # Custom CNN architecture
            inputs = keras.Input(shape=self.input_shape)
            
            # Preprocessing
            x = layers.Rescaling(1./255)(inputs)
            
            # Block 1
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
            
            # Block 2
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
            
            # Block 3
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.MaxPooling2D((2, 2))(x)
            x = layers.Dropout(0.25)(x)
            
            # Dense layers
            x = layers.Flatten()(x)
            x = layers.Dense(256, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            
            x = layers.Dense(128, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            
            # Output
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        # Create model
        self.model = models.Model(inputs=inputs, outputs=outputs)
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer and loss function.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        # Use custom optimizer with learning rate schedule
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=1000,
            decay_rate=0.9
        )
        
        optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Compile with focal loss for imbalanced classes (common in autism datasets)
        self.model.compile(
            optimizer=optimizer,
            loss=self._focal_loss(),
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )
    
    def _focal_loss(self, gamma=2.0, alpha=0.25):
        """
        Focal loss for handling class imbalance.
        Helps with minority emotion classes in autism datasets.
        
        Args:
            gamma: Focusing parameter
            alpha: Balancing parameter
        """
        def focal_loss_fixed(y_true, y_pred):
            epsilon = keras.backend.epsilon()
            y_pred = keras.backend.clip(y_pred, epsilon, 1.0 - epsilon)
            
            cross_entropy = -y_true * keras.backend.log(y_pred)
            loss = alpha * keras.backend.pow(1 - y_pred, gamma) * cross_entropy
            
            return keras.backend.sum(loss, axis=-1)
        
        return focal_loss_fixed
    
    def get_callbacks(self, model_path, patience=10):
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
                monitor='val_loss',
                save_best_only=True,
                mode='min',
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
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            keras.callbacks.TensorBoard(
                log_dir='./logs/face_model',
                histogram_freq=1
            )
        ]
        
        return callbacks
    
    def predict(self, face_image):
        """
        Predict emotion from face image.
        
        Args:
            face_image: Preprocessed face image (48x48x3)
        
        Returns:
            predictions: Array of probabilities for each emotion
        """
        if len(face_image.shape) == 3:
            face_image = np.expand_dims(face_image, axis=0)
        
        predictions = self.model.predict(face_image, verbose=0)
        return predictions[0]
    
    def load_model(self, model_path):
        """Load a trained model from file."""
        self.model = keras.models.load_model(
            model_path,
            custom_objects={'focal_loss_fixed': self._focal_loss()}
        )
    
    def save_model(self, model_path):
        """Save the model to file."""
        self.model.save(model_path)
    
    def get_model_summary(self):
        """Print model architecture summary."""
        if self.model:
            return self.model.summary()
        else:
            return "Model not built yet. Call build_model() first."


# Utility function for creating data augmentation
def get_data_augmentation():
    """
    Create data augmentation layer for training.
    Specific to autism-friendly augmentation (no extreme transformations).
    """
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1)
    ])

