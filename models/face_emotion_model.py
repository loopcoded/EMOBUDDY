# models/face_emotion_model.py
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import numpy as np

# -----------------------------
# Custom Focal Loss Definition
# -----------------------------
class FocalLoss(keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        if self.label_smoothing > 0:
            K = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1. - self.label_smoothing) + self.label_smoothing / K

        eps = keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        cross = -y_true * tf.math.log(y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)
        focal_weight = tf.pow(1. - pt, self.gamma)
        loss = self.alpha * focal_weight * tf.reduce_sum(cross, axis=-1)
        return loss

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing
        })
        return config


# -----------------------------
# Face Emotion Transfer Model
# -----------------------------
class FaceEmotionModelTL:
    def __init__(self, num_classes=7, input_shape=(96, 96, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None

    def build_model(self, backbone_trainable=False):
        base = keras.applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet"
        )
        base.trainable = backbone_trainable

        inputs = keras.Input(shape=self.input_shape, name="input_image")
        # Images are already normalized in preprocessing
        x = base(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(
            256, activation="relu",
            kernel_regularizer=keras.regularizers.l2(1e-4)
        )(x)
        x = layers.Dropout(0.4)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = models.Model(inputs, outputs)
        return self.model

    def compile_model(self, lr=1e-3, label_smoothing=0.0, use_focal_loss=False):
        opt = keras.optimizers.Adam(learning_rate=lr)
        if use_focal_loss:
            loss_fn = FocalLoss(gamma=2.0, alpha=0.25, label_smoothing=label_smoothing)
        else:
            loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

        self.model.compile(
            optimizer=opt,
            loss=loss_fn,
            metrics=[
                keras.metrics.CategoricalAccuracy(name="accuracy"),
                keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy")
            ]
        )

    def unfreeze_top_layers(self, num_layers=30, fine_tune_lr=1e-4):
        """Unfreeze last N layers of backbone for fine-tuning."""
        base_model = self.model.layers[1]  # MobileNetV2 is layer 1 now
        base_model.trainable = True
        
        # Freeze all layers first
        for layer in base_model.layers:
            layer.trainable = False
        
        # Unfreeze last N layers (except BatchNorm)
        unfreeze_from = len(base_model.layers) - num_layers
        for layer in base_model.layers[unfreeze_from:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True
        
        print(f"✅ Unfroze top {num_layers} layers of MobileNetV2 for fine-tuning.")
        self.compile_model(lr=fine_tune_lr, use_focal_loss=True, label_smoothing=0.1)

    def get_callbacks(self, model_path, patience=6):
        return [
            keras.callbacks.ModelCheckpoint(
                model_path, 
                monitor="val_accuracy", 
                save_best_only=True, 
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor="val_accuracy", 
                patience=patience, 
                restore_best_weights=True, 
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", 
                factor=0.5, 
                patience=3, 
                min_lr=1e-7, 
                verbose=1
            ),
        ]

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = keras.models.load_model(
            path, 
            custom_objects={"FocalLoss": FocalLoss}
        )

    def predict(self, x):
        if x.ndim == 3:
            x = np.expand_dims(x, 0)
        # Normalize if not already normalized
        if x.max() > 1.0:
            x = x.astype(np.float32) / 255.0
        return self.model.predict(x, verbose=0)[0]