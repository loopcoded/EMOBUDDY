# models/face_emotion_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers

class FocalLoss(keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        if self.label_smoothing and self.label_smoothing > 0:
            num_classes = tf.cast(tf.shape(y_true)[-1], tf.float32)
            y_true = y_true * (1 - self.label_smoothing) + self.label_smoothing / num_classes

        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce = -y_true * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, self.gamma)
        focal_loss = self.alpha * focal_weight * ce
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=-1))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'alpha': self.alpha, 'gamma': self.gamma, 'label_smoothing': self.label_smoothing})
        return cfg


class FaceEmotionModelTL:
    def __init__(self, num_classes=7, input_shape=(224, 224, 3), backbone='mobilenetv2'):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.backbone_name = backbone
        self.base_model = None
        self.model = None

    def build_model(self, backbone_trainable=False, dropout=0.4):
        inputs = layers.Input(shape=self.input_shape, name='image_input')

        # Model expects raw [0,255] images; Rescaling maps to [-1,1] for ImageNet backbones
        x = layers.Rescaling(scale=1.0/127.5, offset=-1.0)(inputs)

        if self.backbone_name == 'efficientnetb0':
            base = keras.applications.EfficientNetB0(include_top=False, weights='imagenet',
                                                     input_shape=self.input_shape)
        elif self.backbone_name == 'mobilenetv3small':
            base = keras.applications.MobileNetV3Small(include_top=False, weights='imagenet',
                                                       input_shape=self.input_shape)
        else:
            base = keras.applications.MobileNetV2(include_top=False, weights='imagenet',
                                                  input_shape=self.input_shape)

        base.trainable = backbone_trainable
        self.base_model = base  # expose backbone so we can unfreeze later

        x = base(x, training=False)

        gap = layers.GlobalAveragePooling2D(name='gap')(x)
        gmp = layers.GlobalMaxPooling2D(name='gmp')(x)
        x = layers.Concatenate()([gap, gmp])

        x = layers.Dense(512, kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout)(x)

        x_residual = x

        x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout * 0.7)(x)

        x_residual_proj = layers.Dense(256)(x_residual)
        x = layers.Add()([x, x_residual_proj])

        x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(dropout * 0.5)(x)

        outputs = layers.Dense(self.num_classes, activation='softmax', name='emotion',
                               kernel_regularizer=regularizers.l2(0.001))(x)

        self.model = models.Model(inputs=inputs, outputs=outputs, name=f'face_{self.backbone_name}')
        return self.model

    def compile_model(self, lr=1e-3, label_smoothing=0.1, use_focal_loss=False):
        from tensorflow.keras.optimizers import Adam

        if use_focal_loss:
            loss = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=label_smoothing)
        else:
            loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

        self.model.compile(
            optimizer=Adam(learning_rate=lr),
            loss=loss,
            metrics=[
                'accuracy',
                keras.metrics.TopKCategoricalAccuracy(k=2, name='top_2_accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]
        )

    def get_callbacks(self, model_path, patience=10):
        # ModelCheckpoint will save in SavedModel format (folder) given no .keras extension
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False
        )

        early = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=patience,
            restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            verbose=1
        )

        return [checkpoint, early, reduce_lr]

    def unfreeze_backbone(self, fine_tune_at=None):
        """Unfreezes backbone layers. Provide fine_tune_at (int) to unfreeze from that layer index onward."""
        if self.base_model is None:
            raise ValueError("Base model not set. Call build_model() first.")
        # If fine_tune_at not provided, unfreeze last 30 layers
        if fine_tune_at is None:
            fine_tune_at = max(0, len(self.base_model.layers) - 30)
        for i, layer in enumerate(self.base_model.layers):
            layer.trainable = i >= fine_tune_at
        return

    def predict(self, face_image):
        import numpy as np, cv2
        if isinstance(face_image, np.ndarray):
            if face_image.ndim == 2:
                face_image = np.stack([face_image] * 3, axis=-1)
            if face_image.ndim == 3:
                face_image = np.expand_dims(face_image, 0)
            if face_image.shape[1:3] != self.input_shape[:2]:
                resized = cv2.resize(face_image[0], (self.input_shape[1], self.input_shape[0]))
                face_image = np.expand_dims(resized, 0)
        preds = self.model.predict(face_image, verbose=0)
        return preds[0]

    def load_model(self, model_path):
        self.model = keras.models.load_model(model_path, custom_objects={'FocalLoss': FocalLoss})

    def save_model(self, model_path):
        self.model.save(model_path)

    def get_model_summary(self):
        if self.model:
            return self.model.summary()
        print("Model not built yet.")
