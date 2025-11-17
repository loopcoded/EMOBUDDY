# models/face_emotion_model.py
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers

# Enable GPU memory growth + mixed precision (best for NVIDIA GPUs)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs detected: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("⚠ No GPU detected. Using CPU.")


class ImprovedFaceCNN:
    """
    Strong CNN for 96x96 RGB face images.
    - Built-in normalization (Rescaling 1/255)
    - Light data augmentation
    - 4 Conv blocks with BN + Dropout
    - GlobalAveragePooling + Dense head
    """

    def __init__(self, input_shape=(96, 96, 3), num_classes=7):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def _conv_block(self, x, filters, name):
        x = layers.Conv2D(filters, (3, 3), padding="same",
                          use_bias=False, name=f"{name}_conv1")(x)
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)
        x = layers.Activation("relu", name=f"{name}_act1")(x)

        x = layers.Conv2D(filters, (3, 3), padding="same",
                          use_bias=False, name=f"{name}_conv2")(x)
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)
        x = layers.Activation("relu", name=f"{name}_act2")(x)

        x = layers.MaxPooling2D((2, 2), name=f"{name}_pool")(x)
        x = layers.Dropout(0.25, name=f"{name}_drop")(x)
        return x

    def build(self):
        inputs = keras.Input(shape=self.input_shape, name="face_input")

        # Normalization to [0,1]
        x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

        # Lightweight augmentation (only active in training)
        aug = keras.Sequential(
            [
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.08),
                layers.RandomZoom(0.1),
            ],
            name="augmentation",
        )
        x = aug(x)

        # Conv blocks
        x = self._conv_block(x, 32, "block1")
        x = self._conv_block(x, 64, "block2")
        x = self._conv_block(x, 128, "block3")
        x = self._conv_block(x, 256, "block4")

        # Global pooling + dense head
        x = layers.GlobalAveragePooling2D(name="gap")(x)

        x = layers.Dense(256, activation="relu", name="dense_256")(x)
        x = layers.Dropout(0.5, name="drop_256")(x)

        x = layers.Dense(128, activation="relu", name="dense_128")(x)
        x = layers.Dropout(0.4, name="drop_128")(x)

        outputs = layers.Dense(
            self.num_classes,
            activation="softmax",
            name="emotion"
        )(x)

        model = models.Model(inputs=inputs, outputs=outputs, name="ImprovedFaceCNN96")
        return model
