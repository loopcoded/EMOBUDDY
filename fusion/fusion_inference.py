import numpy as np
import tensorflow as tf

FACE_MODEL_PATH = "static/models/face_emotion_cnn_96_improved.keras"
from models.voice_emotion_model import AttentionLayer
VOICE_MODEL_PATH = "static/models/voice_emotion_model_full.keras"

voice_model = tf.keras.models.load_model(
    VOICE_MODEL_PATH,
    custom_objects={"AttentionLayer": AttentionLayer}
)


# Load both models
face_model = tf.keras.models.load_model(FACE_MODEL_PATH)

# Fusion weights (you can tune slightly)
VOICE_WEIGHT = 0.7
FACE_WEIGHT = 0.3

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def predict_face(face_image_96):
    """
    Expects preprocessed (96,96,3) image normalized to [0,1]
    """
    face_image_96 = np.expand_dims(face_image_96, axis=0)
    probs = face_model.predict(face_image_96)[0]
    return probs


def predict_voice(voice_features):
    """
    Expects MFCC / mel-spectrogram features preprocessed exactly like training
    """
    voice_features = np.expand_dims(voice_features, axis=0)
    probs = voice_model.predict(voice_features)[0]
    return probs


def fuse_predictions(face_probs, voice_probs):
    """
    Late fusion of softmax probabilities
    """
    final = (VOICE_WEIGHT * voice_probs) + (FACE_WEIGHT * face_probs)
    final = final / np.sum(final)  # normalize again, optional
    return final


def predict_emotion(face_img, voice_feats):
    """
    Main function used by your backend inference API
    """
    face_probs = predict_face(face_img)
    voice_probs = predict_voice(voice_feats)

    fused_probs = fuse_predictions(face_probs, voice_probs)
    emotion_idx = np.argmax(fused_probs)
    emotion = EMOTIONS[emotion_idx]

    return {
        "face_probs": face_probs.tolist(),
        "voice_probs": voice_probs.tolist(),
        "fused_probs": fused_probs.tolist(),
        "emotion": emotion
    }


if __name__ == "__main__":
    print("Fusion model ready.")
