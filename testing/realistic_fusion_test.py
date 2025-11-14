import numpy as np
import tensorflow as tf
from models.emotion_fusion_enhanced import FusionModel

fusion_model = tf.keras.models.load_model("static/models/emotion_fusion_model")

def predict_fused(face_embedding, voice_embedding):
    face_mask = np.array([[1]]) if face_embedding is not None else np.array([[0]])
    voice_mask = np.array([[1]]) if voice_embedding is not None else np.array([[0]])

    if face_embedding is None:
        face_embedding = np.zeros((1, fusion_model.input_shape[0][1]))
    if voice_embedding is None:
        voice_embedding = np.zeros((1, fusion_model.input_shape[1][1]))

    pred = fusion_model.predict({
        "face_embedding": face_embedding,
        "voice_embedding": voice_embedding,
        "face_mask": face_mask,
        "voice_mask": voice_mask,
    })

    return pred
