import cv2
import numpy as np
from models.face_emotion_model import FaceEmotionModelTL
from utils.data_preprocessing import preprocess_face_image
from config import config

model = FaceEmotionModelTL()
model.load_model(config.FACE_MODEL_PATH)

def predict_emotion(frame):
    # Detect face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = detector.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return "no_face", 0.0

    x, y, w, h = faces[0]
    face = frame[y:y+h, x:x+w]
    pre = preprocess_face_image(face, (96, 96))

    pred = model.predict(pre)
    emotion_idx = np.argmax(pred)
    emotion = config.EMOTIONS[emotion_idx]
    confidence = pred[emotion_idx]

    return emotion, float(confidence)
