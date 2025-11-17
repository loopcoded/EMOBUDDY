# rebuild_voice_model.py

import tensorflow as tf
from models.voice_emotion_model import VoiceEmotionModel, AttentionLayer

WEIGHTS_PATH = "static/models/voice_emotion_model_best_weights.h5"
FULL_MODEL_PATH = "static/models/voice_emotion_model_full.keras"

NUM_CLASSES = 7
INPUT_SHAPE = (128, 128, 1)

def main():
    print("🔧 Rebuilding Voice Emotion Model Architecture...")

    # 1. Build architecture
    vem = VoiceEmotionModel(num_classes=NUM_CLASSES, input_shape=INPUT_SHAPE)
    model = vem.build_model()

    # 2. Compile (needed before loading weights)
    vem.compile_model(learning_rate=1e-4)

    # 3. Load weights file
    print("📥 Loading weights:", WEIGHTS_PATH)
    vem.load_weights(WEIGHTS_PATH)

    # 4. Save FULL model in .keras format
    print("💾 Saving full model:", FULL_MODEL_PATH)
    model.save(FULL_MODEL_PATH)

    print("\n🎉 DONE — Voice model fully rebuilt and saved!")

if __name__ == "__main__":
    main()
