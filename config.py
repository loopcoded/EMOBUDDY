import os

class Config:
    """Base configuration settings."""
    # General Config
    SECRET_KEY = os.environ.get('SECRET_KEY', 'a_secret_key_for_emobuddy')
    DEBUG = False
    TESTING = False
    HOST = '127.0.0.1'
    PORT = 5000
    
    # Emotion Definitions
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    MODEL_EMOTIONS = EMOTIONS  # Alias for compatibility
    NUM_EMOTIONS = len(EMOTIONS)

    # Model Paths - FIXED: Use .keras format (TensorFlow 2.x standard)
    FACE_MODEL_PATH = 'static/models/face_emotion_model_best.keras'
    VOICE_MODEL_PATH = 'static/models/voice_emotion_model_best'
    
    # Real-time Monitoring Parameters
    FACE_DETECTION_INTERVAL = 0.5  # Process face every 0.5 seconds
    VOICE_DETECTION_INTERVAL = 1.0 # Process voice every 1.0 second (for 3s chunk)
    EMOTION_SMOOTHING_WINDOW = 10 # Number of predictions to average
    CONFIDENCE_THRESHOLD = 0.65  # Minimum confidence for strong classification
    MAX_EMOTION_HISTORY = 300 # Max entries in history queue

    # Face Model Parameters
    FACE_IMAGE_SIZE = (48, 48)

    # Voice Model Parameters (Matching data_preprocessing.py)
    VOICE_SAMPLE_RATE = 22050
    VOICE_DURATION = 3 # seconds
    VOICE_N_MFCC = 40
    VOICE_N_MEL = 128
    
    # --- Intervention Logic (Critical for Autistic Kids) ---
    # Levels and the actions they trigger
    INTERVENTION_LEVELS = {
        'green': {
            'emotions': ['neutral', 'happy'],
            'threshold': 0.0,
            'action': 'continue_study',
            'duration': 0
        },
        'yellow': {
            'emotions': ['surprise'],
            'threshold': 0.5,
            'action': 'gentle_prompt_or_break',
            'duration': 60  # 1 minute gentle break suggestion
        },
        'orange': {
            'emotions': ['disgust', 'sad'],
            'threshold': 0.75,
            'action': 'switch_to_calming_game',
            'duration': 180 # 3 minutes calming game
        },
        'red': {
            'emotions': ['fear', 'angry'],
            'threshold': 0.9,
            'action': 'immediate_parent_alert_or_stop',
            'duration': 0 # Stop study immediately
        }
    }


class DevelopmentConfig(Config):
    """Development-specific settings."""
    DEBUG = True

class ProductionConfig(Config):
    """Production-specific settings."""
    DEBUG = False
    # In a real app, HOST/PORT would be configured externally or via Gunicorn

def get_config(env):
    """Helper function to get the correct config class."""
    if env == 'production':
        return ProductionConfig
    return DevelopmentConfig

# Create a global config instance for easy import in other files
config = DevelopmentConfig()